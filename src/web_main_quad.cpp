// 四路网页版入口。用法：./traffic_web_quad 8081 test1 test2 test3 test4 [每路fps]
//
// 推理/跟踪/计数复用现有代码，本文件只管调度：每路 4 个线程（1 读帧 + 2 推理 + 1 跟踪）。
// 两个推理线程并行，结果按帧号排序再送跟踪；视频循环播，每播完一轮 epoch+1 清零计数。

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <pthread.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>   // cv::utils::logging::setLogLevel
#if defined(__linux__)
#include <dlfcn.h>                          // dlopen: 关 libav 的日志
#endif

#include "config.hpp"
#include "counter.hpp"
#include "tracker.hpp"
#ifdef TRAFFIC_USE_MPP
#include "rkmpp_reader.hpp"
#endif
#include "web_server.hpp"
#ifdef USE_RKNN
void rknn_set_verbose(bool on);
#endif
#ifdef USE_RKNN
#include "rknn_detector.hpp"
using Detector = RknnDetector;
#else
#include "detector.hpp"
#endif

namespace fs = std::filesystem;

namespace {

const std::string kRoot = TRAFFIC_PROJECT_ROOT;
const std::string kHtml = TRAFFIC_WEB_DIR;   // 编译期灌进来的是完整文件路径
#ifdef USE_RKNN
const std::string kModel = TRAFFIC_PROJECT_ROOT "/models/best_detrac.rknn";
#else
const std::string kModel = TRAFFIC_PROJECT_ROOT "/models/best_detrac.onnx";
#endif
const std::string kSnapDir = TRAFFIC_PROJECT_ROOT "/output/snapshots";

// TRAFFIC_MPP=0 时置 true, 让所有路都退回 OpenCV 软解
bool g_mpp_off = false;

// 每路几个推理线程(每个线程一个独立 detector, RKNN context 不能多线程共用)
constexpr int kInferPerCam = 2;
// 每路队列里最多存几帧(存整图, 帧数太多吃内存)
constexpr int kQueueDepth = 8;

// ---------------------- 日志降噪 ----------------------
// libavformat/libavcodec 出错会直接往 stderr 打, 视频流连不上时每 2 秒
// 一行 [tcp @ 0x...] Connection refused, 把有用的日志全冲掉.
// OpenCV 的 setLogLevel 管不到它们, 这里用 dlopen 直接动 libav:
//   1) av_log_set_level(-8)  只设级别
//   2) av_log_set_callback() 换成空函数 <- 真正起作用的是这个,
//      OpenCV 的 ffmpeg 后端自己装过回调, 光设级别压不住
// 每次 cap.open() 之后要再调一遍 av_quiet(), 因为它会把回调改回去.
#if defined(__linux__)
namespace {
void av_quiet_cb(void*, int, const char*, va_list) {}   // 什么都不打
using AvSetLevel = void (*)(int);
using AvSetCb = void (*)(void (*)(void*, int, const char*, va_list));

// 拿一次句柄就够, 之后复用. 拿不到就返回 false, 不影响功能.
AvSetLevel g_set_level = nullptr;
AvSetCb g_set_cb = nullptr;

void av_quiet() {
  if (!g_set_cb) {
    static const char* kNames[] = {
        "libavutil.so",    "libavutil.so.60", "libavutil.so.59",
        "libavutil.so.58", "libavutil.so.57", "libavutil.so.56"};
    for (const char* nm : kNames) {
      void* h = ::dlopen(nm, RTLD_LAZY | RTLD_NOLOAD);
      if (!h) continue;
      g_set_level = reinterpret_cast<AvSetLevel>(::dlsym(h, "av_log_set_level"));
      g_set_cb = reinterpret_cast<AvSetCb>(::dlsym(h, "av_log_set_callback"));
      break;   // 句柄不关: 关了符号地址也失效
    }
  }
  if (g_set_level) g_set_level(-8);        // AV_LOG_QUIET
  if (g_set_cb) g_set_cb(av_quiet_cb);
}
}  // namespace
#else
void av_quiet() {}
#endif

// ---------------------- 路径解析 ----------------------

std::string resolve_config_path(const std::string& arg) {
  if (arg.empty()) return "";
  fs::path p(arg);
  if (p.extension() == ".json") {
    std::error_code ec;
    if (fs::exists(p, ec)) return p.string();
  }
  std::error_code ec;
  if (fs::exists(arg, ec)) return arg;
  const std::string cand = kRoot + "/configs/" + arg + ".json";
  if (fs::exists(cand, ec)) return cand;
  return arg + ".json";
}

// rtsp:// / http:// / https:// 这类不是文件: 不能查存在性, 也不能拼目录.
bool is_stream_url(const std::string& p) {
  return p.rfind("rtsp://", 0) == 0 || p.rfind("rtsps://", 0) == 0 ||
         p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0;
}

bool video_ok(const std::string& path) {
  if (is_stream_url(path)) return true;   // 流要真连上才知道通不通, 这里只放行
  std::error_code ec;
  return fs::exists(path, ec) && fs::file_size(path, ec) > 1024;
}

std::string resolve_video_path(const std::string& video, const std::string& cfg_path) {
  if (video.empty()) return "";
  if (is_stream_url(video)) return video;   // 流地址原样用, 不参与本地路径拼装
  if (video[0] == '/') {
    if (video_ok(video)) return video;
  } else {
    const fs::path cfg_dir = fs::path(cfg_path).parent_path();
    const fs::path cand = cfg_dir / video;
    if (video_ok(cand.string())) return cand.string();
    const std::string cand2 = kRoot + "/data/videos/" + video;
    if (video_ok(cand2)) return cand2;
  }
  return video;
}

cv::Scalar to_scalar(const std::array<int, 3>& c) { return cv::Scalar(c[0], c[1], c[2]); }

std::vector<cv::Point> to_cv_poly(const std::vector<Point>& pts) {
  std::vector<cv::Point> p;
  p.reserve(pts.size());
  for (const auto& q : pts) p.emplace_back(q.x, q.y);
  return p;
}

// ---------------------- 画图 ----------------------

void draw_tracks(cv::Mat& frame, const std::vector<TrackView>& tracks, const Config& cfg,
                 const Overlay& ov) {
  if (!ov.box && !ov.id) return;
  for (const auto& tr : tracks) {
    const cv::Rect2f& b = tr.box;
    const cv::Point tl(cvRound(b.x), cvRound(b.y));
    const cv::Point br(cvRound(b.x + b.width), cvRound(b.y + b.height));
    if (ov.box) {
      cv::rectangle(frame, cv::Rect(tl, br), to_scalar(cfg.draw.box_color), cfg.draw.box_width);
      const int cx = cvRound(b.x + b.width * 0.5f);
      const int cy = cvRound(b.y + b.height * 0.5f);
      const int r = 7;
      const cv::Scalar kCross(0, 255, 255);
      cv::line(frame, cv::Point(cx - r, cy), cv::Point(cx + r, cy), kCross, 1);
      cv::line(frame, cv::Point(cx, cy - r), cv::Point(cx, cy + r), kCross, 1);
    }
    if (ov.id) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "#%lld", tr.id);
      cv::putText(frame, buf, tl, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  to_scalar(cfg.draw.text_color), cfg.draw.text_width, cv::LINE_AA);
    }
  }
}

void draw_counters(cv::Mat& frame, const Config& cfg, const Counter& counter) {
  for (size_t i = 0; i < cfg.lines.size(); ++i) {
    const Line& L = cfg.lines[i];
    cv::line(frame, cv::Point(L.a.x, L.a.y), cv::Point(L.b.x, L.b.y),
             to_scalar(cfg.draw.line_color), cfg.draw.line_width);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %d", L.name.c_str(), counter.count(i));
    cv::Point tp = L.has_label_at ? cv::Point(static_cast<int>(L.label_at.x),
                                              static_cast<int>(L.label_at.y))
                                  : cv::Point(static_cast<int>(L.a.x), static_cast<int>(L.a.y));
    cv::putText(frame, buf, tp, cv::FONT_HERSHEY_SIMPLEX, cfg.draw.counter_scale,
                to_scalar(cfg.draw.text_color), cfg.draw.text_width, cv::LINE_AA);
  }
}

// ---------------------- 过线抓拍(跟单路版一致) ----------------------

std::string save_snap(const cv::Mat& frame, int cx, int cy, const std::string& tag,
                      const std::vector<TrackView>& live, const cv::Scalar& color) {
  const int half = 64;
  const int x = std::max(0, cx - half);
  const int y = std::max(0, cy - half);
  const int w = std::min(frame.cols - x, half * 2);
  const int h = std::min(frame.rows - y, half * 2);
  if (w <= 0 || h <= 0) return std::string();
  cv::Mat patch = frame(cv::Rect(x, y, w, h)).clone();

  // 过线记录里的坐标就是框中心, 找中心最近的那个 track, 在裁剪图上画它的框
  const TrackView* best = nullptr;
  double best_d = 1e9;
  for (const auto& tr : live) {
    const cv::Point2f c = center_of(tr.box);
    const double d = std::hypot(c.x - cx, c.y - cy);
    if (d < best_d) { best_d = d; best = &tr; }
  }
  if (best && best_d < 8.0) {
    const cv::Rect r(cvRound(best->box.x) - x, cvRound(best->box.y) - y,
                     cvRound(best->box.width), cvRound(best->box.height));
    cv::rectangle(patch, r & cv::Rect(0, 0, w, h), color, 1);
  }
  const std::string name = tag + ".jpg";
  if (!cv::imwrite(kSnapDir + "/" + name, patch)) return std::string();
  return name;
}

// ---------------------- 一路的运行时状态 ----------------------

struct CamJob {
  int64_t frame_no = 0;
  int epoch = 0;
  std::shared_ptr<cv::Mat> full;    // 整图(画框用)
  cv::Mat roi_img;                   // ROI 裁剪后的图(推理用)
};

struct CamRes {
  int64_t frame_no = 0;
  int epoch = 0;
  std::vector<Detection> dets;
  double infer_ms = 0.0;
  std::shared_ptr<cv::Mat> full;     // 整图跟着结果走, track 线程拿来画框
};

struct Cam {
  int id = 0;
  std::string name;
  Config cfg;
  std::string video_path;
  cv::Rect roi_rect;
  std::vector<cv::Point> poly;

  std::atomic<bool> alive{true};
  std::atomic<int> epoch{0};
  std::atomic<bool> restart{false};
  std::atomic<int64_t> frame_no{0};
  std::atomic<int> tracks{0};
  std::atomic<double> fps{0.0};
  std::atomic<double> infer_ms{0.0};

  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::shared_ptr<CamJob>> q_in;
  std::map<int64_t, CamRes> q_done;

#ifdef TRAFFIC_USE_MPP
  MppReader mpp;                 // 视频流走它, 本地文件不用
#endif
};

// ---------------------- 读帧线程: 循环播 ----------------------

void reader_loop(Cam& c, int fps_limit, WebServer& server) {
  const bool is_stream = is_stream_url(c.video_path);
  // 实时流重连后要接着编号: 一重连就从 0 开始的话, 跟踪线程等的还是旧帧号,
  // 得靠丢帧兜底(连续 5 次等不到)才能缓过来, 那几帧就白等了.
  int64_t n = 0;
  // 连不上就退避重试: 2s, 4s, 6s ... 最多 10s.
  // 固定 2 秒会刷屏(OpenCV 的 GStreamer 后端连不上时自己也往 stderr 打一堆),
  // 而且地址填错时根本没人看, 慢点重试反而省 CPU.
  int fails = 0;
  if (is_stream) {
    std::printf("[%s] 视频流输入: %s\n", c.name.c_str(), c.video_path.c_str());
  }
  while (c.alive) {
    cv::VideoCapture cap;
    bool use_mpp = false;
#ifdef TRAFFIC_USE_MPP
    // 视频流优先用硬件解码: 拉流仍交给 ffmpeg -c:v copy, 只有解码这一步换成 VPU.
    // 本地文件不占 CPU 大头, 而且 OpenCV 读文件本来就快, 保持原样.
    if (is_stream && !g_mpp_off && MppReader::available()) {
      use_mpp = c.mpp.open(c.video_path, 10000);
      if (!use_mpp) {
        std::fprintf(stderr, "[%s] 硬件解码没打开(%s), 这次退回 OpenCV 软解\n",
                     c.name.c_str(), c.mpp.last_error().c_str());
      }
    }
#endif
    if (!use_mpp) {
      cap.open(c.video_path);
      av_quiet();        // OpenCV 开流时会改 libav 的回调, 这里按回去
    }
    if (!use_mpp && !cap.isOpened()) {
      // 归零: 否则网页上停在上一次的帧率, 看着像还在跑, 实际一帧都没进来
      c.fps = 0.0;
      ++fails;
      if (fails == 1 || fails % 15 == 0) {
        std::fprintf(stderr, "[%s] 打不开%s: %s (%d 次了, 继续重试)\n",
                     c.name.c_str(), is_stream ? "视频流" : "视频",
                     c.video_path.c_str(), fails);
      }
      const int backoff = (fails < 5) ? fails * 2 : 10;
      for (int k = 0; k < backoff * 10 && c.alive; ++k) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;
    }
    fails = 0;
    int w = 0, h = 0;
#ifdef TRAFFIC_USE_MPP
    if (use_mpp) {
      w = c.mpp.width();
      h = c.mpp.height();
    }
#endif
    if (w <= 0 || h <= 0) {
      w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
      h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    }
    // 送帧节奏: 命令行没给 fps 就按源自己的帧率走, 和视频流完全一致.
    // 不能不限速: 不限速会用最快速度读完, 队列一满就丢帧, 后面的帧全乱.
    const double src_fps = use_mpp ? 0.0 : cap.get(cv::CAP_PROP_FPS);
    const double pace = (fps_limit > 0) ? static_cast<double>(fps_limit)
                                        : (src_fps > 1.0 ? src_fps : 0.0);
    if (w > 0 && h > 0) {
      c.roi_rect = cv::Rect(0, 0, w, h);
      c.poly.clear();
      if (c.cfg.has_roi && c.cfg.roi.size() >= 3) {
        c.poly = to_cv_poly(c.cfg.roi);
        c.roi_rect = cv::boundingRect(c.poly) & cv::Rect(0, 0, w, h);
        if (c.roi_rect.width <= 0 || c.roi_rect.height <= 0) {
          c.roi_rect = cv::Rect(0, 0, w, h);
          c.poly.clear();
        }
      }
    }

    if (!is_stream) n = 0;   // 文件每次从头播, 帧号也从头计; 实时流不重置
    auto t_start = std::chrono::steady_clock::now();
    cv::Mat frame;
    bool restart = false;
    bool was_paused = false;
    while (c.alive) {
      // 暂停时连视频都不读: 队列保持原样, 恢复后帧号连续, 画面定格在最后一帧
      if (server.state() == 2) {
        was_paused = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      // 从暂停恢复: 重新计时. 否则 sleep_until 的目标时间都已过去,
      // 会一路狂奔补帧, CPU 飙满、fps 显示乱跳.
      if (was_paused) {
        was_paused = false;
        t_start = std::chrono::steady_clock::now();
        n = 0;
      }
      if (c.restart.load()) { restart = true; break; }
      if (use_mpp) {
#ifdef TRAFFIC_USE_MPP
        if (!c.mpp.read(frame) || frame.empty()) break;
#endif
      } else if (!cap.read(frame) || frame.empty()) {
        break;
      }

      // 按固定节奏送帧, 等价于摄像头推流(不管源视频多少 fps).
      // 实时流不控节奏: 摄像头本来就是按自己的帧率送, 再限速只会让缓冲越积越多、延迟越来越大.
      if (pace > 0.5 && !is_stream) {
        const auto target = t_start + std::chrono::microseconds(
            static_cast<long long>(n * 1000000.0 / pace));
        std::this_thread::sleep_until(target);
      }

      auto job = std::make_shared<CamJob>();
      job->frame_no = n;
      job->epoch = c.epoch.load();
      job->full = std::make_shared<cv::Mat>(frame.clone());
      job->roi_img = frame(c.roi_rect).clone();

      {
        std::lock_guard<std::mutex> lk(c.mu);
        c.q_in.push_back(std::move(job));
        while (c.q_in.size() > kQueueDepth) c.q_in.pop_front();
      }
      c.cv.notify_all();
      ++n;
      c.frame_no.store(n);
    }
    cap.release();
#ifdef TRAFFIC_USE_MPP
    c.mpp.release();
#endif
    if (restart) c.restart.store(false);

    if (is_stream) {
      // 实时流"读完"= 断流, 不是播完. 计数不能清, 一清前面数的就白数了.
      // 队列里残留的帧作废, 重连后帧号接着往下走.
      std::fprintf(stderr, "[%s] 视频流断开, 2 秒后重连\n", c.name.c_str());
      {
        std::lock_guard<std::mutex> lk(c.mu);
        c.q_in.clear();
        c.q_done.clear();
      }
      c.cv.notify_all();
      for (int k = 0; k < 20 && c.alive; ++k) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;
    }

    // 播完一轮 -> epoch+1, track 线程看到这个变化就清空计数重新开始
    c.epoch.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(c.mu);
      c.q_in.clear();
      c.q_done.clear();
    }
    c.cv.notify_all();
  }
}

// ---------------------- 推理线程 ----------------------

void infer_loop(Cam& c, int slot, int core, WebServer& server) {
  std::unique_ptr<Detector> det;
  float det_conf = -1.f;
  float det_iou = -1.f;

  while (c.alive) {
    // 参数变了就重建 detector(RKNN 的 conf/iou 是建的时候定死的)
    const float conf = server.conf(c.id);
    const float iou = server.iou(c.id);
    if (!det || conf != det_conf || iou != det_iou) {
      try {
        det = std::make_unique<Detector>(kModel, c.cfg.detect.imgsz, conf, iou);
        det->set_core(core);
        det_conf = conf;
        det_iou = iou;
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[%s] 建 detector 失败: %s\n", c.name.c_str(), e.what());
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }
    }

    // 暂停: 必须在"取 job 之前"停.
    // 取了再扔等于丢帧, 跟踪线程会一直等一个永远不来的帧号 -> 卡死.
    if (server.state() == 2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    std::shared_ptr<CamJob> job;
    {
      std::unique_lock<std::mutex> lk(c.mu);
      c.cv.wait(lk, [&] { return !c.q_in.empty() || !c.alive; });
      if (!c.alive) break;
      if (server.state() == 2) continue;   // 等锁期间可能刚好被暂停
      job = std::move(c.q_in.front());
      c.q_in.pop_front();
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Detection> dets = det->detect(job->roi_img);
    const auto t1 = std::chrono::steady_clock::now();

    // 推理用的是 ROI 裁剪图, 坐标换算回整图
    for (auto& d : dets) {
      d.x1 += c.roi_rect.x;
      d.x2 += c.roi_rect.x;
      d.y1 += c.roi_rect.y;
      d.y2 += c.roi_rect.y;
    }

    // 框中心落在 ROI 多边形外的直接丢掉.
    // 上面只按外接矩形裁的图: 多边形是梯形时, 斜边外侧那一片(路肩、天空、人行道)
    // 仍然在裁剪图里, 会被当成正常区域检测. 单路版一直有这一步, 四路版漏了.
    if (!c.poly.empty()) {
      dets.erase(std::remove_if(dets.begin(), dets.end(),
                                [&](const Detection& d) {
                                  const cv::Point2f ctr(
                                      static_cast<float>((d.x1 + d.x2) * 0.5),
                                      static_cast<float>((d.y1 + d.y2) * 0.5));
                                  return cv::pointPolygonTest(c.poly, ctr, false) < 0;
                                }),
                 dets.end());
    }

    CamRes res;
    res.frame_no = job->frame_no;
    res.epoch = job->epoch;
    res.dets = std::move(dets);
    res.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.full = std::move(job->full);

    {
      std::lock_guard<std::mutex> lk(c.mu);
      c.q_done[job->frame_no] = std::move(res);
      while (c.q_done.size() > static_cast<size_t>(kQueueDepth) * 2) c.q_done.erase(c.q_done.begin());
    }
    c.cv.notify_all();
  }
}

// ---------------------- 跟踪+计数+画框线程 ----------------------

// 按这一路的配置建跟踪器（阈值来自 cfg.track，没写就用 Tracker 的默认值）
Tracker make_tracker(const Config& cfg) {
  return Tracker(cfg.track.buffer, cfg.track.high, cfg.track.low,
                 cfg.track.new_thresh, cfg.track.match);
}

void track_loop(Cam& c, WebServer& server) {
  Tracker tracker = make_tracker(c.cfg);
  auto counter = std::make_unique<Counter>(c.cfg.lines);
  int64_t next_no = 0;
  int last_epoch = -1;
  int64_t hits_seen = 0;
  auto t_fps = std::chrono::steady_clock::now();
  int frames_in_window = 0;
  int miss = 0;   // 连续等不到 next_no 的次数, 用来判断"这帧被丢了" 

  while (c.alive) {
    // 循环重播 -> 清空现场
    const int ep = c.epoch.load();
    if (ep != last_epoch) {
      last_epoch = ep;
      tracker = make_tracker(c.cfg);
      counter = std::make_unique<Counter>(c.cfg.lines);
      next_no = 0;
      hits_seen = 0;
      miss = 0;
      {
        std::lock_guard<std::mutex> lk(c.mu);
        c.q_done.clear();
      }
      server.clear_events(c.id);
      std::printf("[%s] 新一轮, 计数清零\n", c.name.c_str());
    }

    // 暂停: 停下不处理, 画面定格在最后一帧
    // (必须在取结果之前, 否则会把结果丢掉导致帧号对不上)
    if (server.state() == 2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    CamRes res;
    {
      std::unique_lock<std::mutex> lk(c.mu);
      c.cv.wait_for(lk, std::chrono::milliseconds(200),
                    [&] { return c.q_done.count(next_no) > 0 || !c.alive; });
      if (!c.alive) break;
      if (!c.q_done.count(next_no)) {
        // 等不到这一帧: 可能是被队列上限挤掉了(暂停恢复时最常见).
        // 连续等 1 秒还是没有, 就跳到队列里最小的帧号, 绝不卡死.
        if (++miss >= 5 && !c.q_done.empty()) {
          next_no = c.q_done.begin()->first;
          miss = 0;
        }
        continue;
      }
      res = std::move(c.q_done[next_no]);
      c.q_done.erase(next_no);
      miss = 0;
    }
    ++next_no;

    // 跟踪
    const std::vector<TrackView> tracks = tracker.update(res.dets);

    // 只留"动过的"轨迹. 路牌、灌木、路灯、地面箭头这些静止目标也会被检测到,
    // 一旦进了跟踪器就会挂上编号; 它们不过线、不计数, 但会一直画在画面上,
    // 表现为画面上出现无对应车辆的框. 单路版一直保留该过滤, 四路版此前缺失.
    std::vector<TrackView> live;
    live.reserve(tracks.size());
    for (const auto& tr : tracks) {
      if (tr.moving) live.push_back(tr);
    }
    c.tracks.store(static_cast<int>(live.size()));

    // 计数: 只有 moving 的才数(静止等红灯的车不数)
    for (const auto& tr : live) {
      const cv::Point2f ctr = center_of(tr.box);
      counter->update(tr.id, Point{static_cast<double>(ctr.x), static_cast<double>(ctr.y)},
                      "vehicle", static_cast<int>(res.frame_no));
    }

    // 新增的过线记录 -> 事件列表 + 抓拍
    const auto& hits = counter->hits();
    if (static_cast<int>(hits.size()) > hits_seen) {
      for (size_t i = hits_seen; i < hits.size(); ++i) {
        Event e;
        e.frame = hits[i].frame;
        e.line = hits[i].line;
        e.cls = hits[i].cls;
        e.x = hits[i].x;
        e.y = hits[i].y;
        e.ts = static_cast<double>(res.frame_no) / 12.0;
        // 抓拍: 每路文件名带 cam 编号, 不会互相覆盖
        if (res.full && !res.full->empty()) {
          char tag[128];
          std::snprintf(tag, sizeof(tag), "cam%d_%06d_%s", c.id, hits[i].frame,
                        hits[i].line.c_str());
          const cv::Scalar col = to_scalar(c.cfg.draw.text_color);
          e.snap = save_snap(*res.full, hits[i].x, hits[i].y, tag, live, col);
        }
        server.add_event(e, c.id);
      }
      hits_seen = static_cast<int>(hits.size());
    }

    // 画框
    if (res.full && !res.full->empty()) {
      cv::Mat out = *res.full;
      const Overlay ov = server.overlay(c.id);
      if (ov.roi && !c.poly.empty()) {
        cv::polylines(out, c.poly, true, to_scalar(c.cfg.draw.roi_color), 2);
      }
      draw_tracks(out, live, c.cfg, ov);
      if (ov.line) draw_counters(out, c.cfg, *counter);

      // 算 fps
      ++frames_in_window;
      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - t_fps).count();
      if (dt >= 1.0) {
        c.fps.store(frames_in_window / dt);
        frames_in_window = 0;
        t_fps = now;
      }
      c.infer_ms.store(res.infer_ms);

      // 状态数字
      std::vector<LineStat> ls;
      for (size_t i = 0; i < c.cfg.lines.size(); ++i) {
        ls.push_back(LineStat{c.cfg.lines[i].name, counter->count(i)});
      }
      server.update_stats(ls, static_cast<int>(live.size()), c.fps.load(),
                          res.infer_ms, static_cast<int>(res.frame_no), c.id);

      // 推流
      server.push_frame(out, c.id);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // 行缓冲: 否则重定向到文件时 stdout 会被全缓冲, 链接迟迟不显示
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  // 关掉 OpenCV 与 FFmpeg 的日志. 视频流连不上时它们每次重试都打一行
  // GStreamer WARN 或 "[tcp @ ...] Connection refused", 一屏一屏地刷;
  // 上层已打印"打不开视频流", 此处不再重复输出.
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
  // libav 的日志 OpenCV 管不到, 用 dlopen 拿符号关掉.
  // 不直接链接 libavutil: 板端只装运行库, 无开发包, 链接会失败.
  av_quiet();

  // 两个环境变量, 只为了做 A/B 对比时不用重新编译:
  //   TRAFFIC_MPP=0         关掉硬件解码, 全走 OpenCV 软解 (默认 1)
  //   TRAFFIC_CV_THREADS=N  设 OpenCV 线程数 (默认不设, 用库自带的 8)
  // 正常使用一个都不用设.
  if (const char* e = std::getenv("TRAFFIC_MPP")) {
    if (e[0] == '0') g_mpp_off = true;
  }
  if (const char* e = std::getenv("TRAFFIC_CV_THREADS")) {
    const int t = std::atoi(e);
    if (t > 0) cv::setNumThreads(t);
  }

  int port = 8081;
  // 0 = 跟源帧率 (本地文件用文件自带帧率, 视频流不控节奏).
  // 传了第 2 个数字参数才限速, 老写法仍然兼容.
  int fps_limit = 0;

  // 位置参数: [端口] 配置1 配置2 配置3 配置4 [每路fps]
  // 仅需指定端口: 配置名默认取 test1~test4, 每路帧率跟随视频源.
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (!a.empty() && a[0] == '-') continue;      // 忽略开关
    pos.push_back(a);
  }
  auto is_num = [](const std::string& t) {
    return !t.empty() && std::isdigit(static_cast<unsigned char>(t[0]));
  };
  size_t p = 0;
  if (p < pos.size() && is_num(pos[p])) { port = std::atoi(pos[p].c_str()); ++p; }
  std::vector<std::string> cfg_names;
  while (p < pos.size() && cfg_names.size() < 4 && !is_num(pos[p])) {
    cfg_names.push_back(pos[p]);
    ++p;
  }
  if (p < pos.size() && is_num(pos[p])) { fps_limit = std::atoi(pos[p].c_str()); ++p; }
  while (cfg_names.size() < 4) cfg_names.push_back("test" + std::to_string(cfg_names.size() + 1));

  try {
#ifdef USE_RKNN
    rknn_set_verbose(false);   // 8 个 detector 的初始化日志太吵, 关掉
#endif
    WebServer server(port, kHtml, kRoot + "/configs", kSnapDir);
    server.start();
    server.set_state(1);

    std::array<Cam, kMaxCams> cams;
    // 每一路自己一组线程. 分开存是为了能单独起停: 某一路换视频源只动这一路,
    // 其它三路照常跑, 计数不受影响.
    std::array<std::vector<std::thread>, kMaxCams> cam_threads;

    for (int i = 0; i < kMaxCams; ++i) {
      Cam& c = cams[i];
      c.id = i;
      c.name = cfg_names[i];
      const std::string cfg_path = resolve_config_path(c.name);
      c.cfg = load_config(cfg_path);
      // 把这一路配置里的 conf/iou 灌给网页, 作为参数面板的初值.
      // 只写内存: 之后网页上怎么改都不会落回 config 文件,
      // 下次启动又从这里重新读, 这样"配置是底盘, 页面改动是临时覆盖".
      // 放在视频检查之前, 这样某路视频缺失时面板显示的仍是它的配置值.
      server.init_params(i, static_cast<float>(c.cfg.detect.conf),
                         static_cast<float>(c.cfg.detect.iou));
      c.video_path = resolve_video_path(c.cfg.video, cfg_path);
      if (!video_ok(c.video_path)) {
        // 此处不能用 continue 跳过整路, 否则该路无法在网页上重新接入.
        // 跳过了网页上就既不能看状态也不能改地址, 等于这一路彻底废掉.
        // 正确做法是留成空槽(idle), 等网页上填了地址再拉起来.
        std::fprintf(stderr, "[%s] 视频不存在, 留空: %s\n", c.name.c_str(),
                     c.video_path.c_str());
        c.video_path.clear();
      }
      // 让网页一开始就显示当前地址和配置名
      server.set_source(i, c.video_path, c.name);
      server.set_idle(i, c.video_path.empty());
    }

    // 正在跑几路 -> 每路分几个推理线程.
    // 1 路时一个核给 3 个线程才能喂满(单线程只能到 23 fps);
    // 2 路以上每路 2 个就够, 再多线程也是排队等 NPU.
    auto active_cams = [&]() {
      int n = 0;
      for (int i = 0; i < kMaxCams; ++i)
        if (!cams[i].video_path.empty()) ++n;
      return n;
    };
    auto infer_threads_for = [&](int active) {
      return (active <= 1) ? 3 : 2;
    };

    // 起一路: 读帧 1 + 推理 N + 跟踪 1
    auto start_one = [&](int i) {
      Cam& c = cams[i];
      if (c.video_path.empty()) return;
      if (!cam_threads[i].empty()) return;   // 已经在跑

      const int active = active_cams();
      const int n_infer = infer_threads_for(active);
      c.alive = true;
      c.epoch = 0;
      c.restart = false;
      c.frame_no = 0;
      c.tracks = 0;
      c.fps = 0.0;

      // 绑核: 全局轮着分 0/1/2, 不同路的线程尽量摊到三个核上
      static std::atomic<int> core_i{0};
      auto& th = cam_threads[i];
      // 线程起名字: 15 字符以内, top -H / 逐线程 CPU 一看就知道是谁在吃 CPU
      auto name_of = [](const char* kind, int cam, int slot) {
        char buf[16] = {0};
        if (slot >= 0) std::snprintf(buf, sizeof(buf), "cam%d_%s%d", cam, kind, slot);
        else           std::snprintf(buf, sizeof(buf), "cam%d_%s", cam, kind);
        return std::string(buf);
      };
      th.emplace_back([&c, &server, fps_limit, nm = name_of("rd", i, -1)]() {
        ::pthread_setname_np(::pthread_self(), nm.c_str());
        reader_loop(c, fps_limit, server);
      });
      for (int t = 0; t < n_infer; ++t) {
        const int core = core_i.fetch_add(1) % 3;
        const std::string nm = name_of("inf", i, t);
        th.emplace_back([&c, &server, t, core, nm]() {
          ::pthread_setname_np(::pthread_self(), nm.c_str());
          infer_loop(c, t, core, server);
        });
      }
      {
        const std::string nm = name_of("trk", i, -1);
        th.emplace_back([&c, &server, nm]() {
          ::pthread_setname_np(::pthread_self(), nm.c_str());
          track_loop(c, server);
        });
      }

      server.set_idle(i, false);
      std::printf("[%s] 启动: %s (%d 路在跑, 每路 %d 推理线程)\n", c.name.c_str(),
                  c.video_path.c_str(), active, n_infer);
    };

    auto stop_one = [&](int i) {
      Cam& c = cams[i];
      if (cam_threads[i].empty()) return;
      c.alive = false;
      // 必须叫醒所有在等的线程, 否则 join 会永远卡住:
      // 推理线程卡在 c.cv.wait(...) 上, 谓词里检查了 alive, 但没人 notify 就不会醒;
      // 读帧线程若正卡在打不开的流上, 也不会往队列里放东西来间接唤醒它.
      // 主线程 join 卡死的后果是整个服务不再响应换源, 帧号和计数停在出事那一刻.
      c.cv.notify_all();
      for (auto& t : cam_threads[i])
        if (t.joinable()) t.join();
      cam_threads[i].clear();
      {
        std::lock_guard<std::mutex> lk(c.mu);
        c.q_in.clear();
        c.q_done.clear();
      }
      server.set_idle(i, true);
      std::printf("[%s] 已停止\n", c.name.c_str());
    };

    // 换源: 停 -> 重载配置 -> 重新解析地址 -> 起(地址空就只停)
    auto restart_one = [&](int i) {
      Cam& c = cams[i];
      stop_one(i);

      const std::string want_cfg = server.source_config(i);
      if (!want_cfg.empty()) {
        c.name = want_cfg;
        try {
          c.cfg = load_config(resolve_config_path(want_cfg));
          server.init_params(i, static_cast<float>(c.cfg.detect.conf),
                             static_cast<float>(c.cfg.detect.iou));
        } catch (const std::exception& e) {
          std::fprintf(stderr, "[%s] 读配置失败: %s\n", want_cfg.c_str(), e.what());
        }
      }

      const std::string url = server.source(i);
      if (url.empty()) {
        c.video_path.clear();
        server.set_idle(i, true);
        return;
      }
      c.video_path = resolve_video_path(url, resolve_config_path(c.name));
      if (!video_ok(c.video_path)) {
        std::fprintf(stderr, "[%s] 地址无效: %s\n", c.name.c_str(), c.video_path.c_str());
        c.video_path.clear();
        server.set_idle(i, true);
        return;
      }
      start_one(i);
    };

    auto start_all = [&]() {
      for (int i = 0; i < kMaxCams; ++i) start_one(i);
    };
    auto stop_all = [&]() {
      for (int i = 0; i < kMaxCams; ++i) stop_one(i);
    };

    // 网页点"断开": 停线程, 但 server 里存的地址和配置不动.
    // 这样刷新页面还是原来填的那一路, 再点接入就回来了.
    auto halt_one = [&](int i) {
      stop_one(i);
      cams[i].video_path.clear();
      server.set_idle(i, true);
    };

    start_all();

    std::printf("\n四路网页版已启动(静默模式, 状态看网页)\n");
    std::printf("  监控台   http://192.168.1.105:%d\n", port);
    std::printf("  单路画面 http://192.168.1.105:%d/?cam=0\n", port);
    // 先存入局部变量再 printf, 避免实参指向已析构的临时对象.
    // 写成 printf("%s", (std::to_string(x) + " fps").c_str()) 是悬垂指针:
    // 临时 string 在 printf 之前就析构了, c_str() 指向已释放的内存.
    std::string pace_desc = (fps_limit > 0) ? (std::to_string(fps_limit) + " fps")
                                            : std::string("跟源帧率");
    std::printf("  送帧 %s, 线程数按在跑的路数自动定, Ctrl+C 退出\n\n", pace_desc.c_str());

    // 主线程: 处理网页命令(暂停/继续/结束/重新开始/单路重启) + 换视频源
    std::array<int, kMaxCams> last_gen{0, 0, 0, 0};
    for (int i = 0; i < kMaxCams; ++i) last_gen[i] = server.source_gen(i);

    while (true) {
      bool quit = false;
      bool again = false;
      for (int i = 0; i < kMaxCams; ++i) {
        const Command cmd = server.take_command(i);
        if (cmd == kCmdQuit) quit = true;
        else if (cmd == kCmdStart) again = true;
        else if (cmd == kCmdRestart) {
          cams[i].restart.store(true);
          std::printf("[%s] 重启这一路\n", cams[i].name.c_str());
        }
        // 网页改了视频源: 只重起这一路, 其它路照跑
        // 先处理"断开", 再处理"换源": 两个都来了就以后者为准
        if (server.take_stop(i)) halt_one(i);
        const int g = server.source_gen(i);
        if (g != last_gen[i]) {
          last_gen[i] = g;
          restart_one(i);
        }
      }

      if (quit) {
        stop_all();
        server.set_state(0);
        std::printf("已结束检测(网页仍可访问). 点\"重新开始\"恢复.\n");
        // 等"重新开始"
        while (true) {
          bool go = false;
          for (int i = 0; i < kMaxCams; ++i) {
            const Command c2 = server.take_command(i);
            if (c2 == kCmdStart) go = true;
          }
          if (go) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        start_all();
        server.set_state(1);
        std::printf("已重新开始.\n");
      } else if (again) {
        // 直接点"重新开始"也能拉起来
        start_all();
        server.set_state(1);
        std::printf("已重新开始.\n");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "启动失败: %s\n", e.what());
    return 1;
  }
  return 0;
}

