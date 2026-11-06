// 四路网页版入口. 用法:
//
//   ./traffic_web_quad 8081 test1 test2 test3 test4
//   ./traffic_web_quad 8081 test1 test2 test3 test4 20     ← 每路送帧帧率(默认 20)
//
// 和 traffic_web_single 的区别:
//   single : 一路, 网页下命令才开始跑, 跑完退出
//   quad   : 四路, 开机就跑, 视频循环播, 计数每轮清零
//            一次只推一路的画面(网页可切换), 但四路都在后台跑
//
// 推理/跟踪/计数逻辑全部复用现有代码(tracker.cpp / counter.cpp / rknn_detector.cpp),
// 这份只管调度:
//
//   每路 4 个线程:
//     reader  : 循环读视频, 按固定 fps 节奏送帧进队列, 播完一轮 epoch+1
//     infer x2: 各带独立 detector(绑不同 NPU 核), 从本路队列取帧推理
//     track   : 按帧号顺序取结果 -> 跟踪 -> 计数 -> 画框 -> 推给 WebServer
//
//   为什么要帧号: 2 个 infer 线程并行, 完成顺序不一定, 必须按帧号排回去,
//   否则第 5 帧的结果会画到第 3 帧的画面上, 框会跳到别的车上.
//
//   为什么要 epoch: 视频循环播, 每播完一轮计数要清零.
//   reader 播完一轮就 epoch+1, track 线程看到 epoch 变了就重建跟踪器 + 清计数器.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
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
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "counter.hpp"
#include "tracker.hpp"
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

// 每路几个推理线程(每个线程一个独立 detector, RKNN context 不能多线程共用)
constexpr int kInferPerCam = 2;
// 每路队列里最多存几帧(存整图, 帧数太多吃内存)
constexpr int kQueueDepth = 8;

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

bool video_ok(const std::string& path) {
  std::error_code ec;
  return fs::exists(path, ec) && fs::file_size(path, ec) > 1024;
}

std::string resolve_video_path(const std::string& video, const std::string& cfg_path) {
  if (video.empty()) return "";
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
};

// ---------------------- 读帧线程: 循环播 ----------------------

void reader_loop(Cam& c, int fps_limit, WebServer& server) {
  while (c.alive) {
    cv::VideoCapture cap;
    cap.open(c.video_path);
    if (!cap.isOpened()) {
      std::fprintf(stderr, "[%s] 打不开视频: %s\n", c.name.c_str(), c.video_path.c_str());
      std::this_thread::sleep_for(std::chrono::seconds(2));
      continue;
    }
    const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
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

    int64_t n = 0;
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
      if (!cap.read(frame) || frame.empty()) break;

      // 按固定节奏送帧, 等价于摄像头推流(不管源视频多少 fps)
      if (fps_limit > 0) {
        const auto target = t_start + std::chrono::microseconds(
            static_cast<long long>(n * 1000000.0 / fps_limit));
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

    // 播完一轮 -> epoch+1, track 线程看到这个变化就清空计数重新开始
    c.epoch.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(c.mu);
      c.q_in.clear();
      c.q_done.clear();
    }
    c.cv.notify_all();
    if (restart) c.restart.store(false);
  }
}

// ---------------------- 推理线程 ----------------------

void infer_loop(Cam& c, int slot, WebServer& server) {
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
        det->set_core((c.id * kInferPerCam + slot) % 3);
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

void track_loop(Cam& c, WebServer& server) {
  Tracker tracker;
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
      tracker = Tracker();
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
    c.tracks.store(static_cast<int>(tracks.size()));

    // 计数: 只有 moving 的才数(静止等红灯的车不数)
    for (const auto& tr : tracks) {
      if (!tr.moving) continue;
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
          e.snap = save_snap(*res.full, hits[i].x, hits[i].y, tag, tracks, col);
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
      draw_tracks(out, tracks, c.cfg, ov);
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
      server.update_stats(ls, static_cast<int>(tracks.size()), c.fps.load(),
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

  int port = 8081;
  int fps_limit = 20;

  // 位置参数: [端口] 配置1 配置2 配置3 配置4 [每路fps]
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
    std::vector<std::thread> threads;

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
        std::fprintf(stderr, "[%s] 视频不存在: %s\n", c.name.c_str(), c.video_path.c_str());
        continue;
      }
    }

    // 线程启停: 结束后能"重新开始"再拉起来
    auto start_threads = [&]() {
      for (auto& t : threads) if (t.joinable()) t.join();
      threads.clear();
      for (int i = 0; i < kMaxCams; ++i) {
        Cam& c = cams[i];
        if (c.video_path.empty()) continue;
        c.alive = true;
        c.epoch = 0;
        c.restart = false;
        threads.emplace_back(reader_loop, std::ref(c), fps_limit, std::ref(server));
        for (int s = 0; s < kInferPerCam; ++s) {
          threads.emplace_back(infer_loop, std::ref(c), s, std::ref(server));
        }
        threads.emplace_back(track_loop, std::ref(c), std::ref(server));
      }
    };

    auto stop_threads = [&]() {
      for (int i = 0; i < kMaxCams; ++i) cams[i].alive = false;
      for (auto& t : threads) if (t.joinable()) t.join();
      threads.clear();
    };

    start_threads();

    std::printf("\n四路网页版已启动(静默模式, 状态看网页)\n");
    std::printf("  监控台   http://192.168.1.105:%d\n", port);
    std::printf("  单路画面 http://192.168.1.105:%d/?cam=0\n", port);
    std::printf("  每路 %d 推理线程, 送帧 %d fps, Ctrl+C 退出\n\n", kInferPerCam, fps_limit);

    // 主线程: 处理网页命令(暂停/继续/结束/重新开始/单路重启)
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
      }

      if (quit) {
        stop_threads();
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
        start_threads();
        server.set_state(1);
        std::printf("已重新开始.\n");
      } else if (again) {
        // 直接点"重新开始"也能拉起来
        start_threads();
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
