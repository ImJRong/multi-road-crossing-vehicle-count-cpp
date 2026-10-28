// 网页版入口. 用法:
//   ./traffic_web          监听 8080
//   ./traffic_web 9000     指定端口
//
// 跟 traffic_count 的区别:
//   traffic_count 是"给个配置就跑到底, 跑完退出";
//   traffic_web 是"起来之后蹲在那, 等网页下命令", 可以反复启停、换配置、改参数.
//
// 它不复制任何计数/跟踪/推理逻辑 —— counter.cpp / tracker.cpp / detector.cpp
// 都是直接调用的同一份代码, 所以两边跑同一段视频, 数字必须一样.
//
// 这里只多了三件事:
//   1. 会话管理: 什么时候建视频/模型, 什么时候销毁
//   2. 抓拍: 每过一辆车, 把它过线那一小块截下来存成 jpg, 给网页的事件列表用
//   3. 推帧: 画完的帧交给 WebServer 编码推流
//
// 参数(conf/iou)只改内存, 不动 configs/*.json:
//   启动一路时先用配置里的值当初始值(跟命令行版保持一致),
//   之后网页拖滑块才覆盖它. 进程重启就回到配置值.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "counter.hpp"
#include "tracker.hpp"
#include "web_server.hpp"
// 推理后端跟 traffic_count 一样, 由编译时的 USE_RKNN 决定
#ifdef USE_RKNN
#include "rknn_detector.hpp"
using Detector = RknnDetector;
#else
#include "detector.hpp"
#endif

namespace fs = std::filesystem;

namespace {

const std::string kRoot = TRAFFIC_PROJECT_ROOT;
// 页面文件属于这个 C++ 工程自己, 不在 TRAFFIC_PROJECT_ROOT 那边(那边只有配置和视频),
// 所以路径在编译期由 CMake 灌进来
const std::string kHtml = TRAFFIC_WEB_DIR;
// 两种后端用同一份权重, 只是文件格式不同
#ifdef USE_RKNN
const std::string kModel = TRAFFIC_PROJECT_ROOT "/models/best_detrac.rknn";
#else
const std::string kModel = TRAFFIC_PROJECT_ROOT "/models/best_detrac.onnx";
#endif

// 抓拍图存这儿, Windows 里可以直接打开看
const std::string kSnapDir = TRAFFIC_PROJECT_ROOT "/output/snapshots";

// ---------------------- 路径解析(逻辑同 main.cpp) ----------------------

std::string resolve_config_path(const std::string& arg) {
  if (arg.find('/') != std::string::npos) return arg;
  if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".json") == 0)
    return kRoot + "/configs/" + arg;
  return kRoot + "/configs/" + arg + ".json";
}

bool video_ok(const std::string& path) {
  cv::VideoCapture probe(path);
  return probe.isOpened();
}

std::string resolve_video_path(const std::string& video, const std::string& cfg_path) {
  if (!video.empty() && video[0] == '/') return video;
  const size_t slash = cfg_path.find_last_of('/');
  const std::string cfg_dir = slash == std::string::npos ? "" : cfg_path.substr(0, slash + 1);
  const std::string candidates[] = {
      video,
      cfg_dir + video,
      kRoot + "/" + video,
      cfg_dir + "../" + video,
  };
  for (const auto& c : candidates) {
    if (video_ok(c)) return c;
  }
  return kRoot + "/" + video;
}

cv::Scalar to_scalar(const std::array<int, 3>& c) { return cv::Scalar(c[0], c[1], c[2]); }

std::vector<cv::Point> to_cv_poly(const std::vector<Point>& pts) {
  std::vector<cv::Point> out;
  for (const auto& p : pts) out.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
  return out;
}

// WebServer 只把 conf/iou 暴露成 HTTP 接口, 没有公开的 setter.
// 这里自己发一个本地请求把值写进去 —— 比改 web_server 的接口省事,
// 而且保证"网页滑块显示的值"和"实际用的值"永远一致.
void push_params(int port, float conf, float iou) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    char req[192];
    const int n = std::snprintf(req, sizeof(req),
                                "GET /api/param?conf=%.3f&iou=%.3f HTTP/1.0\r\n"
                                "Host: localhost\r\n\r\n",
                                conf, iou);
    if (n > 0 && ::send(fd, req, static_cast<size_t>(n), MSG_NOSIGNAL) > 0) {
      char junk[256];
      ::recv(fd, junk, sizeof(junk), 0);   // 不等完整响应, 读一下就关
    }
  }
  ::close(fd);
}

// ---------------------- 画图(按网页的图层开关) ----------------------

void draw_tracks(cv::Mat& frame, const std::vector<TrackView>& tracks, const Config& cfg,
                 const Overlay& ov) {
  if (!ov.box && !ov.id) return;
  for (const auto& tr : tracks) {
    const cv::Rect2f& b = tr.box;
    const cv::Point tl(cvRound(b.x), cvRound(b.y));
    const cv::Point br(cvRound(b.x + b.width), cvRound(b.y + b.height));
    if (ov.box) {
      cv::rectangle(frame, cv::Rect(tl, br), to_scalar(cfg.draw.box_color), cfg.draw.box_width);
    }
    if (ov.id) {
      char txt[32];
      std::snprintf(txt, sizeof(txt), "#%lld", tr.id);
      cv::putText(frame, txt, cv::Point(tl.x, std::max(12, tl.y - 6)),
                  cv::FONT_HERSHEY_SIMPLEX, cfg.draw.font_scale, to_scalar(cfg.draw.box_color),
                  cfg.draw.text_width);
    }
  }
}

void draw_counters(cv::Mat& frame, const Config& cfg, const Counter& counter) {
  std::vector<cv::Rect> placed;
  for (size_t i = 0; i < cfg.lines.size(); ++i) {
    const Line& L = cfg.lines[i];
    const cv::Point a(static_cast<int>(L.a.x), static_cast<int>(L.a.y));
    const cv::Point b(static_cast<int>(L.b.x), static_cast<int>(L.b.y));
    cv::line(frame, a, b, to_scalar(cfg.draw.line_color), cfg.draw.line_width);

    char txt[64];
    std::snprintf(txt, sizeof(txt), "%s: %d", L.name.c_str(), counter.count(i));
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, cfg.draw.counter_scale,
                                        cfg.draw.text_width, &baseline);
    int tx, ty;
    if (L.has_label_at) {
      tx = static_cast<int>(L.label_at.x);
      ty = static_cast<int>(L.label_at.y);
    } else {
      const cv::Point p = (a.y <= b.y) ? a : b;
      tx = p.x + 6;
      ty = (p.y - 10 > ts.height + 6) ? p.y - 10 : p.y + 26;
    }
    for (int k = 0; k < 6; ++k) {
      const cv::Rect box(tx, ty - ts.height, ts.width, ts.height + 4);
      bool clash = false;
      for (const auto& q : placed) {
        if ((box & q).area() > 0) {
          clash = true;
          break;
        }
      }
      if (!clash) break;
      ty += ts.height + 6;
    }
    placed.emplace_back(tx, ty - ts.height, ts.width, ts.height + 4);
    cv::putText(frame, txt, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, cfg.draw.counter_scale,
                to_scalar(cfg.draw.text_color), cfg.draw.text_width);
  }
}

// 抓一张以 (cx, cy) 为中心的小图(128×128, 出界就自动裁), 只补一个检测框.
// 过车记录里只要"是哪辆车", 计数线和跟踪编号一律不画, 画多了反而看不清.
// 返回文件名, 失败返回空
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
    if (d < best_d) {
      best_d = d;
      best = &tr;
    }
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

// ---------------------- 一次运行的全部状态 ----------------------

struct Session {
  std::string name;            // 配置名, "重新开始"时按它重建
  Config cfg;
  std::string video_path;
  cv::VideoCapture cap;
  std::unique_ptr<Detector> det;
  std::unique_ptr<Counter> counter;
  Tracker tracker;

  std::vector<cv::Point> poly;
  cv::Rect roi_rect;

  int frame_no = 0;
  size_t hits_seen = 0;        // 已经处理过的过线记录条数, 用来认出"新增的那几条"
  double fps = 0.0;
  double infer_ms = 0.0;
  float det_conf = 0.f;        // 建 detector 时用的参数, 跟网页滑块比较, 变了就重建
  float det_iou = 0.f;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

  explicit Session(const std::string& cfg_name) : name(cfg_name) {
    const std::string cfg_path = resolve_config_path(cfg_name);
    cfg = load_config(cfg_path);
    video_path = resolve_video_path(cfg.video, cfg_path);
    cap.open(video_path);
    if (!cap.isOpened()) throw std::runtime_error("打不开视频: " + video_path);
  }

  void build_detector(float conf, float iou) {
    det = std::make_unique<Detector>(kModel, cfg.detect.imgsz, conf, iou);
    det_conf = conf;
    det_iou = iou;
  }

  void prepare_geometry(int width, int height) {
    roi_rect = cv::Rect(0, 0, width, height);
    poly.clear();
    if (cfg.has_roi && cfg.roi.size() >= 3) {
      poly = to_cv_poly(cfg.roi);
      roi_rect = cv::boundingRect(poly) & cv::Rect(0, 0, width, height);
      if (roi_rect.width <= 0 || roi_rect.height <= 0) {
        roi_rect = cv::Rect(0, 0, width, height);
        poly.clear();
      }
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  int port = 8080;
  if (argc > 1) port = std::atoi(argv[1]);

  try {
    WebServer server(port, kHtml, kRoot + "/configs", kSnapDir);
    server.start();

    std::unique_ptr<Session> s;
    int last_frame = -1;
    auto last_tick = std::chrono::steady_clock::now();

    // 打开(或重开)一次会话: 读配置 -> 开视频 -> 建计数器/跟踪器/检测器.
    // "开始"和"重新开始"都走这里, 区别只是要不要先清空现场.
    auto open_session = [&](const std::string& cfg_name) {
      s.reset();
      server.clear_events();
      s = std::make_unique<Session>(cfg_name);
      const int w = static_cast<int>(s->cap.get(cv::CAP_PROP_FRAME_WIDTH));
      const int h = static_cast<int>(s->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      s->prepare_geometry(w, h);
      s->counter = std::make_unique<Counter>(s->cfg.lines);

      // 初始参数用配置文件里的值, 这样跟命令行版 traffic_count 跑出来的数字一致;
      // 同时同步给 WebServer, 网页滑块上显示的就是实际在用的值
      const float c0 = static_cast<float>(s->cfg.detect.conf);
      const float i0 = static_cast<float>(s->cfg.detect.iou);
      push_params(port, c0, i0);
      s->build_detector(c0, i0);

      server.set_state(1);
      last_tick = std::chrono::steady_clock::now();
      std::printf("开始: %s  %dx%d  conf=%.2f iou=%.2f  %s\n", s->name.c_str(), w, h,
                  static_cast<double>(c0), static_cast<double>(i0), s->video_path.c_str());
    };

    std::printf("服务已启动. 浏览器打开 http://localhost:%d\n", port);
    std::printf("等待网页下命令(Ctrl+C 退出)\n");

    while (true) {
      // ---- 1. 处理网页命令 ----
      switch (server.take_command()) {
        case kCmdStart: {
          const std::string name = server.start_config();
          try {
            open_session(name.empty() ? std::string("test1") : name);
          } catch (const std::exception& e) {
            std::fprintf(stderr, "启动失败: %s\n", e.what());
            s.reset();
            server.set_state(0);
          }
          break;
        }
        case kCmdStop:
          s.reset();
          server.set_state(0);
          std::printf("已停止\n");
          break;
        case kCmdRestart:
          if (s) {
            const std::string name = s->name;
            try {
              open_session(name);   // 整个会话重建, 计数/事件/跟踪全部归零
              std::printf("重新开始: %s\n", name.c_str());
            } catch (const std::exception& e) {
              std::fprintf(stderr, "重新开始失败: %s\n", e.what());
              s.reset();
              server.set_state(0);
            }
          }
          break;
        default:
          break;
      }

      const int st = server.state();
      if (st == 0 || st == 2 || !s) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        continue;
      }

      // ---- 2. 网页拖了滑块就重建检测器(一两百毫秒生效) ----
      if (std::fabs(server.conf() - s->det_conf) > 0.001f ||
          std::fabs(server.iou() - s->det_iou) > 0.001f) {
        s->build_detector(server.conf(), server.iou());
      }

      // ---- 3. 读一帧 ----
      cv::Mat frame;
      if (!s->cap.read(frame)) {
        // 视频放完了: 停在最后一帧, 转成暂停状态, 等用户点"重新开始"
        if (server.state() != 2) {
          server.set_state(2);
          std::printf("视频播放完毕, 共 %d 帧\n", s->frame_no);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        continue;
      }
      ++s->frame_no;

      // ---- 4. 推理 + 跟踪(跟 main.cpp 完全一致) ----
      const auto t0 = std::chrono::steady_clock::now();
      cv::Mat crop = frame(s->roi_rect);
      std::vector<Detection> dets = s->det->detect(crop);
      for (auto& d : dets) {
        d.x1 += s->roi_rect.x;
        d.x2 += s->roi_rect.x;
        d.y1 += s->roi_rect.y;
        d.y2 += s->roi_rect.y;
      }
      if (!s->poly.empty()) {
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                                  [&](const Detection& d) {
                                    const cv::Point2f c(static_cast<float>((d.x1 + d.x2) * 0.5),
                                                        static_cast<float>((d.y1 + d.y2) * 0.5));
                                    return cv::pointPolygonTest(s->poly, c, false) < 0;
                                  }),
                   dets.end());
      }

      const auto tracks = s->tracker.update(dets);
      std::vector<TrackView> live;
      live.reserve(tracks.size());
      for (const auto& tr : tracks) {
        if (tr.moving) live.push_back(tr);
      }
      for (const auto& tr : live) {
        const cv::Point2f c = center_of(tr.box);
        s->counter->update(tr.id, Point{c.x, c.y}, "vehicle", s->frame_no);
      }
      s->infer_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      // ---- 5. 有新过线记录就抓拍 + 记事件 ----
      // 抓拍必须放在画线之前: 记录里只要车本身, 不要计数线和编号.
      const auto& hits = s->counter->hits();
      if (hits.size() > s->hits_seen) {
        const double ts =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - s->t0).count();
        for (size_t i = s->hits_seen; i < hits.size(); ++i) {
          const Hit& h = hits[i];
          char tag[96];
          std::snprintf(tag, sizeof(tag), "%06d_%s_%zu", h.frame, h.line.c_str(), i);
          Event e;
          e.frame = h.frame;
          e.line = h.line;
          e.cls = h.cls;
          e.x = h.x;
          e.y = h.y;
          e.snap = save_snap(frame, h.x, h.y, tag, live, to_scalar(s->cfg.draw.box_color));
          e.ts = ts;
          server.add_event(e);
        }
        s->hits_seen = hits.size();
      }

      // ---- 6. 画 ----
      const Overlay ov = server.overlay();
      if (!s->poly.empty() && ov.roi) {
        cv::polylines(frame, s->poly, true, to_scalar(s->cfg.draw.roi_color), 2);
      }
      draw_tracks(frame, live, s->cfg, ov);
      if (ov.line) draw_counters(frame, s->cfg, *s->counter);

      // ---- 7. 推帧 + 更新数字 ----
      server.push_frame(frame);

      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last_tick).count();
      if (dt > 0) {
        const double inst = 1.0 / dt;
        s->fps = (s->fps <= 0) ? inst : s->fps * 0.9 + inst * 0.1;
      }
      last_tick = now;

      if (s->frame_no != last_frame) {
        last_frame = s->frame_no;
        std::vector<LineStat> stats;
        stats.reserve(s->cfg.lines.size());
        for (size_t i = 0; i < s->cfg.lines.size(); ++i) {
          stats.push_back(LineStat{s->cfg.lines[i].name, s->counter->count(i)});
        }
        server.update_stats(stats, static_cast<int>(live.size()), s->fps, s->infer_ms,
                            s->frame_no);
      }

      if (s->cfg.max_frames > 0 && s->frame_no >= s->cfg.max_frames) {
        server.set_state(2);
        std::printf("到达配置的 max_frames=%d, 暂停\n", s->cfg.max_frames);
      }
    }

    server.stop();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "出错: %s\n", e.what());
    return 1;
  }
  return 0;
}
