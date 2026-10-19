// 多路口车流量统计 —— C++ 版入口 (板端用).
//
// 这一版: 读配置 -> 裁 ROI -> ONNX 推理 -> IoU 跟踪发 ID -> 静止目标过滤
//         -> 穿线计数 -> 画框/线/数字 -> 播放或存视频.
//
// 跟踪是自己写的极简版: 相邻两帧按 IoU 贪心匹配, 不是 ByteTrack.
// 没有卡尔曼预测, 遮挡或漏检时会断 ID (和 Python 版的差距就在这).
// 车牌号会跳, 但计数只看"同一 ID 有没有从线的一侧走到另一侧", 断了会多数.
//
// 用法:
//   ./traffic_count test4
// 不带参数就列出现有配置. 退出: 窗口里按 Q 或 Esc.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "counter.hpp"
//#include "detector.hpp"
#ifdef USE_RKNN
#include "rknn_detector.hpp"
using Detector = RknnDetector;
#else
#include "detector.hpp"
#endif

#include "recorder.hpp"


#ifndef TRAFFIC_PROJECT_ROOT
#define TRAFFIC_PROJECT_ROOT "."
#endif
#ifndef TRAFFIC_MODEL
#define TRAFFIC_MODEL TRAFFIC_PROJECT_ROOT "/models/best_detrac.onnx"
#endif

namespace fs = std::filesystem;

namespace {

const std::string kRoot = TRAFFIC_PROJECT_ROOT;
const std::string kModel = TRAFFIC_MODEL;

std::string resolve_config_path(const std::string& arg) {
  if (arg.find('/') != std::string::npos) return arg;
  if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".json") == 0)
    return kRoot + "/configs/" + arg;
  return kRoot + "/configs/" + arg + ".json";
}

std::vector<std::string> list_configs() {
  std::vector<std::string> names;
  const std::string dir = kRoot + "/configs";
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return names;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file()) continue;
    const std::string fn = entry.path().filename().string();
    if (fn.size() > 5 && fn.compare(fn.size() - 5, 5, ".json") == 0)
      names.push_back(fn.substr(0, fn.size() - 5));
  }
  std::sort(names.begin(), names.end());
  return names;
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
      video, cfg_dir + video, kRoot + "/" + video, cfg_dir + "../" + video,
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

// ---------------- 极简跟踪器 ----------------
// 只做一件事: 给每一帧的检测框发一个稳定的 ID.
// 匹配规则: 上一帧的轨迹和这一帧的框算 IoU, 大于阈值就认成同一辆, 大的配对优先.
// 匹配不上的轨迹先留着(max_lost 帧), 等它回来; 超时才真正删掉.

struct Track {
  long long id = 0;
  cv::Rect2f box;
  cv::Point2f origin;  // 第一次出现时的中心, 用来判"它到底动没动"
  int lost = 0;        // 连续几帧没匹配上
};

// 这一帧要拿去做计数的东西
struct TrackView {
  long long id = 0;
  cv::Rect2f box;
  bool moving = false;  // 离出现位置超过 kMinTravel 像素才算真的在开
};

// 少于这个位移就当成静止目标: 红绿灯、路牌、被风吹的广告牌.
// 阈值不能太大, 否则刚进画面的车要等很久才开始计数.
const double kMinTravel = 12.0;

cv::Point2f center_of(const cv::Rect2f& b) {
  return cv::Point2f(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
}

cv::Rect2f to_rect(const Detection& d) {
  return cv::Rect2f(static_cast<float>(d.x1), static_cast<float>(d.y1),
                    static_cast<float>(d.x2 - d.x1), static_cast<float>(d.y2 - d.y1));
}

double iou_rect(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float iw = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
  const float ih = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
  if (iw <= 0 || ih <= 0) return 0.0;
  const float inter = iw * ih;
  return inter / (a.width * a.height + b.width * b.height - inter);
}

class Tracker {
 public:
  explicit Tracker(int max_lost = 15, double iou_thr = 0.3)
      : max_lost_(max_lost), iou_thr_(iou_thr) {}

  // 返回这一帧活着的轨迹: (ID, 框, 动没动). 坐标跟 dets 一样是整图坐标
  std::vector<TrackView> update(const std::vector<Detection>& dets) {
    std::vector<bool> used(dets.size(), false);
    std::vector<TrackView> out;

    // 第一步: 老轨迹找框
    for (auto& t : tracks_) {
      int best = -1;
      double best_iou = iou_thr_;
      for (size_t i = 0; i < dets.size(); ++i) {
        if (used[i]) continue;
        const double v = iou_rect(t.box, to_rect(dets[i]));
        if (v > best_iou) {
          best_iou = v;
          best = static_cast<int>(i);
        }
      }
      if (best >= 0) {
        used[best] = true;
        t.box = to_rect(dets[best]);
        t.lost = 0;
        const cv::Point2f d = center_of(t.box) - t.origin;
        TrackView v;
        v.id = t.id;
        v.box = t.box;
        v.moving = cv::norm(d) >= kMinTravel;
        out.push_back(v);
      } else {
        ++t.lost;
      }
    }

    // 第二步: 剩下的框当成新车, 发新 ID
    for (size_t i = 0; i < dets.size(); ++i) {
      if (used[i]) continue;
      Track t;
      t.id = next_id_++;
      t.box = to_rect(dets[i]);
      t.origin = center_of(t.box);
      tracks_.push_back(t);
      TrackView v;
      v.id = t.id;
      v.box = t.box;
      v.moving = false;  // 刚出现, 还没走动
      out.push_back(v);
    }

    // 第三步: 太久没出现的轨迹删掉
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& t) { return t.lost > max_lost_; }),
                  tracks_.end());
    return out;
  }

  size_t size() const { return tracks_.size(); }

 private:
  std::vector<Track> tracks_;
  long long next_id_ = 1;
  int max_lost_ = 15;
  double iou_thr_ = 0.3;
};

// ---------------- 画面 ----------------

void draw_tracks(cv::Mat& frame, const std::vector<TrackView>& tracks, const Config& cfg) {
  for (const auto& tr : tracks) {
    const cv::Rect2f& b = tr.box;
    const cv::Point tl(cvRound(b.x), cvRound(b.y));
    const cv::Point br(cvRound(b.x + b.width), cvRound(b.y + b.height));

    if (cfg.draw.box) {
      cv::rectangle(frame, cv::Rect(tl, br), to_scalar(cfg.draw.box_color), cfg.draw.box_width);
    }
    if (cfg.draw.id) {
      char txt[32];
      std::snprintf(txt, sizeof(txt), "#%lld", tr.id);
      cv::putText(frame, txt, cv::Point(tl.x, std::max(12, tl.y - 6)),
                  cv::FONT_HERSHEY_SIMPLEX, cfg.draw.font_scale,
                  to_scalar(cfg.draw.box_color), cfg.draw.text_width);
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
    const cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX,
                                        cfg.draw.counter_scale, cfg.draw.text_width,
                                        &baseline);
    int tx, ty;
    if (L.has_label_at) {
      tx = static_cast<int>(L.label_at.x);
      ty = static_cast<int>(L.label_at.y);
    } else {
      const cv::Point p = (a.y <= b.y) ? a : b;
      tx = p.x + 6;
      ty = (p.y - 10 > ts.height + 6) ? p.y - 10 : p.y + 26;
    }
    // 几条线的数字压在一起时往下挪
    for (int k = 0; k < 6; ++k) {
      cv::Rect box(tx, ty - ts.height, ts.width, ts.height + 4);
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
    cv::putText(frame, txt, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX,
                cfg.draw.counter_scale, to_scalar(cfg.draw.text_color), cfg.draw.text_width);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("用法: %s <配置名>\n", argv[0]);
    std::printf("现有配置 (%s/configs):\n", kRoot.c_str());
    for (const auto& n : list_configs()) std::printf("  %s\n", n.c_str());
    return 1;
  }
  if (argc > 2) {
    std::fprintf(stderr, "用法: %s <配置名>\n", argv[0]);
    return 1;
  }

  try {
    const std::string cfg_path = resolve_config_path(argv[1]);
    const Config cfg = load_config(cfg_path);
    const std::string video_path = resolve_video_path(cfg.video, cfg_path);

    std::printf("配置: %s\n视频: %s\n模型: %s\n", cfg_path.c_str(), video_path.c_str(),
                kModel.c_str());

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::fprintf(stderr, "打不开视频: %s\n", video_path.c_str());
      return 1;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    std::printf("画面: %dx%d  %.1f fps\n", width, height, fps);
    std::printf("线: %zu 条  ROI: %s\n", cfg.lines.size(), cfg.has_roi ? "有" : "无(整帧)");

    // ROI: 外接矩形用来裁图, 多边形用来判"框中心在不在区域内"(跟 Python 版同样两步)
    std::vector<cv::Point> poly;
    cv::Rect roi_rect(0, 0, width, height);
    if (cfg.has_roi && cfg.roi.size() >= 3) {
      poly = to_cv_poly(cfg.roi);
      roi_rect = cv::boundingRect(poly) & cv::Rect(0, 0, width, height);
      if (roi_rect.width <= 0 || roi_rect.height <= 0) {  // 配置写歪了, 退回整帧
        roi_rect = cv::Rect(0, 0, width, height);
        poly.clear();
      }
    }
    std::printf("推理区域: x%d~%d y%d~%d (%dx%d)  ROI过滤: %s\n", roi_rect.x,
                roi_rect.x + roi_rect.width, roi_rect.y, roi_rect.y + roi_rect.height,
                roi_rect.width, roi_rect.height, poly.empty() ? "关" : "开");

    Detector detector(kModel, cfg.detect.imgsz, cfg.detect.conf, cfg.detect.iou);

    cv::VideoWriter writer;
    const bool to_file = (cfg.output != "show");
    if (to_file) {
      const std::string out = "output/" + cfg.name + "_lanes.mp4";
      writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                  cv::Size(width, height));
      std::printf("输出视频: %s\n", out.c_str());
    }

    Counter counter(cfg.lines);
    Tracker tracker;
    cv::Mat frame;
    int frame_no = 0;
    int total_boxes = 0;
    int total_tracks = 0;
    const auto t0 = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
      ++frame_no;

      // 1. 只把 ROI 外接矩形送进模型
      cv::Mat crop = frame(roi_rect);
      std::vector<Detection> dets = detector.detect(crop);

      // 2. 加回偏移, 变整图坐标
      for (auto& d : dets) {
        d.x1 += roi_rect.x;
        d.x2 += roi_rect.x;
        d.y1 += roi_rect.y;
        d.y2 += roi_rect.y;
      }

      // 3. 外接矩形比多边形大, 中心点落在多边形外的框丢掉
      if (!poly.empty()) {
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                                  [&](const Detection& d) {
                                    const cv::Point2f c(static_cast<float>((d.x1 + d.x2) * 0.5),
                                                        static_cast<float>((d.y1 + d.y2) * 0.5));
                                    return cv::pointPolygonTest(poly, c, false) < 0;
                                  }),
                   dets.end());
      }
      total_boxes += static_cast<int>(dets.size());

      // 4. 跟踪发 ID. 红绿灯/路牌这种一直不动的, 既不算也不画:
      //    它们的框在线上抖来抖去会被反复算成过车, 留着只会干扰看图.
      const auto tracks = tracker.update(dets);
      std::vector<TrackView> live;
      live.reserve(tracks.size());
      for (const auto& tr : tracks) {
        if (tr.moving) live.push_back(tr);
      }
      for (const auto& tr : live) {
        const cv::Point2f c = center_of(tr.box);
        counter.update(tr.id, Point{c.x, c.y}, "vehicle", frame_no);
      }
      total_tracks = static_cast<int>(tracker.size());

      // 5. 画
      if (!poly.empty() && cfg.draw.roi) {
        cv::polylines(frame, poly, true, to_scalar(cfg.draw.roi_color), 2);
      }
      draw_tracks(frame, live, cfg);
      draw_counters(frame, cfg, counter);

      if (to_file) {
        writer.write(frame);
      } else {
        cv::imshow(cfg.name, frame);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q') break;
      }

      if (cfg.max_frames > 0 && frame_no >= cfg.max_frames) break;
    }

    if (writer.isOpened()) writer.release();
    write_records(std::string("output/records_") + cfg.name + ".csv", counter);

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("共 %d 帧, %.2f 秒, %.1f 帧/秒, 平均每帧 %.1f 个框, 同时在跟踪 %d 条\n",
                frame_no, sec, sec > 0 ? frame_no / sec : 0.0,
                frame_no > 0 ? static_cast<double>(total_boxes) / frame_no : 0.0,
                total_tracks);
    for (size_t i = 0; i < cfg.lines.size(); ++i) {
      std::printf("  %s: %d\n", cfg.lines[i].name.c_str(), counter.count(i));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "出错: %s\n", e.what());
    return 1;
  }
  return 0;
}
