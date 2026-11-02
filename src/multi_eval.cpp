// 多路并行评测：N 个线程同时跑不同视频，每个线程绑定到指定的 NPU 核。
//
// 用途：批量评测模型（对比 v7 与 head4 的漏计差异），同时跑多路把 NPU 吃满。
//
// 为什么另写一个而不复用 traffic_count：
//   traffic_count 是业务程序，要画框、存视频、给人看。
//   评测只要计数的数字，所以循环可以极简，不画图不编码，速度更快。
//
// 处理流程与 main.cpp 保持一致（只去掉画图和存视频）：
//   读帧 -> 裁 ROI -> 推理 -> 外接矩形外框丢中心点落在多边形外的
//   -> 跟踪发编号 -> 只把 moving 的送去计数 -> retain 清理死 id
//
// NPU 核绑定：每个线程创建自己的 RknnDetector（各自一份 context），
//   按 i % 3 绑到 0/1/2 号核。4 路时为 0,1,2,0。
//
// 用法：
//   ./multi_eval <配置1> [配置2] ... [--model path] [--frames N] [--out 前缀]

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "counter.hpp"
#include "rknn_detector.hpp"
#include "tracker.hpp"

namespace {

struct RouteResult {
  std::string name;
  int frames = 0;
  long long total_ids = 0;         // 累计分配过多少编号（断号指标）
  size_t live_at_end = 0;          // 结束时还活着的轨迹数
  int total_boxes = 0;             // 累计送进跟踪的框数
  double seconds = 0.0;
  std::vector<std::pair<std::string, int>> counts;   // 每条线的计数
};

std::mutex g_print_mutex;

std::string stem_of(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

// 取框中心
cv::Point2f center_of(const cv::Rect2f& b) {
  return cv::Point2f((b.x + b.x + b.width) * 0.5f, (b.y + b.y + b.height) * 0.5f);
}

// 配置里的 video 是相对项目根的路径，拼成绝对路径
std::string resolve_video(const Config& cfg) {
  namespace fs = std::filesystem;
  fs::path p(cfg.video);
  if (p.is_absolute() && fs::exists(p)) return p.string();
  const fs::path full = fs::path(TRAFFIC_PROJECT_ROOT) / p;
  if (fs::exists(full)) return full.string();
  return cfg.video;
}

RouteResult run_route(const std::string& cfg_path, const std::string& model_path,
                      int core_id, int max_frames) {
  RouteResult r;
  r.name = stem_of(cfg_path);

  Config cfg = load_config(cfg_path);
  const std::string video = resolve_video(cfg);

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    std::lock_guard<std::mutex> lk(g_print_mutex);
    std::printf("[%s] 打不开视频: %s\n", r.name.c_str(), video.c_str());
    return r;
  }

  // 每路独立一份检测器和 context，绑到指定核
  RknnDetector detector(model_path, cfg.detect.imgsz, cfg.detect.conf, cfg.detect.iou);
  detector.set_core(core_id);

  Tracker tracker;
  Counter counter(cfg.lines);

  // ROI：外接矩形用于裁图，多边形用于过滤中心点
  const int fw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int fh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::Rect roi_rect(0, 0, fw, fh);
  std::vector<cv::Point> poly;
  if (cfg.has_roi && cfg.roi.size() >= 3) {
    poly.reserve(cfg.roi.size());
    for (const auto& p : cfg.roi) {
      poly.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    roi_rect = cv::boundingRect(poly) & cv::Rect(0, 0, fw, fh);
    if (roi_rect.width <= 0 || roi_rect.height <= 0) roi_rect = cv::Rect(0, 0, fw, fh);
  }

  const auto t0 = std::chrono::steady_clock::now();
  cv::Mat frame;
  int frame_no = 0;
  while (cap.read(frame) && !frame.empty()) {
    if (max_frames > 0 && frame_no >= max_frames) break;
    ++frame_no;
    ++r.frames;

    std::vector<Detection> dets = detector.detect(frame(roi_rect));
    for (auto& d : dets) {
      d.x1 += roi_rect.x; d.x2 += roi_rect.x;
      d.y1 += roi_rect.y; d.y2 += roi_rect.y;
    }

    // 中心点落在多边形外的框丢掉（与 main.cpp 7.3 一致）
    if (!poly.empty()) {
      dets.erase(std::remove_if(dets.begin(), dets.end(),
                                [&](const Detection& d) {
                                  const cv::Point2f c(static_cast<float>((d.x1 + d.x2) * 0.5),
                                                      static_cast<float>((d.y1 + d.y2) * 0.5));
                                  return cv::pointPolygonTest(poly, c, false) < 0;
                                }),
                 dets.end());
    }
    r.total_boxes += static_cast<int>(dets.size());

    const auto tracks = tracker.update(dets);
    for (const auto& tr : tracks) {
      if (!tr.moving) continue;      // 静止目标不计数（与 main.cpp 7.4 一致）
      const cv::Point2f c = center_of(tr.box);
      counter.update(tr.id, Point{c.x, c.y}, "vehicle", frame_no);
    }
    counter.retain(tracker.live_ids());
  }
  cap.release();

  const auto t1 = std::chrono::steady_clock::now();
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  r.total_ids = tracker.total_ids();
  r.live_at_end = tracker.size();
  for (size_t i = 0; i < cfg.lines.size(); ++i) {
    r.counts.emplace_back(cfg.lines[i].name, counter.count(i));
  }

  {
    std::lock_guard<std::mutex> lk(g_print_mutex);
    std::printf("[%-8s] %5d 帧  %7.1f 秒  %6.1f fps  编号 %lld\n", r.name.c_str(),
                r.frames, r.seconds, r.frames / (r.seconds > 0 ? r.seconds : 1), r.total_ids);
  }
  return r;
}

void usage() {
  std::printf(
      "用法: multi_eval <配置1> [配置2] ... [选项]\n"
      "\n"
      "选项:\n"
      "  --model <path>   使用哪个 .rknn 模型（默认 models/best_detrac.rknn）\n"
      "  --frames <N>     每路最多跑多少帧（默认 0 = 全部）\n"
      "  --out <prefix>   结果写到 <prefix>_summary.csv\n"
      "                   （默认 output/multieval）\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> cfgs;
  std::string model = std::string(TRAFFIC_PROJECT_ROOT) + "/models/best_detrac.rknn";
  std::string out_prefix = std::string(TRAFFIC_PROJECT_ROOT) + "/output/multieval";
  int max_frames = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if (a == "--frames" && i + 1 < argc) {
      max_frames = std::atoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (!a.empty() && a[0] == '-') {
      std::printf("未知选项: %s\n", a.c_str());
      usage();
      return 1;
    } else {
      cfgs.push_back(a);
    }
  }
  if (cfgs.empty()) {
    usage();
    return 1;
  }

  std::printf("模型: %s\n", model.c_str());
  std::printf("路线: %zu 路，每路最多 %d 帧\n", cfgs.size(), max_frames);
  std::printf("核分配: ");
  for (size_t i = 0; i < cfgs.size(); ++i) {
    std::printf("%s->核%d ", stem_of(cfgs[i]).c_str(), static_cast<int>(i % 3));
  }
  std::printf("\n\n");

  const auto t_start = std::chrono::steady_clock::now();

  std::vector<RouteResult> results(cfgs.size());
  std::vector<std::thread> threads;
  threads.reserve(cfgs.size());
  for (size_t i = 0; i < cfgs.size(); ++i) {
    const int core = static_cast<int>(i % 3);
    threads.emplace_back([&, i, core]() {
      try {
        results[i] = run_route(cfgs[i], model, core, max_frames);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        std::printf("[%s] 出错: %s\n", stem_of(cfgs[i]).c_str(), e.what());
        results[i].name = stem_of(cfgs[i]);
      }
    });
  }
  for (auto& t : threads) t.join();

  const auto t_end = std::chrono::steady_clock::now();
  const double wall = std::chrono::duration<double>(t_end - t_start).count();

  int total_frames = 0;
  long long total_ids = 0;
  std::printf("\n===== 汇总 =====\n");
  std::printf("%-10s %8s %10s %10s %10s\n", "路线", "帧数", "编号数", "耗时(s)", "fps");
  for (const auto& r : results) {
    total_frames += r.frames;
    total_ids += r.total_ids;
    std::printf("%-10s %8d %10lld %10.1f %10.1f\n", r.name.c_str(), r.frames,
                r.total_ids, r.seconds, r.frames / (r.seconds > 0 ? r.seconds : 1));
  }
  std::printf("%-10s %8d %10lld %10.1f %10.1f  <- 合计 / 总墙钟\n", "合计",
              total_frames, total_ids, wall, total_frames / (wall > 0 ? wall : 1));

  std::printf("\n===== 各线计数 =====\n");
  for (const auto& r : results) {
    std::printf("%-10s", r.name.c_str());
    for (const auto& c : r.counts) {
      std::printf("  %s=%d", c.first.c_str(), c.second);
    }
    std::printf("\n");
  }

  {
    std::ofstream f(out_prefix + "_summary.csv");
    if (f) {
      f << "route,frames,total_ids,seconds,fps,total_boxes";
      for (const auto& r : results) {
        for (const auto& c : r.counts) f << "," << c.first;
        if (!r.counts.empty()) break;
      }
      f << "\n";
      for (const auto& r : results) {
        f << r.name << "," << r.frames << "," << r.total_ids << "," << r.seconds << ","
          << (r.frames / (r.seconds > 0 ? r.seconds : 1)) << "," << r.total_boxes;
        for (const auto& c : r.counts) f << "," << c.second;
        f << "\n";
      }
      std::printf("\n结果已写: %s\n", (out_prefix + "_summary.csv").c_str());
    }
  }
  return 0;
}
