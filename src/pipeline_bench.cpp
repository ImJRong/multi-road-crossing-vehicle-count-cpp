// 两级流水线性能测试：读帧 -> 并行推理 -> 串行跟踪，测吞吐上限。
// 每个推理线程独立 RknnDetector 并绑核，队列有界背压，结果按帧号排序再送跟踪。
// 用法：./pipeline_bench <配置> [--workers N] [--frames N] [--model path]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "counter.hpp"
#include "rknn_detector.hpp"
#include "tracker.hpp"

namespace {

// 有界队列。满了让生产者等，空了让消费者等。
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t cap) : cap_(cap) {}

  // 返回 false 表示已关闭
  bool push(T item) {
    std::unique_lock<std::mutex> lk(mu_);
    not_full_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
    if (closed_) return false;
    q_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  // 返回 false 表示已关闭且取空
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(mu_);
    not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;               // closed_ 且空了
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable not_empty_, not_full_;
  std::deque<T> q_;
  size_t cap_;
  bool closed_ = false;
};

struct FrameJob {
  int frame_no = 0;
  cv::Mat image;          // ROI 裁剪后的图
  std::chrono::steady_clock::time_point t_enqueue;   // 入队时刻，用于测延迟
};

struct DetResult {
  int frame_no = 0;
  std::vector<Detection> dets;   // 已换算回整图坐标
  std::chrono::steady_clock::time_point t_enqueue;   // 进入待推理队列
  std::chrono::steady_clock::time_point t_infer_begin; // 推理线程取到它
  std::chrono::steady_clock::time_point t_infer_end;   // 推理完成
};

std::string stem_of(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

std::string resolve_video(const Config& cfg) {
  namespace fs = std::filesystem;
  fs::path p(cfg.video);
  if (p.is_absolute() && fs::exists(p)) return p.string();
  const fs::path full = fs::path(TRAFFIC_PROJECT_ROOT) / p;
  if (fs::exists(full)) return full.string();
  return cfg.video;
}

void usage() {
  std::printf(
      "用法: pipeline_bench <配置> [选项]\n"
      "\n"
      "选项:\n"
      "  --workers N    推理线程数（默认 3，对应 3 个 NPU 核）\n"
      "  --frames N     最多跑多少帧（默认 0 = 全部）\n"
      "  --model path   使用哪个 .rknn 模型\n"
      "  --buf N        读帧队列上限（默认 32）。调小可降低排队延迟\n"
      "  --realtime N   按 N fps 的节奏送帧，模拟实时摄像头（默认 0 = 尽快读）\n"
      "  --no-track     只统计检测，跳过跟踪与计数（用于测跟踪开销）\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path;
  std::string model = std::string(TRAFFIC_PROJECT_ROOT) + "/models/best_detrac.rknn";
  int workers = 3;
  int max_frames = 0;
  bool no_track = false;
  int buf = 32;
  double realtime_fps = 0.0;   // >0 表示按摄像头节奏送帧

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--workers" && i + 1 < argc) {
      workers = std::atoi(argv[++i]);
    } else if (a == "--frames" && i + 1 < argc) {
      max_frames = std::atoi(argv[++i]);
    } else if (a == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if (a == "--buf" && i + 1 < argc) {
      buf = std::atoi(argv[++i]);
    } else if (a == "--realtime" && i + 1 < argc) {
      realtime_fps = std::atof(argv[++i]);
    } else if (a == "--no-track") {
      no_track = true;
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (!a.empty() && a[0] == '-') {
      std::printf("未知选项: %s\n", a.c_str());
      usage();
      return 1;
    } else {
      cfg_path = a;
    }
  }
  if (cfg_path.empty()) {
    usage();
    return 1;
  }

  Config cfg = load_config(cfg_path);
  const std::string video = resolve_video(cfg);

  // 先打开视频读一遍元信息（后面读帧线程会重新打开）
  int fw = 0, fh = 0;
  {
    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
      std::printf("打不开视频: %s\n", video.c_str());
      return 1;
    }
    fw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    fh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  }

  cv::Rect roi_rect(0, 0, fw, fh);
  std::vector<cv::Point> poly;
  if (cfg.has_roi && cfg.roi.size() >= 3) {
    poly.reserve(cfg.roi.size());
    for (const auto& p : cfg.roi) poly.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    roi_rect = cv::boundingRect(poly) & cv::Rect(0, 0, fw, fh);
    if (roi_rect.width <= 0 || roi_rect.height <= 0) roi_rect = cv::Rect(0, 0, fw, fh);
  }

  std::printf("配置: %s\n模型: %s\n", cfg_path.c_str(), model.c_str());
  std::printf("视频: %s (%dx%d), ROI %dx%d\n", video.c_str(), fw, fh,
              roi_rect.width, roi_rect.height);
  std::printf("推理线程: %d  最多帧数: %d  跟踪: %s\n\n", workers, max_frames,
              no_track ? "关闭" : "开启");

  BoundedQueue<FrameJob> q_frame(static_cast<size_t>(buf > 0 ? buf : 1));     // 读帧 -> 推理
  BoundedQueue<DetResult> q_det(static_cast<size_t>(buf > 0 ? buf : 1) * 2);      // 推理 -> 跟踪

  std::atomic<int> read_count{0};
  std::atomic<int> infer_count{0};
  std::atomic<bool> read_done{false};

  // ---- 读帧线程 ----
  std::thread reader([&] {
    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
      q_frame.close();
      return;
    }
    // 模拟实时流：按 realtime_fps 的节奏送帧，等价于摄像头推流。
    // 0 表示不限制，按机器能力尽快读（离线批处理模式）。
    const double interval_ms = realtime_fps > 0 ? 1000.0 / realtime_fps : 0.0;
    const auto t_stream_start = std::chrono::steady_clock::now();

    cv::Mat frame;
    int n = 0;
    while (cap.read(frame) && !frame.empty()) {
      if (max_frames > 0 && n >= max_frames) break;

      // 按节奏对齐：第 n 帧应在 t_stream_start + n * interval 时刻送达
      if (interval_ms > 0.0) {
        const auto target = t_stream_start +
                            std::chrono::microseconds(static_cast<long long>(n * interval_ms * 1000));
        std::this_thread::sleep_until(target);
      }

      FrameJob job;
      job.frame_no = n;
      job.image = frame(roi_rect).clone();
      job.t_enqueue = std::chrono::steady_clock::now();
      ++n;
      read_count.store(n);
      if (!q_frame.push(std::move(job))) return;
    }
    cap.release();
    read_done.store(true);
    q_frame.close();
  });

  // ---- 推理线程（各绑一个核）----
  std::vector<std::thread> infer_threads;
  infer_threads.reserve(workers);
  for (int w = 0; w < workers; ++w) {
    infer_threads.emplace_back([&, w] {
      RknnDetector det(model, cfg.detect.imgsz, cfg.detect.conf, cfg.detect.iou);
      det.set_core(w % 3);

      FrameJob job;
      while (q_frame.pop(job)) {
        const auto t_infer_begin = std::chrono::steady_clock::now();
        std::vector<Detection> dets = det.detect(job.image);
        const auto t_infer_end = std::chrono::steady_clock::now();
        // 换算回整图坐标
        for (auto& d : dets) {
          d.x1 += roi_rect.x; d.x2 += roi_rect.x;
          d.y1 += roi_rect.y; d.y2 += roi_rect.y;
        }
        DetResult r;
        r.frame_no = job.frame_no;
        r.dets = std::move(dets);
        r.t_enqueue = job.t_enqueue;
        r.t_infer_begin = t_infer_begin;
        r.t_infer_end = t_infer_end;
        if (!q_det.push(std::move(r))) return;
        infer_count.fetch_add(1);
      }
    });
  }

  // ---- 跟踪线程（单线程串行消费）----
  // 用一个小缓冲按帧号排序：允许一定乱序，超过就强制放行最老的，
  // 避免个别帧卡住导致整个流水线停摆。
  Tracker tracker(cfg.track.buffer, cfg.track.high, cfg.track.low,
                  cfg.track.new_thresh, cfg.track.match);
  Counter counter(cfg.lines);
  long long tracked_frames = 0;
  std::map<int, std::vector<Detection>> pending;

  // 队列深度监控：判断流水线跟不跟得上送帧速度
  size_t qmax_seen = 0;          // 待推理队列出现过的最大深度
  size_t qlast = 0;              // 收尾时的深度

  // 延迟统计：把一段完整耗时拆成三段，区分「排队」和「真处理」
  std::vector<double> lat_infer;    // 纯推理耗时
  std::vector<double> lat_queue;    // 入队到开始推理（排队等待）
  std::vector<double> lat_total;    // 入队到跟踪完成（端到端）
  lat_infer.reserve(4096);
  lat_queue.reserve(4096);
  lat_total.reserve(4096);

  const auto t0 = std::chrono::steady_clock::now();

  std::thread consumer([&] {
    DetResult r;
    int next_expected = 0;
    while (q_det.pop(r)) {
      pending.emplace(r.frame_no, std::move(r.dets));

      // 攒够 8 帧或已有期望帧，就按序放行
      while (!pending.empty()) {
        auto it = pending.begin();
        const int min_no = it->first;

        // 距期望帧还远且缓冲不大时，等一等别的线程补上
        if (min_no > next_expected && pending.size() < 8) break;

        std::vector<Detection> dets = std::move(it->second);
        pending.erase(it);

        // ROI 多边形过滤（与 main.cpp 一致）
        if (!poly.empty()) {
          dets.erase(std::remove_if(dets.begin(), dets.end(),
                                    [&](const Detection& d) {
                                      const cv::Point2f c(
                                          static_cast<float>((d.x1 + d.x2) * 0.5),
                                          static_cast<float>((d.y1 + d.y2) * 0.5));
                                      return cv::pointPolygonTest(poly, c, false) < 0;
                                    }),
                     dets.end());
        }

        if (no_track) { ++tracked_frames; next_expected = min_no + 1; continue; }
        const auto tracks = tracker.update(dets);
        const int frame_no = min_no + 1;
        for (const auto& tr : tracks) {
          if (!tr.moving) continue;
          const cv::Point2f c((tr.box.x + tr.box.x + tr.box.width) * 0.5f,
                              (tr.box.y + tr.box.y + tr.box.height) * 0.5f);
          counter.update(tr.id, Point{c.x, c.y}, "vehicle", frame_no);
        }
        counter.retain(tracker.live_ids());
        ++tracked_frames;
        next_expected = min_no + 1;
        {
          const size_t d = q_frame.size();
          if (d > qmax_seen) qmax_seen = d;
          qlast = d;
        }
        {
          const auto now = std::chrono::steady_clock::now();
          lat_infer.push_back(std::chrono::duration<double, std::milli>(
              r.t_infer_end - r.t_infer_begin).count());
          lat_queue.push_back(std::chrono::duration<double, std::milli>(
              r.t_infer_begin - r.t_enqueue).count());
          lat_total.push_back(std::chrono::duration<double, std::milli>(
              now - r.t_enqueue).count());
        }
      }
    }
    // 收尾：剩余的一并处理
    for (auto& kv : pending) {
      const auto tracks = tracker.update(kv.second);
      for (const auto& tr : tracks) {
        if (!tr.moving) continue;
        const cv::Point2f c((tr.box.x + tr.box.x + tr.box.width) * 0.5f,
                            (tr.box.y + tr.box.y + tr.box.height) * 0.5f);
        counter.update(tr.id, Point{c.x, c.y}, "vehicle", kv.first + 1);
      }
      counter.retain(tracker.live_ids());
      ++tracked_frames;
    }
  });

  reader.join();
  for (auto& t : infer_threads) t.join();
  q_det.close();
  consumer.join();

  const auto t1 = std::chrono::steady_clock::now();
  const double wall = std::chrono::duration<double>(t1 - t0).count();

  std::printf("===== 流水线结果 =====\n");
  std::printf("读帧     : %d\n", read_count.load());
  std::printf("推理完成 : %d\n", infer_count.load());
  std::printf("跟踪完成 : %lld\n", tracked_frames);
  std::printf("墙钟     : %.2f 秒\n", wall);
  std::printf("吞吐     : %.1f fps\n",
              tracked_frames / (wall > 0 ? wall : 1));
  auto stat_line = [](const char* tag, std::vector<double>& v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    const size_t n = v.size();
    std::printf("%-18s 平均 %7.1f  中位 %7.1f  P90 %7.1f  最慢 %7.1f\n",
                tag, sum / n, v[n / 2], v[static_cast<size_t>(n * 0.9)], v[n - 1]);
  };
  if (!lat_total.empty()) {
    std::printf("\n--- 延迟拆解（毫秒）---\n");
    stat_line("纯推理耗时:", lat_infer);
    stat_line("排队等待:", lat_queue);
    stat_line("端到端合计:", lat_total);
  }
  if (realtime_fps > 0.0) {
    std::printf("\n--- 实时流模拟（送帧 %.1f fps，队列上限 %d）---\n", realtime_fps, buf);
    std::printf("待推理队列最大深度: %zu / %d\n", qmax_seen, buf);
    std::printf("收尾时队列深度: %zu\n", qlast);
    // 判据看队列深度，不看吞吐：送帧速度受节奏控制，吞吐必然约等于送帧速度。
    // 队列能保持在低位，说明处理速度跟得上，不会累积延迟。
    const double fill = buf > 0 ? static_cast<double>(qmax_seen) / buf : 0.0;
    std::printf("队列占用率: %.0f%%  →  %s\n", fill * 100.0,
                fill < 0.25 ? "跟得上，延迟稳定"
                            : (fill < 0.75 ? "勉强跟得上，偶有积压" : "跟不上，延迟会持续上涨"));
  }
  std::printf("编号总数 : %lld\n", tracker.total_ids());
  for (size_t i = 0; i < cfg.lines.size(); ++i) {
    std::printf("  %s = %d\n", cfg.lines[i].name.c_str(), counter.count(i));
  }
  return 0;
}

