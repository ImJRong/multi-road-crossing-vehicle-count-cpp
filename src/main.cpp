// 多路口车流量统计 —— C++ 版入口.
//
// 一帧的处理顺序:
//   读配置 -> 裁 ROI -> 推理(ONNX 或 RKNN) -> 跟踪发 ID -> 静止目标过滤
//   -> 穿线计数 -> 画框/线/数字 -> 播放或存视频.
//
// 本文件只干三件事, 其余都在别的文件里:
//   1. 路径解析   resolve_config_path / resolve_video_path  —— 从命令行参数找到 json 和视频
//   2. 画图       draw_tracks / draw_counters               —— 框、编号、计数数字
//   3. 主流程     main()                                    —— 把各部分串起来
//
// 跟踪在 tracker.hpp / tracker.cpp, 计数在 counter.cpp, 推理在 detector.cpp 或 rknn_detector.cpp.
// 推理后端在编译时决定(CMake 的 USE_RKNN 开关):
//   USE_RKNN 关 -> detector.cpp      (ONNX Runtime, PC 上调试用)
//   USE_RKNN 开 -> rknn_detector.cpp (RKNN, 板子上跑)
//
// 用法:
//   ./traffic_count test4
// 不带参数就列出现有配置. 退出: 窗口里按 Q 或 Esc.

#include <algorithm>    // std::sort, std::remove_if
#include <array>        // std::array, 存画图用的颜色
#include <chrono>       // steady_clock, 算帧率
#include <cmath>        // std::round, std::fabs
#include <cstdio>       // printf
#include <exception>    // std::exception, 统一捕获错误
#include <filesystem>   // 遍历 configs 目录
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>   // 读视频、缩放、画框画线

#include "config.hpp"     // Config 结构体 + load_config()
#include "counter.hpp"    // 穿线计数: 在线的哪一侧、死区、换边加一
#include "recorder.hpp"   // 把过线记录写成 csv
#include "tracker.hpp"    // 跟踪器: 给检测框发稳定编号
#ifdef USE_RKNN
#include "rknn_detector.hpp"   // 板子上的推理(RKNN)
using Detector = RknnDetector; // 换个名字, 下面代码不用改
#else
#include "detector.hpp"        // PC 上的推理(ONNX Runtime)
#endif


// 模型路径. 编译时定死, 运行时不改.
// 两种后端用的是同一份权重, 只是文件格式不同, 所以路径要分开写.
#ifndef TRAFFIC_MODEL
#ifdef USE_RKNN
#define TRAFFIC_MODEL TRAFFIC_PROJECT_ROOT "/models/best_detrac.rknn"
#else
#define TRAFFIC_MODEL TRAFFIC_PROJECT_ROOT "/models/best_detrac.onnx"
#endif
#endif


namespace fs = std::filesystem;

// 匿名 namespace: 里面的东西只在本文件可见, 别的 .cpp 调不到.
// 好处是名字不会冲突, 也不用在头文件里声明.
namespace {

// 两个路径常量, 都在编译期由 CMakeLists 灌进来(见 CMakeLists 的 TRAFFIC_PROJECT_ROOT).
const std::string kRoot = TRAFFIC_PROJECT_ROOT;  // 工程根目录, configs/ data/ models/ 都在这下面
const std::string kModel = TRAFFIC_MODEL;        // 模型文件的完整路径

// 把命令行参数变成配置文件路径. 三种写法都支持:
//   test3             -> <根>/configs/test3.json
//   test3.json        -> <根>/configs/test3.json
//   /tmp/mine.json    -> 原样使用(带斜杠就当完整路径)
std::string resolve_config_path(const std::string& arg) {
  if (arg.find('/') != std::string::npos) return arg;
  if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".json") == 0)
    return kRoot + "/configs/" + arg;
  return kRoot + "/configs/" + arg + ".json";
}

// 不带参数运行时, 列出 configs/ 下所有可用的配置名
std::vector<std::string> list_configs() {
  std::vector<std::string> names;
  const std::string dir = kRoot + "/configs";
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return names;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file()) continue;
    const std::string fn = entry.path().filename().string();
    if (fn.size() > 5 && fn.compare(fn.size() - 5, 5, ".json") == 0)
      names.push_back(fn.substr(0, fn.size() - 5));   // 去掉 .json 后缀
  }
  std::sort(names.begin(), names.end());
  return names;
}

// 试探这个路径是不是能当视频打开. 用"真打开一次"来判断, 比查文件是否存在可靠
// (比如路径拼对了但文件是坏的, 这里也会返回 false)
bool video_ok(const std::string& path) {
  cv::VideoCapture probe(path);
  return probe.isOpened();
}

// 把 json 里的 video 字段变成真正能打开的路径.
// json 里一般写相对路径(如 "data/videos/test3.mp4"), 目的是整个项目文件夹
// 拷到任何机器都不用改配置. 所以这里要试着拼出几种可能, 哪个能打开用哪个.
std::string resolve_video_path(const std::string& video, const std::string& cfg_path) {
  if (!video.empty() && video[0] == '/') return video;   // 写的是绝对路径, 直接用

  // 配置文件自己所在的目录, 用来拼相对路径(比如 json 和视频放一起)
  const size_t slash = cfg_path.find_last_of('/');
  const std::string cfg_dir = slash == std::string::npos ? "" : cfg_path.substr(0, slash + 1);

  const std::string candidates[] = {
      video,                    // 1. 相对当前工作目录
      cfg_dir + video,          // 2. 相对配置文件所在目录
      kRoot + "/" + video,      // 3. 相对工程根目录(实际最常用的是这条)
      cfg_dir + "../" + video,  // 4. 配置文件目录的上一级
  };
  for (const auto& c : candidates) {
    if (video_ok(c)) return c;
  }
  return kRoot + "/" + video;   // 都不行, 返回最后一种, 让调用处去报错
}

// json 里的颜色是 [B, G, R] 三个整数, 转成 OpenCV 的颜色类型
cv::Scalar to_scalar(const std::array<int, 3>& c) { return cv::Scalar(c[0], c[1], c[2]); }

// 我们的 Point(浮点) 转成 OpenCV 的 Point(整数), 画多边形要用
std::vector<cv::Point> to_cv_poly(const std::vector<Point>& pts) {
  std::vector<cv::Point> out;
  for (const auto& p : pts) out.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
  return out;
}

// ---------------- 画图 ----------------

// 画检测框和编号(#12 这种). 具体画什么由配置里的 draw 小节控制.
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
      std::snprintf(txt, sizeof(txt), "#%lld", tr.id);   // 编号, 用来看跟踪稳不稳
      cv::putText(frame, txt, cv::Point(tl.x, std::max(12, tl.y - 6)),
                  cv::FONT_HERSHEY_SIMPLEX, cfg.draw.font_scale,
                  to_scalar(cfg.draw.box_color), cfg.draw.text_width);
    }
  }
}

// 画计数线和每条线的数字.
// 数字默认写在线的上端; 几条线的数字压在一起时, 会自动往下挪开.
void draw_counters(cv::Mat& frame, const Config& cfg, const Counter& counter) {
  std::vector<cv::Rect> placed;   // 已经放过的数字区域, 用来检测重叠
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
    if (L.has_label_at) {                       // 配置里指定了位置
      tx = static_cast<int>(L.label_at.x);
      ty = static_cast<int>(L.label_at.y);
    } else {                                    // 没指定就放在线的上端
      const cv::Point p = (a.y <= b.y) ? a : b;
      tx = p.x + 6;
      ty = (p.y - 10 > ts.height + 6) ? p.y - 10 : p.y + 26;
    }
    // 和已放过的数字重叠就往下挪, 最多试 6 次(防止死循环)
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

// ============================ 主流程 ============================
int main(int argc, char** argv) {
  // --- 参数检查: 必须给一个配置名 ---
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

  // 整个流程包在 try 里: 任何一步抛异常, 统一在这里打印原因, 不会直接崩
  try {
    // --- 1. 读配置, 定位视频 ---
    const std::string cfg_path = resolve_config_path(argv[1]);
    const Config cfg = load_config(cfg_path);                       // 解析 json
    const std::string video_path = resolve_video_path(cfg.video, cfg_path);

    std::printf("配置: %s\n视频: %s\n模型: %s\n", cfg_path.c_str(), video_path.c_str(),
                kModel.c_str());

    // --- 2. 打开视频, 读画面参数 ---
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

    // --- 3. 算 ROI ---
    // 两步: 外接矩形用来裁图(省算力), 多边形用来过滤框中心(排干扰).
    // 外接矩形比多边形大, 所以裁完还得再用多边形筛一次.
    std::vector<cv::Point> poly;
    cv::Rect roi_rect(0, 0, width, height);     // 默认整帧
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

    // --- 4. 建检测器. 这里用的是哪个类, 由编译时的 USE_RKNN 决定 ---
    Detector detector(kModel, cfg.detect.imgsz, cfg.detect.conf, cfg.detect.iou);

    // --- 5. 准备输出 ---
    // output 填 "save" 存 mp4, 填 "show" 开窗口实时播放.
    // 注意 output/ 是相对当前目录的, 所以在哪个目录跑就写到哪.
    cv::VideoWriter writer;
    const bool to_file = (cfg.output != "show");
    if (to_file) {
      const std::string out = "output/" + cfg.name + "_lanes.mp4";
      writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                  cv::Size(width, height));
      std::printf("输出视频: %s\n", out.c_str());
    }

    // --- 6. 计数器和跟踪器, 都是跨帧一直活着的对象 ---
    Counter counter(cfg.lines);
    Tracker tracker;
    cv::Mat frame;
    int frame_no = 0;
    int total_boxes = 0;    // 累计检测框数, 最后算平均值
    int total_tracks = 0;
    const auto t0 = std::chrono::steady_clock::now();   // 计时起点

    // --- 7. 主循环: 每读一帧走一遍 ---
    while (cap.read(frame)) {
      ++frame_no;

      // 7.1 只把 ROI 外接矩形送进模型. 区域外的东西不会被误检, 而且裁小等于放大.
      cv::Mat crop = frame(roi_rect);
      std::vector<Detection> dets = detector.detect(crop);

      // 7.2 模型给的是裁剪图坐标, 加回偏移才是整图坐标. 后面全靠整图坐标.
      for (auto& d : dets) {
        d.x1 += roi_rect.x;
        d.x2 += roi_rect.x;
        d.y1 += roi_rect.y;
        d.y2 += roi_rect.y;
      }

      // 7.3 外接矩形比多边形大, 中心点落在多边形外的框丢掉
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

      // 7.4 跟踪发编号, 然后只把"真的在动"的那些送去计数.
      //     红绿灯、路牌这类一直不动的东西既不算也不画: 它们的框在线上抖来抖去,
      //     会被反复算成过车, 留着只会干扰看图.
      const auto tracks = tracker.update(dets);
      std::vector<TrackView> live;
      live.reserve(tracks.size());
      for (const auto& tr : tracks) {
        if (tr.moving) live.push_back(tr);
      }
      for (const auto& tr : live) {
        const cv::Point2f c = center_of(tr.box);
        counter.update(tr.id, Point{c.x, c.y}, "vehicle", frame_no);   // 判过线
      }
      total_tracks = static_cast<int>(tracker.size());

      // 7.5 画: ROI 轮廓 -> 框和编号 -> 计数线和不数字
      if (!poly.empty() && cfg.draw.roi) {
        cv::polylines(frame, poly, true, to_scalar(cfg.draw.roi_color), 2);
      }
      draw_tracks(frame, live, cfg);
      draw_counters(frame, cfg, counter);

      // 7.6 输出: 存文件 或 显示窗口
      if (to_file) {
        writer.write(frame);
      } else {
        cv::imshow(cfg.name, frame);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q') break;   // Esc / Q 提前退出
      }

      // 7.7 配置里限定了帧数就停(调试时很有用, 不用等整段跑完)
      if (cfg.max_frames > 0 && frame_no >= cfg.max_frames) break;
    }

    // --- 8. 收尾: 关文件, 写过线记录, 打印统计 ---
    if (writer.isOpened()) writer.release();
    write_records(std::string("output/records_") + cfg.name + ".csv", counter);
    std::printf("累计分配编号: %lld 个\n", tracker.total_ids());

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
