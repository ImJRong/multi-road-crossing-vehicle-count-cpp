// 极简 HTTP 服务：给网页看画面、看数字、下命令。
//
// 只用 POSIX socket + OpenCV，不引第三方库（跟 mini_json.hpp 的思路一致）。
//
// 端点:
//   GET /                 页面本身(web/index.html)
//   GET /stream[?cam=N][&thumb=1]  MJPEG 推流(thumb=1 推缩略图, 默认推原图)
//   GET /api/configs      configs/ 下有哪些配置
//   GET /api/state[?cam=N] 当前状态: 各线计数/跟踪数/fps/时间
//   GET /api/events[?cam=N] 最近若干条过车记录(带抓拍图名)
//   GET /api/start?config=test1
//   GET /api/stop
//   GET /api/restart[?cam=N] 从第一帧重来, 计数清零
//   GET /api/pause?on=1[&cam=N] 暂停/继续
//   GET /api/param?conf=0.3&iou=0.5[&cam=N] 改参数(暂停后生效)
//   GET /api/overlay?box=1&id=0&line=1&roi=1[&cam=N] 图层开关
//   GET /api/quit          结束检测(停止推理), 网页继续留着
//   GET /snap/<文件名>    抓拍小图
//
// 多路(cam)说明:
//   单路版只用一个通道(cam=0), 行为跟以前完全一样.
//   四路版开 4 个通道, 每路一套独立的帧/状态/参数/命令.
//
// 节流:
//   每路记"有多少客户端在看全图 / 看缩略图".
//   没人看的那路, 连 JPEG 编码都不做 —— 省 CPU 和带宽.
//
// 线程模型:
//   一个 accept 线程, 每个客户端一个 detached 线程.
//   推流线程靠条件变量等新帧, 不空转.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

// 最多几路同时推
constexpr int kMaxCams = 4;
// 缩略图宽度(监控台卡片用)
constexpr int kThumbWidth = 480;

// 网页上能勾的图层。服务端每帧按这个决定画什么, 不是前端叠上去的
struct Overlay {
  bool box = true;    // 检测框
  bool id = true;     // 跟踪编号 #12
  bool line = true;   // 计数线和线上数字
  bool roi = true;    // 检测区域轮廓
};

// 一条计数线当前的数字
struct LineStat {
  std::string name;
  int count = 0;
};

// 一条过车事件, 网页右侧列表用
struct Event {
  int frame = 0;
  std::string line;
  std::string cls;
  int x = 0;
  int y = 0;
  std::string snap;   // 抓拍文件名, 没抓就是空
  double ts = 0.0;    // 事件发生的秒数(从启动算)
};

// 网页发过来的命令。主循环每帧取一次, 取完清零
enum Command {
  kCmdNone = 0,
  kCmdStart,     // 开始跑(带一个配置名)
  kCmdStop,      // 停下来, 释放视频和模型
  kCmdRestart,   // 从第一帧重来, 计数清零
  kCmdQuit,      // 结束检测(停止推理), 网页留着
};

class WebServer {
 public:
  WebServer(int port, std::string html_path, std::string config_dir,
            std::string snap_dir);
  ~WebServer();

  void start();   // 起监听线程, 立刻返回
  void stop();    // 关监听, 等线程退出

  // ---- 主循环调用 ----
  // 每帧一次. 只在有人看这一路时才编码 JPEG(没人看就直接返回, 省 CPU)
  void push_frame(const cv::Mat& frame, int cam = 0);
  void update_stats(const std::vector<LineStat>& lines, int tracks,
                    double fps, double infer_ms, int frame_no, int cam = 0);
  void add_event(const Event& e, int cam = 0);
  void clear_events(int cam = 0);

  Command take_command(int cam = 0);          // 取走命令(取完变 kCmdNone)
  std::string start_config();                 // kCmdStart 时带的配置名
  Overlay overlay(int cam = 0) const;         // 当前图层开关
  float conf(int cam = 0) const;              // 当前置信度阈值
  float iou(int cam = 0) const;               // 当前 NMS IoU
  // 启动时把这一路配置文件里的 conf/iou 灌进来当初值.
  // 只写内存: 网页上改的也只活在内存里, 不碰 config 文件;
  // 进程重开又从配置读一遍, 等于"配置是底盘, 页面是临时覆盖".
  void init_params(int cam, float conf, float iou);
  int state() const { return state_.load(); }
  void set_state(int s) { state_.store(s); }   // 0 空闲 1 运行 2 暂停

 private:
  void accept_loop();
  void handle_client(int fd);
  void serve_index(int fd);
  void serve_stream(int fd, int cam, bool thumb);
  void serve_json(int fd, const std::string& body);
  void serve_snap(int fd, const std::string& name);
  void route(int fd, const std::string& method, const std::string& path,
             const std::string& query);

  int port_;
  std::string html_path_;
  std::string config_dir_;
  std::string snap_dir_;

  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;

  // 最新一帧(JPEG 字节). 每路两套: 全图 + 缩略图
  std::mutex frame_mu_[kMaxCams];
  std::condition_variable frame_cv_[kMaxCams];
  std::vector<uchar> jpeg_[kMaxCams];
  std::vector<uchar> thumb_jpeg_[kMaxCams];
  uint64_t frame_seq_[kMaxCams] = {0, 0, 0, 0};   // 每来一帧 +1, 推流端靠它判断"有没有新帧"

  // 有多少客户端在看这一路. 0 就不编码不推(节流)
  // 注意: atomic 的默认构造不做值初始化, 必须显式给 0, 否则是垃圾值
  std::atomic<int> full_viewers_[kMaxCams] = {};
  std::atomic<int> thumb_viewers_[kMaxCams] = {};

  // 状态数字. 每路一份
  std::mutex stat_mu_;
  std::string state_json_[kMaxCams] = {"{}", "{}", "{}", "{}"};

  // 事件列表(只留最近 200 条). 每路一份
  std::mutex event_mu_;
  std::deque<Event> events_[kMaxCams];

  // 网页下发的控制. 命令每路一份
  std::atomic<int> cmd_[kMaxCams] = {kCmdNone, kCmdNone, kCmdNone, kCmdNone};
  std::mutex cmd_mu_;
  std::string start_config_;

  std::atomic<int> overlay_mask_[kMaxCams] = {0b1111, 0b1111, 0b1111, 0b1111};
  std::atomic<float> conf_[kMaxCams] = {0.3f, 0.3f, 0.3f, 0.3f};
  std::atomic<float> iou_[kMaxCams] = {0.5f, 0.5f, 0.5f, 0.5f};
  std::atomic<int> state_{0};

  // 系统状态：CPU 使用率 / 温度 / NPU GPU 频率
  unsigned long long last_cpu_total_ = 0;
  unsigned long long last_cpu_idle_ = 0;
  double cpu_usage_ = 0.0;
  double temp_c_ = 0.0;
  int npu_mhz_ = 0;
  int gpu_mhz_ = 0;
  int npu_c0_ = 0;
  int npu_c1_ = 0;
  int npu_c2_ = 0;
  void sample_sys();
};
