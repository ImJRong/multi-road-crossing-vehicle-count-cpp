// 极简 HTTP 服务（只用 POSIX socket + OpenCV，不引第三方库）。
// 端点：/ 页面、/stream MJPEG 推流、/api/state 状态、/api/start|stop|restart|pause|param|overlay|quit 命令、/snap/ 抓拍。
// 每路独立一套帧/状态/参数，没人看的那路不做 JPEG 编码省 CPU。
// 线程模型：一个 accept 线程，每个客户端一个 detached 线程，推流靠条件变量等新帧。

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

  // ---- 视频源: 网页上填地址 -> 主线程起停这一路 ----
  // 这里只存字符串和一个版本号, 绝不起线程:
  // HTTP 线程碰线程生命周期很容易和主线程打架, 统一交给 main 轮询.
  void set_source(int cam, const std::string& url, const std::string& cfg);
  std::string source(int cam = 0) const;
  std::string source_config(int cam = 0) const;
  int source_gen(int cam = 0) const;
  void bump_source_gen(int cam = 0);
  void set_idle(int cam, bool v);
  void request_stop(int cam);
  bool take_stop(int cam);   // 取走标志, 取到返回 true
  std::vector<std::string> config_list() const;
  std::vector<std::string> video_list() const;   // data/videos 下的 mp4
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
  // atomic 的默认构造不做值初始化, 需显式给出初值, 否则为未定义值.
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

  // 视频源: 地址/配置名由 HTTP 线程写, 版本号每改一次 +1;
  // main 发现版本号变了才去停/起这一路的线程, 保证线程只在主线程手上动.
  std::atomic<bool> idle_[kMaxCams] = {true, true, true, true};
  mutable std::mutex src_mu_;
  std::string source_[kMaxCams];
  std::string src_cfg_[kMaxCams];
  std::atomic<int> src_gen_[kMaxCams] = {0, 0, 0, 0};
  // 网页点了"断开": 只停这一路, 不清 source_[]. 再点接入就是原地址.
  std::atomic<bool> src_stop_[kMaxCams] = {false, false, false, false};

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

