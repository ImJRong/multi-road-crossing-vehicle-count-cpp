// 极简 HTTP 服务：给网页看画面、看数字、下命令。
//
// 只用 POSIX socket + OpenCV，不引第三方库（跟 mini_json.hpp 的思路一致）。
//
// 端点:
//   GET /                 页面本身(web/index.html)
//   GET /stream           MJPEG 推流, 服务端已经画好的帧
//   GET /api/configs      configs/ 下有哪些配置
//   GET /api/state        当前状态: 各线计数/跟踪数/fps/时间
//   GET /api/events       最近若干条过车记录(带抓拍图名)
//   GET /api/start?config=test1
//   GET /api/stop
//   GET /api/restart      从第一帧重来, 计数清零
//   GET /api/pause?on=1   暂停/继续
//   GET /api/param?conf=0.3&iou=0.5     改参数(暂停后生效)
//   GET /api/overlay?box=1&id=0&line=1&roi=1   图层开关
//   GET /snap/<文件名>    抓拍小图
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
};

class WebServer {
 public:
  WebServer(int port, std::string html_path, std::string config_dir,
            std::string snap_dir);
  ~WebServer();

  void start();   // 起监听线程, 立刻返回
  void stop();    // 关监听, 等线程退出

  // ---- 主循环调用 ----
  void push_frame(const cv::Mat& frame);                    // 每帧一次, 内部编码成 JPEG
  void update_stats(const std::vector<LineStat>& lines, int tracks,
                    double fps, double infer_ms, int frame_no);
  void add_event(const Event& e);
  void clear_events();

  Command take_command();          // 取走命令(取完变 kCmdNone)
  std::string start_config();      // kCmdStart 时带的配置名
  Overlay overlay() const;         // 当前图层开关
  float conf() const;              // 当前置信度阈值
  float iou() const;               // 当前 NMS IoU
  int state() const { return state_.load(); }
  void set_state(int s) { state_.store(s); }   // 0 空闲 1 运行 2 暂停

 private:
  void accept_loop();
  void handle_client(int fd);
  void serve_index(int fd);
  void serve_stream(int fd);
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

  // 最新一帧(JPEG 字节), 推流线程从这里取
  std::mutex frame_mu_;
  std::condition_variable frame_cv_;
  std::vector<uchar> jpeg_;
  uint64_t frame_seq_ = 0;          // 每来一帧 +1, 推流端靠它判断"有没有新帧"

  // 状态数字
  std::mutex stat_mu_;
  std::string state_json_ = "{}";

  // 事件列表(只留最近 200 条)
  std::mutex event_mu_;
  std::deque<Event> events_;

  // 网页下发的控制
  std::atomic<int> cmd_{kCmdNone};
  std::mutex cmd_mu_;
  std::string start_config_;

  std::atomic<int> overlay_mask_{0b1111};   // bit0 box bit1 id bit2 line bit3 roi
  std::atomic<float> conf_{0.3f};
  std::atomic<float> iou_{0.5f};
  std::atomic<int> state_{0};
};
