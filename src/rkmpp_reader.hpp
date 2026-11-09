// 视频流硬件解码读帧器（RK3588 专用）。
// libavformat 拉流解封装，mpp 解码输出 NV12 再转 BGR。
//
// 用法：MppReader r; r.open(url) 后循环 r.read(bgr)。提前打断先 r.cancel() 再 join 读线程。
// USE_MPP=OFF 时 available() 为 false，上层退回 OpenCV。只支持 rtsp:// 与 rtsps://。

#ifndef TRAFFIC_RKMPP_READER_HPP
#define TRAFFIC_RKMPP_READER_HPP

#include <atomic>
#include <deque>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

class MppReader {
 public:
  MppReader() = default;
  ~MppReader();
  MppReader(const MppReader&) = delete;
  MppReader& operator=(const MppReader&) = delete;

  // 这个二进制里有没有编进硬件解码
  static bool available();

  // 打开一路视频流。timeout_ms 是拉流握手 + 等第一帧的总上限。
  // 成功时宽高已填好（width()/height()），第一帧被缓存，下一次 read() 直接交出去。
  bool open(const std::string& url, int timeout_ms = 10000);

  // 取下一帧 BGR。返回 false 表示断了/出错了，调用方应当重连。
  bool read(cv::Mat& bgr);

  // 从另一个线程打断阻塞中的 av_read_frame。只能用来叫停，不能用来复用。
  void cancel();

  void release();

  int width() const { return w_; }
  int height() const { return h_; }
  const std::string& last_error() const { return err_; }

 private:
  bool open_input(const std::string& url, int timeout_ms);
  void close_input();
  int read_packet();          // 1=拿到包 0=流结束 -1=出错
  void next_packet();         // 换到池里下一个 AVPacket
  void release_pkts();        // 从 live_ 头部放掉一个已被消费的包
  bool feed_extradata();      // 把 SPS/PPS 当第一个包喂进去
  bool init_decoder();
  void drain();               // 把解码器里已好的帧收上来
  bool frame_to_bgr(void* frame, cv::Mat& bgr);
  void on_info_change(void* frame);

  std::string url_;
  std::string err_;

  bool ok_ = false;
  int w_ = 0, h_ = 0;

  // ---- libavformat（全部用 void* 存，避免把它的头文件带进上层）----
  void* fmt_ = nullptr;                 // AVFormatContext*
  void* apkt_ = nullptr;                // AVPacket*（当前正在装的那个）
  std::vector<void*> pkt_pool_;         // AVPacket* 池
  int pkt_idx_ = 0;
  std::deque<void*> live_;              // 已提交、还被解码器引用着的包
  int vstream_ = -1;
  std::atomic<bool> cancel_{false};

  bool have_pkt_ = false;               // apkt_ 里有数据还没提交
  bool seen_key_ = false;               // 见过第一个关键帧没有
  std::vector<uint8_t> extradata_;      // SPS/PPS，打开后先喂一次

  // ---- 解码器 ----
  void* ctx_ = nullptr;                 // MppCtx
  void* mpi_ = nullptr;                 // MppApi*
  void* pkt_ = nullptr;                 // MppPacket（全程复用同一个）

  // ---- 转换缓冲，预分配复用 ----
  cv::Mat nv12_;                // vs*3/2 行 x hs 列
  cv::Mat bgr_;                 // 可见区域的输出，read() 会把它整体搬走
  std::deque<cv::Mat> ready_;   // 已经解好但还没被 read() 取走的帧

  bool logged_size_ = false;    // 分辨率只打印一次
};

#endif  // TRAFFIC_RKMPP_READER_HPP

