// 视频流硬件解码实现：同进程 libavformat 解封装 + mpp 解码。
// 数据流：av_read_frame -> decode_put_packet -> decode_get_frame(NV12) -> cvtColor -> BGR。

#include "rkmpp_reader.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavformat/avformat.h>
}

#include "rk_mpi.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_buffer.h"
#include "mpp_dec_cfg.h"

namespace {

constexpr int kPktPoolSize = 8;      // 输入队列实测深度 4，留一倍余量
constexpr int kSpinGuard = 4000;     // 连续多少轮不出帧就判失败

// 中断回调：av_read_frame / avformat_open_input 内部会反复调它，
// 返回非 0 就立刻放弃当前阻塞操作。
int interrupt_cb(void* opaque) {
  auto* p = static_cast<std::atomic<bool>*>(opaque);
  return (p && p->load()) ? 1 : 0;
}

bool is_rtsp_url(const std::string& u) {
  return u.rfind("rtsp://", 0) == 0 || u.rfind("rtsps://", 0) == 0;
}

}  // namespace

// ---------------------------------------------------------------- 生命周期

MppReader::~MppReader() { release(); }

bool MppReader::available() { return true; }

void MppReader::cancel() { cancel_.store(true); }

void MppReader::release() {
  cancel_.store(true);        // 先叫停，释放过程中任何阻塞操作都会立刻返回

  if (ctx_) {
    MppCtx ctx = static_cast<MppCtx>(ctx_);
    MppApi* mpi = static_cast<MppApi*>(mpi_);
    if (mpi) mpi->reset(ctx);
    mpp_destroy(ctx);
    ctx_ = nullptr;
    mpi_ = nullptr;
  }
  if (pkt_) {
    MppPacket p = static_cast<MppPacket>(pkt_);
    mpp_packet_deinit(&p);
    pkt_ = nullptr;
  }

  close_input();

  ready_.clear();
  nv12_.release();
  bgr_.release();
  ok_ = false;
}

// ---------------------------------------------------------------- 解封装

bool MppReader::open_input(const std::string& url, int timeout_ms) {
  avformat_network_init();

  AVFormatContext* fmt = avformat_alloc_context();
  if (!fmt) {
    err_ = "分配格式上下文失败";
    return false;
  }
  // interrupt_callback 必须在 open 之前注册, 否则 open 自身可能阻塞十几秒.
  fmt->interrupt_callback.callback = interrupt_cb;
  fmt->interrupt_callback.opaque = &cancel_;

  AVDictionary* opts = nullptr;
  const int sec = timeout_ms > 0 ? timeout_ms / 1000 : 10;
  const std::string us = std::to_string(sec * 1000000);
  if (is_rtsp_url(url)) {
    // TCP：UDP 在这块板子上实测丢包，日志一片 "RTP packets lost"
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    // RTSP 的 "timeout" 是等待入站连接的监听超时, 设置后 ffmpeg 会进入 listen 模式,
    // 报 Unable to open RTSP for listening / Cannot assign requested address.
    // socket 读超时对应 stimeout (微秒, ffmpeg 4.4 的选项名; 5.x 改名为 timeout,
    // 语义不同, 不可混用).
    av_dict_set(&opts, "stimeout", us.c_str(), 0);
  } else {
    av_dict_set(&opts, "timeout", us.c_str(), 0);
  }

  int ret = avformat_open_input(&fmt, url.c_str(), nullptr, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    char buf[128] = {0};
    av_strerror(ret, buf, sizeof(buf));
    err_ = std::string("打不开视频流: ") + buf;
    fmt_ = nullptr;
    return false;
  }
  fmt_ = fmt;

  if (avformat_find_stream_info(fmt, nullptr) < 0) {
    err_ = "读不到流信息";
    return false;
  }
  const int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (vs < 0) {
    err_ = "这条流里没有视频";
    return false;
  }
  vstream_ = vs;

  // 码流格式检查：AVCC（长度前缀）喂给 mpp 会全黑，这里直接拒绝并让上层退回软解
  AVCodecParameters* par = fmt->streams[vs]->codecpar;
  if (par->codec_id != AV_CODEC_ID_H264) {
    err_ = "硬解只支持 H.264";
    return false;
  }

  // 记下 SPS/PPS：RTSP 流本身可能不带，只能从这里拿
  extradata_.clear();
  if (par->extradata && par->extradata_size > 0) {
    extradata_.assign(par->extradata, par->extradata + par->extradata_size);
  }
  seen_key_ = false;

  pkt_pool_.clear();
  for (int i = 0; i < kPktPoolSize; ++i) {
    AVPacket* p = av_packet_alloc();
    if (!p) {
      err_ = "分配 AVPacket 失败";
      return false;
    }
    pkt_pool_.push_back(p);
  }
  pkt_idx_ = 0;
  apkt_ = pkt_pool_[0];

  return true;
}

void MppReader::close_input() {
  for (void* p : live_) {
    AVPacket* ap = static_cast<AVPacket*>(p);
    if (ap) av_packet_unref(ap);
  }
  live_.clear();

  for (void* p : pkt_pool_) {
    AVPacket* ap = static_cast<AVPacket*>(p);
    if (ap) av_packet_free(&ap);
  }
  pkt_pool_.clear();
  apkt_ = nullptr;
  pkt_idx_ = 0;

  if (fmt_) {
    AVFormatContext* fmt = static_cast<AVFormatContext*>(fmt_);
    avformat_close_input(&fmt);
    fmt_ = nullptr;
    avformat_network_deinit();
  }
  vstream_ = -1;
  have_pkt_ = false;
  seen_key_ = false;
  extradata_.clear();
}

int MppReader::read_packet() {
  if (!fmt_) return -1;
  AVFormatContext* fmt = static_cast<AVFormatContext*>(fmt_);
  if (cancel_.load()) return -1;

  for (;;) {
    AVPacket* ap = static_cast<AVPacket*>(apkt_);
    if (!ap) return -1;
    int ret = av_read_frame(fmt, ap);
    if (ret == AVERROR(EAGAIN)) continue;
    if (ret == AVERROR_EOF) return 0;
    if (ret < 0) {
      if (cancel_.load()) {
        err_ = "已取消";
      } else {
        char buf[128] = {0};
        av_strerror(ret, buf, sizeof(buf));
        err_ = std::string("读流失败: ") + buf;
      }
      return -1;
    }
    if (ap->stream_index != vstream_) {
      av_packet_unref(ap);      // 不是视频包（音频/数据），丢掉继续
      continue;
    }
    // 关键帧之前的 P 帧解不出东西（缺参考帧），喂进去只会把队列塞满。
    // 实测 test2 的流第一个 IDR 在第 66 个包，前 66 个全是 P 帧。
    if (!seen_key_) {
      if (!(ap->flags & AV_PKT_FLAG_KEY)) {
        av_packet_unref(ap);
        continue;
      }
      seen_key_ = true;
    }
    return 1;
  }
}

void MppReader::next_packet() {
  pkt_idx_ = (pkt_idx_ + 1) % static_cast<int>(pkt_pool_.size());
  apkt_ = pkt_pool_[pkt_idx_];
}

bool MppReader::feed_extradata() {
  if (extradata_.empty() || !ctx_ || !mpi_ || !pkt_) return true;
  MppCtx ctx = static_cast<MppCtx>(ctx_);
  MppApi* mpi = static_cast<MppApi*>(mpi_);
  MppPacket p = static_cast<MppPacket>(pkt_);

  mpp_packet_set_data(p, extradata_.data());
  mpp_packet_set_size(p, extradata_.size());
  mpp_packet_set_pos(p, extradata_.data());
  mpp_packet_set_length(p, extradata_.size());

  for (int i = 0; i < 500; ++i) {
    const MPP_RET pr = mpi->decode_put_packet(ctx, p);
    if (pr == MPP_OK) return true;
    drain();
    if (pr != MPP_ERR_BUFFER_FULL) break;
    ::usleep(200);
  }
  return false;      // 喂不进去不算致命，继续按正常流程走
}

void MppReader::release_pkts() {
  if (live_.empty()) return;
  AVPacket* ap = static_cast<AVPacket*>(live_.front());
  live_.pop_front();
  av_packet_unref(ap);
}

// ---------------------------------------------------------------- 解码器

bool MppReader::init_decoder() {
  MppCtx ctx = nullptr;
  MppApi* mpi = nullptr;
  if (mpp_create(&ctx, &mpi) != MPP_OK) {
    err_ = "mpp_create 失败";
    return false;
  }
  if (mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC) != MPP_OK) {
    mpp_destroy(ctx);
    err_ = "mpp_init 失败";
    return false;
  }

  MppDecCfg cfg = nullptr;
  mpp_dec_cfg_init(&cfg);
  if (mpi->control(ctx, MPP_DEC_GET_CFG, cfg) == MPP_OK &&
      mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1) == MPP_OK) {
    mpi->control(ctx, MPP_DEC_SET_CFG, cfg);
    // MPP_DEC_SET_CFG 之后解码器接管 cfg 的所有权, 再调用 mpp_dec_cfg_deinit
    // 会构成双重释放, 进程以 "free(): invalid size" 退出. 此处不可恢复该调用.
  } else {
    mpp_dec_cfg_deinit(&cfg);
    mpp_destroy(ctx);
    err_ = "设 split_parse 失败";
    return false;
  }

  MppPacket packet = nullptr;
  if (mpp_packet_init(&packet, nullptr, 0) != MPP_OK) {
    mpp_destroy(ctx);
    err_ = "packet_init 失败";
    return false;
  }

  ctx_ = ctx;
  mpi_ = mpi;
  pkt_ = packet;
  return true;
}

void MppReader::on_info_change(void* raw) {
  MppFrame frame = static_cast<MppFrame>(raw);
  MppCtx ctx = static_cast<MppCtx>(ctx_);
  MppApi* mpi = static_cast<MppApi*>(mpi_);

  const int stride_w = static_cast<int>(mpp_frame_get_hor_stride(frame));
  const int stride_h = static_cast<int>(mpp_frame_get_ver_stride(frame));
  w_ = static_cast<int>(mpp_frame_get_width(frame));
  h_ = static_cast<int>(mpp_frame_get_height(frame));

  // 此处不创建外部 buffer group, 由解码器自行分配.
  // 外部 group (DRM/ION) 会使 mpp_buffer_get_ptr 返回的内存 CPU 访问极慢.
  mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);

  if (stride_w > 0 && stride_h > 0) {
    nv12_.create(stride_h + stride_h / 2, stride_w, CV_8UC1);
    bgr_.create(h_, w_, CV_8UC3);
  }
  if (!logged_size_) {
    logged_size_ = true;
    std::fprintf(stderr, "[硬解] %dx%d, stride %dx%d, 转换%s\n", w_, h_,
                 stride_w, stride_h,
                 (stride_w == w_ && stride_h == h_) ? "无补白" : "需裁边");
  }
}

void MppReader::drain() {
  MppCtx ctx = static_cast<MppCtx>(ctx_);
  MppApi* mpi = static_cast<MppApi*>(mpi_);
  if (!ctx || !mpi) return;

  for (;;) {
    MppFrame frame = nullptr;
    if (mpi->decode_get_frame(ctx, &frame) != MPP_OK || !frame) break;

    if (mpp_frame_get_info_change(frame)) {
      on_info_change(frame);
      mpp_frame_deinit(&frame);
      continue;
    }
    if (mpp_frame_get_eos(frame)) {
      mpp_frame_deinit(&frame);
      break;
    }
    // discard 的帧是明确作废的残帧，丢掉。
    // 注意不能拿 errinfo 判断：这一路几乎每帧都带 errinfo，用它过滤会把
    // 200 帧筛到只剩 20 多帧。
    if (mpp_frame_get_discard(frame)) {
      release_pkts();
      mpp_frame_deinit(&frame);
      continue;
    }
    cv::Mat out;
    if (frame_to_bgr(frame, out)) {
      ready_.push_back(std::move(out));
      release_pkts();      // 解出一帧 = 解码器已经消费掉最老的那个包
    }
    mpp_frame_deinit(&frame);
  }
}

bool MppReader::frame_to_bgr(void* raw, cv::Mat& bgr) {
  MppFrame frame = static_cast<MppFrame>(raw);
  MppBuffer mb = mpp_frame_get_buffer(frame);
  if (!mb) return false;

  const int hs = static_cast<int>(mpp_frame_get_hor_stride(frame));
  const int vs = static_cast<int>(mpp_frame_get_ver_stride(frame));
  if (hs <= 0 || vs <= 0 || w_ <= 0 || h_ <= 0) return false;

  void* base = mpp_buffer_get_ptr(mb);
  if (!base) return false;

  if (nv12_.rows != vs + vs / 2 || nv12_.cols != hs) {
    nv12_.create(vs + vs / 2, hs, CV_8UC1);
  }
  // 直接包住 DMA 内存，不拷贝：cvtColor 会把它当只读源
  cv::Mat src(vs + vs / 2, hs, CV_8UC1, base);

  if (hs == w_ && vs == h_) {
    // stride 正好等于可见尺寸：一步转换，零拷贝进去
    if (bgr_.rows != h_ || bgr_.cols != w_) bgr_.create(h_, w_, CV_8UC3);
    cv::cvtColor(src, bgr_, cv::COLOR_YUV2BGR_NV12);
    bgr = bgr_.clone();      // 交出去的必须独立，bgr_ 下一帧还要复用
    return true;
  }

  // 有对齐补白：先转成 stride 尺寸，再裁出可见区域
  cv::Mat conv;
  cv::cvtColor(src, conv, cv::COLOR_YUV2BGR_NV12);
  const int cw = std::min(w_, conv.cols);
  const int ch = std::min(h_, conv.rows);
  bgr = conv(cv::Rect(0, 0, cw, ch)).clone();
  return true;
}

// ---------------------------------------------------------------- 对外接口

bool MppReader::open(const std::string& url, int timeout_ms) {
  release();
  cancel_.store(false);
  err_.clear();
  url_ = url;
  logged_size_ = false;

  if (!open_input(url, timeout_ms)) {
    close_input();
    return false;
  }
  if (!init_decoder()) {
    close_input();
    return false;
  }

  ok_ = true;
  feed_extradata();
  cv::Mat first;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (read(first) && !first.empty()) {
      ready_.push_front(std::move(first));
      return true;
    }
    if (!err_.empty()) {
      ok_ = false;
      release();
      return false;
    }
  }
  ok_ = false;
  err_ = "等第一帧超时";
  release();
  return false;
}

bool MppReader::read(cv::Mat& bgr) {
  if (!ok_) return false;
  if (!ready_.empty()) {
    bgr = std::move(ready_.front());
    ready_.pop_front();
    return true;
  }

  MppCtx ctx = static_cast<MppCtx>(ctx_);
  MppApi* mpi = static_cast<MppApi*>(mpi_);
  MppPacket p = static_cast<MppPacket>(pkt_);
  if (!ctx || !mpi || !p) return false;

  int guard = 0;
  int full_spin = 0;
  const int pool_n = static_cast<int>(pkt_pool_.size());
  for (;;) {
    if (cancel_.load()) {
      err_ = "已取消";
      return false;
    }

    // 1) 没数据就去取一个包（可能阻塞，可被 cancel 打断）
    if (!have_pkt_) {
      // 关键限制: av_read_frame 入口会先 av_packet_unref(pkt)，
      // 所以绝不能拿一个还挂在 live_ 里的槽去接新包 —— 那会把解码器
      // 还在用的码流缓冲提前释放掉。槽位都在用就先收帧，腾出来再取。
      while (pool_n > 0 && static_cast<int>(live_.size()) >= pool_n) {
        drain();
        if (!ready_.empty()) {
          bgr = std::move(ready_.front());
          ready_.pop_front();
          return true;
        }
        if (++full_spin > 400) {
          err_ = "解码器一直满";
          return false;
        }
        ::usleep(200);
      }
      const int r = read_packet();
      if (r <= 0) {
        drain();                                  // 把解码器里剩的收干净
        if (!ready_.empty()) {
          bgr = std::move(ready_.front());
          ready_.pop_front();
          return true;
        }
        if (r == 0) err_ = "视频流结束";
        return false;
      }
      have_pkt_ = true;
    }

    // 2) 提交。队列满就重试，绝不丢数据
    AVPacket* ap = static_cast<AVPacket*>(apkt_);
    mpp_packet_set_data(p, ap->data);
    mpp_packet_set_size(p, ap->size);
    mpp_packet_set_pos(p, ap->data);
    mpp_packet_set_length(p, ap->size);

    const MPP_RET pr = mpi->decode_put_packet(ctx, p);
    if (pr == MPP_OK) {
      live_.push_back(apkt_);
      have_pkt_ = false;
      next_packet();
      full_spin = 0;
    } else if (pr != MPP_ERR_BUFFER_FULL) {
      err_ = "解码器拒收数据";
      return false;
    } else {
      // 队列满：下一轮 drain 会腾位置。连续多次仍满说明解码器真卡住了
      if (++full_spin > 200) {
        err_ = "解码器一直满";
        return false;
      }
    }

    // 3) 收帧。稳态下这里必有一帧，不需要 sleep
    drain();
    if (!ready_.empty()) {
      bgr = std::move(ready_.front());
      ready_.pop_front();
      return true;
    }

    if (++guard > kSpinGuard) {
      err_ = "解码器长时间不出帧";
      return false;
    }
    // 只有真的空转时才让一下 CPU，正常路径走不到这里
    if (pr == MPP_ERR_BUFFER_FULL) ::usleep(200);
  }
}

