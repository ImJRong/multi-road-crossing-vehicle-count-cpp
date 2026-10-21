// 跟踪器：给每一帧的检测框发一个稳定的编号. 实现的是 ByteTrack 的核心逻辑.
//
// 一帧的流程（和 ultralytics 的 bytetrack 对齐）：
//   ① 每条轨迹用卡尔曼预测"这一帧该在哪"
//   ② 本帧的框按分数分两堆：高分(>= high) / 低分(>= low 且 < high)
//   ③ 第一轮：预测位置 × 高分框 → 匈牙利分配（代价 = 1 - IoU）
//   ④ 第二轮：第一轮没配上的轨迹 × 低分框 → 匈牙利（判据更严）
//   ⑤ 高分框里没人认领的 → 建新轨迹发新 ID；低分框没人认领的 → 丢弃
//   ⑥ 两轮都没配上的轨迹 → lost+1，超过 track_buffer 删掉
//
// 为什么第二轮要用低分框：
//   车被挡住时置信度会掉到 0.1~0.25，但位置还对得上. 丢掉这些框等于直接断号，
//   拿它去和"刚才丢了的轨迹"比一次，位置吻合就能续上.
//
// 为什么低分框不能建新轨迹：
//   低分框里混着树影、反光、广告牌边缘. 让它们建轨迹，一帧能冒出十几个假 ID，
//   这些假 ID 在计数线上抖一抖就是一串假计数.
#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"
#include "kalman_track.hpp"

// 对外输出的精简版: 只要编号、位置、动没动这三样
struct TrackView {
  long long id = 0;
  cv::Rect2f box;
  bool moving = false;  // 离出生位置超过 kMinTravel 像素才算真的在开
};

// 轨迹状态
enum class TrackState {
  Tracked,   // 正常跟踪中（这一帧有框）
  Lost,      // 这一帧没配上，还在缓冲期
};

// 一条轨迹的完整状态（内部用，不对外）
struct Track {
  long long id = 0;                   // 编号，从 1 开始只增不减
  cv::Rect2f box;                     // 最后一次看到它的位置
  cv::Point2f origin;                 // 第一次出现时的中心，用来判断"它到底动没动"
  int lost = 0;                       // 连续几帧没匹配上，超过 track_buffer 就删
  double score = 0;                   // 最后一次匹配到的检测分数
  int frame_id = 0;                   // 最后一次被更新的帧序号
  TrackState state = TrackState::Tracked;
  cv::Rect2f pred;                    // 卡尔曼算出的"这一帧它该在哪"，匹配时用它
  KalmanBox kf;                       // 每条轨迹自己的卡尔曼滤波器
};

// 位移小于这个值就当静止目标：红绿灯、路牌、被风吹的广告牌。
// 这类东西不计数也不画框。不能设太大，否则刚进画面的慢车会来不及计数。
const double kMinTravel = 4.0;

// 工具函数
cv::Point2f center_of(const cv::Rect2f& b);                 // 框的中心点
cv::Rect2f to_rect(const Detection& d);                     // 左上右下 -> 左上 + 宽高
double iou_rect(const cv::Rect2f& a, const cv::Rect2f& b);   // 交并比

class Tracker {
 public:
  // track_buffer : 轨迹丢了之后还能活几帧（30 对应约 1.2 秒 @24fps）
  // high_thresh  : 高低分界. >= 它的进高分堆，参与第一轮
  // low_thresh   : 低分下限. 低于它的直接丢掉
  // new_thresh   : 建新轨迹的最低分数（防止勉强及格的框开新户）
  // match_thresh : 第一轮的代价上限. 代价 = 1 - IoU，所以 0.8 相当于 IoU >= 0.2
  explicit Tracker(int track_buffer = 30, double high_thresh = 0.25,
                   double low_thresh = 0.1, double new_thresh = 0.25,
                   double match_thresh = 0.8);

  // 每帧调用一次. 传入这一帧所有检测框, 返回这一帧真正看到的轨迹（带编号）.
  // 坐标和 dets 一样，都是整图坐标.
  std::vector<TrackView> update(const std::vector<Detection>& dets);

  // 当前有多少条轨迹活着（只用于打印统计）
  size_t size() const { return tracks_.size(); }

  // 累计分配过多少个编号（用来判断是不是在疯狂断号）
  long long total_ids() const { return next_id_ - 1; }

 private:
  std::vector<Track> tracks_;   // 这就是"记忆", 跨帧一直存在
  long long next_id_ = 1;       // 下一个要发的编号
  int frame_ = 0;               // 帧序号，从 1 开始

  int track_buffer_ = 30;
  double high_thresh_ = 0.25;
  double low_thresh_ = 0.1;
  double new_thresh_ = 0.25;
  double match_thresh_ = 0.8;
};
