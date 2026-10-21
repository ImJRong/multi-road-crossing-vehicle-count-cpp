// 跟踪器：给每帧检测框发稳定编号，实现 ByteTrack 核心逻辑。
//
// 一帧流程：卡尔曼预测 -> 按分数分高分/低分两堆 -> 高分框匈牙利匹配 -> 低分框救没配上的轨迹
// -> 没认领的高分框建新轨迹 -> 两轮没配上的轨迹 lost+1，超 track_buffer 删除。
// 低分框只续命不建新轨迹，避免树影反光等假框刷出假 ID。

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
  cv::Rect2f box;                     // 最后一次看到它的位置（原始检测框）
  cv::Point2f origin;                 // 第一次出现时的中心，用来判断"它到底动没动"
  int lost = 0;                       // 连续几帧没匹配上，超过 track_buffer 就删
  double score = 0;                   // 最后一次匹配到的检测分数
  int frame_id = 0;                   // 最后一次被更新的帧序号
  TrackState state = TrackState::Tracked;
  cv::Rect2f pred;                    // 卡尔曼算出的"这一帧它该在哪"，匹配时用它
  KalmanBox kf;                       // 每条轨迹自己的卡尔曼滤波器
};

// 位移小于这个值就当静止目标：红绿灯、路牌、被风吹的广告牌。
// 这类目标不计数也不画框。阈值不宜过大，否则刚进入画面的慢速车辆会来不及计数。
const double kMinTravel = 35.0;

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
  // 默认值取 2026-11-08 实测最优（四路 3000 帧对比：编号数 1547 -> 1297，
  // 计数 836 -> 831）。各配置可在 json 的 track 段单独覆盖。
  explicit Tracker(int track_buffer = 30, double high_thresh = 0.45,
                   double low_thresh = 0.30, double new_thresh = 0.45,
                   double match_thresh = 0.8);

  // 每帧调用一次. 传入这一帧所有检测框, 返回这一帧真正看到的轨迹（带编号）.
  // 坐标和 dets 一样，都是整图坐标.
  std::vector<TrackView> update(const std::vector<Detection>& dets);

  // 当前有多少条轨迹活着（只用于打印统计）
  size_t size() const { return tracks_.size(); }

  // 现在还活着的轨迹编号（含 Lost 缓冲期里的）.
  // 给 Counter::retain 用: 不在这份名单里的车, 计数器那边就不用再记着它了.
  std::vector<long long> live_ids() const {
    std::vector<long long> ids;
    ids.reserve(tracks_.size());
    for (const auto& t : tracks_) ids.push_back(t.id);
    return ids;
  }

  // 累计分配的编号总数, 用于判断断号频率
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

