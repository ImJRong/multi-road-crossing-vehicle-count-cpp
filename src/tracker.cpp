#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// ---------------- 工具函数 ----------------

cv::Point2f center_of(const cv::Rect2f& b) {
  return cv::Point2f(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
}

cv::Rect2f to_rect(const Detection& d) {
  return cv::Rect2f(static_cast<float>(d.x1), static_cast<float>(d.y1),
                    static_cast<float>(d.x2 - d.x1), static_cast<float>(d.y2 - d.y1));
}

// 交并比 = 交集面积 / 并集面积, 范围 0~1
double iou_rect(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float iw = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
  const float ih = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
  if (iw <= 0 || ih <= 0) return 0.0;     // 完全不相交
  const float inter = iw * ih;
  return inter / (a.width * a.height + b.width * b.height - inter);
}

namespace {

// ---------------- 匈牙利算法 ----------------
// 求"总代价最小"的完美匹配. 要求 行数 <= 列数.
// 返回 row2col[i] = 该行配到的列号, -1 表示没配上.
//
// 最经典的 O(n³) 实现（Kuhn-Munkres，带势函数 u/v）：
//   每次给一行找增广路，沿途用"顶标"收缩，保证找到的是全局最优。
//   和贪心的区别就在"全局最优"：贪心先到先得，可能让后面的行无路可走。
//
// 注意：它一定会给每一行都配上一列（数学上叫完美匹配），
// 所以调用方必须自己把"代价超过阈值"的配对剔除，否则所有轨迹都会配上。
std::vector<int> hungarian(const std::vector<std::vector<double>>& a) {
  const int n = static_cast<int>(a.size());
  const int m = n > 0 ? static_cast<int>(a[0].size()) : 0;
  std::vector<int> res(n, -1);
  if (n == 0 || m == 0 || n > m) return res;

  const double INF = std::numeric_limits<double>::max() / 4;
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INF);
    std::vector<char> used(m + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    // 沿增广路回溯，更新每一列的归属
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  for (int j = 1; j <= m; ++j) {
    if (p[j] > 0) res[p[j] - 1] = j - 1;
  }
  return res;
}

// 转成"行数 <= 列数"再调 hungarian，返回 row2col
std::vector<int> assign_min_cost(const std::vector<std::vector<double>>& cost) {
  const int n = static_cast<int>(cost.size());
  const int m = n > 0 ? static_cast<int>(cost[0].size()) : 0;
  if (n == 0 || m == 0) return std::vector<int>(n, -1);
  if (n <= m) return hungarian(cost);

  // 行多于列：转置后算，再把结果倒回来
  std::vector<std::vector<double>> t(m, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) t[j][i] = cost[i][j];
  const std::vector<int> col2row = hungarian(t);
  std::vector<int> row2col(n, -1);
  for (int j = 0; j < m; ++j) {
    if (col2row[j] >= 0) row2col[col2row[j]] = j;
  }
  return row2col;
}

}  // namespace

// ---------------- 跟踪器 ----------------

Tracker::Tracker(int track_buffer, double high_thresh, double low_thresh,
                 double new_thresh, double match_thresh)
    : track_buffer_(track_buffer),
      high_thresh_(high_thresh),
      low_thresh_(low_thresh),
      new_thresh_(new_thresh),
      match_thresh_(match_thresh) {}

std::vector<TrackView> Tracker::update(const std::vector<Detection>& dets) {
  ++frame_;

  // ---- ① 让每条轨迹预测"这一帧我该在哪" ----
  // 卡尔曼按上一帧的位置和速度往前推一步. 没这一步就只能拿上一帧的位置去比,
  // 车开得快时框错开很远, IoU 直接掉到阈值以下, 编号就断了.
  for (auto& t : tracks_) t.pred = t.kf.predict();

  // ---- ② 本帧的框按分数分两堆 ----
  std::vector<int> det_high, det_low;
  for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
    const double s = dets[i].score;
    if (s >= high_thresh_) det_high.push_back(i);
    else if (s >= low_thresh_) det_low.push_back(i);
    // 低于 low_thresh 的框直接丢弃，连下标都不留
  }

  // 第一轮参与匹配的轨迹：全部（Tracked 和 Lost 都算）.
  // 为什么 Lost 也要参与：车被挡住几帧后重新出现，必须能接回原来的编号.
  // 如果 Lost 不参加匹配，那 kept 着的轨迹只能等死，"给它留着"就没意义了.
  std::vector<int> all_tracks;
  for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) all_tracks.push_back(i);

  std::vector<bool> track_used(tracks_.size(), false);   // 这一帧已经配到框的轨迹
  std::vector<bool> det_used(dets.size(), false);        // 已经被认领的框

  // 把一条轨迹和它的框绑定：更新位置、校正卡尔曼、状态归位
  auto bind = [&](int ti, int di) {
    Track& t = tracks_[ti];
    t.box = to_rect(dets[di]);
    t.kf.correct(t.box);
    t.score = dets[di].score;
    t.lost = 0;
    t.frame_id = frame_;
    t.state = TrackState::Tracked;
    track_used[ti] = true;
    det_used[di] = true;
  };

  // 一轮关联：轨迹 × 框，代价 = 1 - IoU，超过 thresh 的配对作废
  auto associate = [&](const std::vector<int>& tracks_in, const std::vector<int>& dets_in,
                       double thresh) {
    if (tracks_in.empty() || dets_in.empty()) return;
    std::vector<std::vector<double>> cost(tracks_in.size(),
                                          std::vector<double>(dets_in.size(), 0.0));
    for (size_t r = 0; r < tracks_in.size(); ++r) {
      for (size_t c = 0; c < dets_in.size(); ++c) {
        const double ov = iou_rect(tracks_[tracks_in[r]].pred, to_rect(dets[dets_in[c]]));
        cost[r][c] = 1.0 - ov;      // 越像代价越小，就能用"最小代价"来分配
      }
    }
    const std::vector<int> row2col = assign_min_cost(cost);
    for (size_t r = 0; r < tracks_in.size(); ++r) {
      const int c = row2col[r];
      if (c < 0) continue;
      if (cost[r][c] > thresh) continue;   // 强制剔除：不够像的一律不认
      if (track_used[tracks_in[r]] || det_used[dets_in[c]]) continue;
      bind(tracks_in[r], dets_in[c]);
    }
  };

  // ---- ③ 第一轮：所有轨迹 × 高分框 ----
  // 阈值 0.8 相当于 IoU >= 0.2，比较宽松：高分框可信，位置差一点也认
  associate(all_tracks, det_high, match_thresh_);

  // ---- ④ 第二轮：第一轮没配上的 Tracked 轨迹 × 低分框 ----
  // 阈值 0.5 相当于 IoU >= 0.5，比第一轮严：低分框不可靠，位置必须高度吻合才敢认.
  // 这一轮只看位置，不把分数融进代价（分数低会把代价推过阈值，等于白配）.
  // 只取 Tracked，不含 Lost：Lost 的位置是推算的，拿它和低分框比容易配错.
  std::vector<int> left_track;
  for (int ti : all_tracks) {
    if (track_used[ti]) continue;
    if (tracks_[ti].state == TrackState::Tracked) left_track.push_back(ti);
  }
  associate(left_track, det_low, 0.5);

  // ---- ⑤ 高分框里没人认领的，建新轨迹 ----
  // 低分框没人认领的直接丢弃：它们唯一的用途是救已有轨迹，救不到就没有价值
  for (int di : det_high) {
    if (det_used[di]) continue;
    if (dets[di].score < new_thresh_) continue;   // 连新轨迹的门槛都不到的，不建

    Track t;
    t.id = next_id_++;                       // 编号只增不减
    t.box = to_rect(dets[di]);
    t.origin = center_of(t.box);             // 记下出生位置，后面判"动没动"靠它
    t.score = dets[di].score;
    t.frame_id = frame_;
    t.state = TrackState::Tracked;
    t.pred = t.box;                          // 刚出生，预测位置就是它自己
    t.kf.init(t.box);                        // 卡尔曼从这里开始接管这条轨迹
    tracks_.push_back(t);
    track_used.push_back(true);              // 新建的这条本帧算"已更新"
    det_used[di] = true;
  }

  // ---- ⑥ 两轮都没配上的轨迹：标 Lost，lost+1 ----
  for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
    if (track_used[ti]) continue;
    Track& t = tracks_[ti];
    t.state = TrackState::Lost;
    ++t.lost;
  }

  // ---- 输出：只给这一帧真正看到过的轨迹 ----
  // 车不在画面里就不该画框，也不该送计数
  std::vector<TrackView> out;
  out.reserve(tracks_.size());
  for (auto& t : tracks_) {
    if (t.state != TrackState::Tracked) continue;
    if (t.frame_id != frame_) continue;

    TrackView v;
    v.id = t.id;
    v.box = t.box;
    const cv::Point2f d = center_of(t.box) - t.origin;
    v.moving = cv::norm(d) >= kMinTravel;   // 离出生点够远才算在开
    out.push_back(v);
  }

  // ---- ⑦ 丢太久的轨迹清掉 ----
  // remove_if 只是把要删的挪到末尾并返回分界位置，必须配 erase 才真正删除
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [this](const Track& t) { return t.lost > track_buffer_; }),
                tracks_.end());

  return out;
}
