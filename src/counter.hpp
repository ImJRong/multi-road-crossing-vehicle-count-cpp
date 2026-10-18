// 穿线计数.
//
// 逻辑跟 Python 版 src/counter.py 的 side() / near_segment() / 过线判断一一对应,
// 两边跑同一段视频, 每条线的数字必须一模一样 —— 这是验收标准.
//
// 核心思路:
//   不做"框和线有没有相交"的几何判断, 而是看"框中心点相对这条线在左边还是右边".
//   同一个跟踪 ID 上一帧在一边、这一帧在另一边 = 它穿过去了.
#pragma once

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"

// 判断点 p 在直线 AB 的哪一侧. 用的是叉积(行列式):
//   结果 > 0 一侧, < 0 另一侧, = 0 正在线上.
// 好处是横线、竖线、斜线用同一个公式, 不用按情况分支.
inline double side(const Point& p, const Point& a, const Point& b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// 线的长度. side() 出来的是"像素 × 线长", 除以线长才是真正的像素距离.
inline double line_length(const Point& a, const Point& b) {
  return std::hypot(b.x - a.x, b.y - a.y);
}

// 点 p 是否落在线段 AB 覆盖的范围内(两端各放宽 pad 像素).
// 做法: 把 p 投影到线段方向上, 看投影比例 t 是否在 0~1 之间.
// 为什么不直接比 x 坐标: 因为线可能是斜的、竖的, 比 x 会算错.
inline bool near_segment(const Point& p, const Point& a, const Point& b, double pad = 0) {
  const double ax = b.x - a.x;
  const double ay = b.y - a.y;
  const double len2 = ax * ax + ay * ay;
  if (len2 <= 0) return false;                       // a 和 b 重合, 这条线无效
  const double t = ((p.x - a.x) * ax + (p.y - a.y) * ay) / len2;
  const double slack = pad / std::sqrt(len2);        // 把像素换算成比例
  return t >= -slack && t <= 1.0 + slack;
}

// 一次过线记录, 对应 Python 版 records 列表里的一条, 后面可以导成 csv
struct Hit {
  std::string line;    // 哪条线
  std::string group;   // 哪个方向分组
  std::string cls;     // 类别
  int frame = 0;       // 第几帧过的线
  int x = 0;           // 过线时的位置
  int y = 0;
};

class Counter {
 public:
  explicit Counter(const std::vector<Line>& lines);

  // 每帧、每辆车调一次. track_id 是跟踪器给的编号, p 是框中心(整图坐标).
  // 注意: 它不关心 p 是车还是别的东西, 只认"同一个 id 有没有换边".
  void update(long long track_id, const Point& p, const std::string& cls, int frame_no);

  int count(size_t line_index) const;                 // 第 i 条线数到多少
  const std::vector<Hit>& hits() const { return hits_; }  // 全部过线记录
  void clear();

 private:
  std::vector<Line> lines_;
  // (线序号, 车 id) -> 上一次记录的那一侧(带符号)
  // 用 map 而不是数组, 因为 id 是流水号, 会一直变大
  std::map<std::pair<size_t, long long>, double> prev_;
  std::vector<int> counters_;
  std::vector<Hit> hits_;
};
