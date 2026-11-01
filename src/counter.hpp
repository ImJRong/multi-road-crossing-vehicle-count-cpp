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
//
// 举个横线的例子(线是 a=(390,140) -> b=(695,140)):
//   b.x-a.x = 305, b.y-a.y = 0, 所以公式简化成 side = 305 × (p.y - 140)
//   p=(500,100) -> 305 × (-40) = -12200  负数, 在线的上方(图像里 y 越小越靠上)
//   p=(500,180) -> 305 × (+40) = +12200  正数, 在线的下方
//   p=(500,140) -> 0                     正好压在线上
// 数值 12200 ÷ 线长 305 = 40, 正好是点离线的真实像素距离.
inline double side(const Point& p, const Point& a, const Point& b) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// 线的长度. side() 出来的是"像素 × 线长", 除以线长才是真正的像素距离.
inline double line_length(const Point& a, const Point& b) {
  return std::hypot(b.x - a.x, b.y - a.y);
}

// 点 p 是否落在线段 AB 覆盖的范围内(两端各放宽 pad 像素).
//
// 做法: 把 p 投影到线段方向上, 得到一个比例 t.
//   t = 0   点在线段起点 a 那个位置
//   t = 1   点在线段终点 b 那个位置
//   0~1     点在 a 和 b 中间, 就算"在线段上"
//   <0 或 >1 点跑到线的延长线上了, 线段管不到那里
// 为什么不用"比 x 坐标在不在 a.x~b.x 之间": 线可能是竖的、斜的, 比 x 会算错.
//
// 这一句就是"转弯车漏计"的嫌疑人: 车从线的端点外侧绕过去时, t 会跑到 0~1 之外,
// 直接被判成"不在线段上", 这一次过线就不算数了.
inline bool near_segment(const Point& p, const Point& a, const Point& b, double pad = 0) {
  const double ax = b.x - a.x;
  const double ay = b.y - a.y;
  const double len2 = ax * ax + ay * ay;
  if (len2 <= 0) return false;                       // a 和 b 重合, 这条线无效
  const double t = ((p.x - a.x) * ax + (p.y - a.y) * ay) / len2;
  const double slack = pad / std::sqrt(len2);        // 把像素换算成比例
  // 两端各放宽 pad 像素(换算成比例就是 slack), 所以范围是 -slack ~ 1+slack
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

  // 一帧处理完, 告诉计数器"这一帧还活着的是哪些 id".
  // 不在这份名单里的 id, 它的 prev_ 记录会被清掉.
  // 不清会怎样: 车开出画面后条目一直留着, 接摄像头连跑几天就是几十万条,
  // 而且同一个 id 号永远不会复用, 留着也没有任何用处.
  // 由调用方传"活着的 id"而不是帧号, 是因为 Counter 不该自己去猜跟踪器
  // 的缓冲期有多长; 判断谁死了是跟踪器的事.
  void retain(const std::vector<long long>& live_ids);

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
