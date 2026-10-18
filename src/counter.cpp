#include "counter.hpp"

Counter::Counter(const std::vector<Line>& lines)
    : lines_(lines), counters_(lines.size(), 0) {}
// 构造时把配置里的线存进来, 每条线配一个从 0 开始的计数.

void Counter::update(long long track_id, const Point& p, const std::string& cls,
                     int frame_no) {
  for (size_t li = 0; li < lines_.size(); ++li) {   // 一辆车可能同时穿过好几条线
    const Line& L = lines_[li];

    // 1. 先判这个点是不是落在这条线自己的那一段上.
    //    线画到哪就管到哪, 线外的过线不算, 免得一条长线把别的车道也算进来.
    if (!near_segment(p, L.a, L.b, L.margin)) continue;

    // 2. 算它在线的哪一侧. 正负代表两侧, 数值大小 = 像素距离 × 线长.
    const double s = side(p, L.a, L.b);

    // 3. 死区: 车压在线上时框中心会来回抖, 这一小段距离内直接不处理,
    //    既不算过线, 也不更新记录, 等它明确走到一侧再说.
    const double len = line_length(L.a, L.b);
    if (len <= 0) continue;
    if (std::fabs(s) / len <= L.margin) continue;

    // 4. 取出这个 id 在这条线上一次记录的状态.
    //    key 用 (线序号, 车号) 而不是只用线序号, 因为一条线上同时有好多车.
    const std::pair<size_t, long long> key(li, track_id);
    auto it = prev_.find(key);

    // 5. 符号变了 = 从一侧跑到另一侧 = 过线.
    //    比较的是 "s < 0" 这个布尔值, 不是比数值大小;
    //    所以不管距离多远, 只要换了边就算.
    if (it != prev_.end() && ((it->second < 0) != (s < 0))) {
      const bool passed = s > 0;     // true = 往正侧穿
      // count_dir 决定数不数: 不写就两个方向都数, 配了 "pos"/"neg" 只数一边.
      const bool skip = (L.count_dir == "pos" && !passed) ||
                        (L.count_dir == "neg" && passed);
      if (!skip) {
        Hit h;
        h.line = L.name;
        h.group = L.group;
        h.cls = cls;
        h.frame = frame_no;
        h.x = static_cast<int>(p.x);
        h.y = static_cast<int>(p.y);
        hits_.push_back(h);          // 记一笔流水, 以后能导 csv
        counters_[li] += 1;          // 画面上的数字加一
      }
    }
    // 6. 不管这次数没数, 都要记住当前在哪一侧, 作为下一帧的比较对象.
    prev_[key] = s;
  }
}

int Counter::count(size_t line_index) const {
  if (line_index >= counters_.size()) return 0;
  return counters_[line_index];
}

void Counter::clear() {
  prev_.clear();
  hits_.clear();
  counters_.assign(lines_.size(), 0);
}
