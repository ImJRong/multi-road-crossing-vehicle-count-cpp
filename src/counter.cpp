#include "counter.hpp"

#include <set>

Counter::Counter(const std::vector<Line>& lines)
    : lines_(lines), counters_(lines.size(), 0) {}
// 构造时把配置里的线存进来, 每条线配一个从 0 开始的计数.

// 一辆车每帧调一次. 判断"它这一帧是不是穿过了某条线", 穿过了就 +1.
// 整段就做三件事: 算点在线哪一侧 -> 和上一帧比符号 -> 变了就计数.
void Counter::update(long long track_id, const Point& p, const std::string& cls,
                     int frame_no) {
  // 一辆车可能同时穿过好几条线(比如同时又过 A 线又过 D 线), 所以逐条线都要判.
  for (size_t li = 0; li < lines_.size(); ++li) {
    const Line& L = lines_[li];

    // 1. 这个点是不是落在这条线自己的那一段上.
    //    线画到哪就管到哪: 点落在线的延长线上不算, 从端点外面绕过去也不算(会漏计).
    if (!near_segment(p, L.a, L.b, L.margin)) continue;

    // 2. 算它在线的哪一侧. 结果正负代表两侧, 数值大小 = 像素距离 × 线长.
    //    例: 线是横的, 点在线上方 -> 负数; 下方 -> 正数; 正好在线上 -> 0.
    const double s = side(p, L.a, L.b);

    // 3. 死区: 车压在线上时框中心会来回抖, 离线的距离小于 margin 就当作"还在线上",
    //    直接跳过这一帧: 既不算过线, 也不更新记录. 不然车停在线上能把数字刷爆.
    const double len = line_length(L.a, L.b);
    if (len <= 0) continue;                      // a 和 b 重合, 这条线是坏的
    if (std::fabs(s) / len <= L.margin) continue;  // s 是"像素×线长", 除以线长才是像素

    // 4. 拿 "第几条线 + 哪个车号" 去表里查: 这辆车上一帧在这条线的哪一侧?
    //    key 必须带车号, 因为一条线上同时跑着好几辆车, 各记各的.
    const std::pair<size_t, long long> key(li, track_id);
    auto it = prev_.find(key);

    // 5. 符号变了 = 从一侧跑到另一侧 = 过线.
    //    it != prev_.end(): 表里查得到, 说明这辆车以前见过(不是刚出现的新车).
    //    后半句: 比较的是 "s < 0" 这个布尔值, 不是比数值大小,
    //    所以 -12200 -> -8000 不算过线(还在同一侧), 而 -12200 -> +500 才算.
    if (it != prev_.end() && ((it->second < 0) != (s < 0))) {
      // 能走到这里, 说明符号已经翻过来了; 那么"现在在哪一侧"就唯一决定了穿线方向:
      const bool passed = s > 0;   // true = 从负侧跑到正侧, false = 从正侧跑到负侧
      // count_dir 决定数不数: 不写就两个方向都数, 配了 "pos"/"neg" 只数一边.
      //   pos 代表从负侧跑到正侧, neg 代表从正侧跑到负侧;
      //   count_dir 与穿线方向相同就不跳过, 不同就跳过.
      const bool skip = (L.count_dir == "pos" && !passed) ||
                        (L.count_dir == "neg" && passed);
      if (!skip) {
        // 数到了: 记一笔流水(以后导 csv), 再把画面上的数字加一.
        Hit h;
        h.line = L.name;
        h.group = L.group;
        h.cls = cls;
        h.frame = frame_no;
        h.x = static_cast<int>(p.x);
        h.y = static_cast<int>(p.y);
        hits_.push_back(h);
        counters_[li] += 1;
      }
    }
    // 6. 不管这次数没数, 都要记住它当前在哪一侧, 作为下一帧的比较对象.
    //    新车第一次也走这里: 先记下来, 下次才能比出"换边了".
    prev_[key] = s;
  }
}

// 第 i 条线数到多少了
int Counter::count(size_t line_index) const {
  if (line_index >= counters_.size()) return 0;
  return counters_[line_index];
}

// 每帧调一次: 告诉计数器"现在还活着的车是哪些号".
// 跟踪器删掉某个车号后(丢超过 30 帧), 那张 prev_ 表里它的记录就没用了,
// 这里顺手清掉. 不清的话, 接摄像头连跑几天会攒到几十万条.
void Counter::retain(const std::vector<long long>& live_ids) {
  if (prev_.empty()) return;      // 一条记录都没有, 什么都不用做

  // live_ids 一般只有十几二十个, 做成集合查起来比挨个比对省事
  const std::set<long long> live(live_ids.begin(), live_ids.end());

  // 边遍历边删: erase(it) 会返回下一个位置, 所以这样写是安全的.
  // 不能写成 for (...) { if (死) prev_.erase(it); } —— 删完 it 就失效了, 再 ++ 会崩.
  for (auto it = prev_.begin(); it != prev_.end();) {
    if (live.count(it->first.second) == 0) {   // first.second = 那个记录里的车号
      it = prev_.erase(it);      // 这辆车已经不在跟踪器里了
    } else {
      ++it;
    }
  }
}

// 全部归零(重新开始跑时用)
void Counter::clear() {
  prev_.clear();
  hits_.clear();
  counters_.assign(lines_.size(), 0);
}
