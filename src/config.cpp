#include "config.hpp"

#include <stdexcept>

#include "mini_json.hpp"

namespace {

// json 里的一个数组 [x, y] 转成 Point
Point to_point(const mini::Value& v) {
  Point p;
  if (v.is_arr() && v.arr.size() >= 2) {
    p.x = v.arr[0].num;
    p.y = v.arr[1].num;
  }
  return p;
}

// json 里的 [b, g, r] 转成颜色; 格式不对就用默认值, 不让程序崩
std::array<int, 3> to_color(const mini::Value& v, std::array<int, 3> def) {
  if (!v.is_arr() || v.arr.size() < 3) return def;
  return {static_cast<int>(v.arr[0].num), static_cast<int>(v.arr[1].num),
          static_cast<int>(v.arr[2].num)};
}

}  // namespace

// 解析整份配置.
// 规矩: 字段没写就用结构体里的默认值, 所以 json 可以只写关心的那几项.
Config load_config(const std::string& path) {
  mini::Value root = mini::parse_file(path);
  if (!root.is_obj()) throw std::runtime_error("配置文件最外层应该是个对象: " + path);

  Config c;
  c.name = root.get_str("name", "");
  c.video = root.get_str("video", "");

  // ---- detect: 模型参数 ----
  if (root.has("detect")) {
    const mini::Value& d = root.obj.at("detect");
    c.detect.imgsz = d.get_int("imgsz", c.detect.imgsz);
    c.detect.conf = d.get_num("conf", c.detect.conf);
    c.detect.iou = d.get_num("iou", c.detect.iou);
  }

  // ---- roi: 检测区域多边形, [[x,y], [x,y], ...] ----
  // 至少 3 个点才算有效; 少于 3 个点画不出多边形, 当整帧处理.
  if (root.has("roi") && root.obj.at("roi").is_arr()) {
    for (const auto& p : root.obj.at("roi").arr) c.roi.push_back(to_point(p));
    c.has_roi = c.roi.size() >= 3;
  }

  // ---- lines: 计数线列表, 有几条就解析几条 ----
  if (root.has("lines") && root.obj.at("lines").is_arr()) {
    for (const auto& item : root.obj.at("lines").arr) {
      Line L;
      L.name = item.get_str("name", "");
      L.a = to_point(item.obj.at("a"));   // 起点, 必填
      L.b = to_point(item.obj.at("b"));   // 终点, 必填
      L.count_dir = item.get_str("count_dir", "");
      L.group = item.get_str("group", "");
      if (L.group.empty()) L.group = L.name;   // 没写 group 就用自己的名字
      L.margin = item.get_num("margin", 6);    // 死区, 默认 6 像素
      if (item.has("label_at")) {              // 数字画在哪, 可选
        L.label_at = to_point(item.obj.at("label_at"));
        L.has_label_at = true;
      }
      c.lines.push_back(L);
    }
  }

  // ---- draw: 画法 ----
  if (root.has("draw")) {
    const mini::Value& d = root.obj.at("draw");
    Draw& w = c.draw;   // 取引用, 后面改 w 就是改 c.draw
    w.roi = d.get_bool("roi", w.roi);
    w.box = d.get_bool("box", w.box);
    w.box_width = d.get_int("box_width", w.box_width);
    if (d.has("box_color")) w.box_color = to_color(d.obj.at("box_color"), w.box_color);
    w.id = d.get_bool("id", w.id);
    w.label = d.get_bool("label", w.label);
    w.font_scale = d.get_num("font_scale", w.font_scale);
    w.text_width = d.get_int("text_width", w.text_width);
    w.counter_scale = d.get_num("counter_scale", w.counter_scale);
    if (d.has("text_color")) w.text_color = to_color(d.obj.at("text_color"), w.text_color);
    if (d.has("line_color")) w.line_color = to_color(d.obj.at("line_color"), w.line_color);
    w.line_width = d.get_int("line_width", w.line_width);
    if (d.has("roi_color")) w.roi_color = to_color(d.obj.at("roi_color"), w.roi_color);
  }

  // ---- 顶层杂项 ----
  c.output = root.get_str("output", c.output);        // "save" / "show"
  c.max_frames = root.get_int("max_frames", c.max_frames);  // 0 = 全部
  return c;
}
