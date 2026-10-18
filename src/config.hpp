// 配置的数据结构. 字段跟 configs/testN.json 一一对应.
//
// 读 json 的代码在 config.cpp, 这个文件只负责"长什么样".
// 加一个新配置项的顺序: 先在这里加字段 → 再去 config.cpp 加解析 → 最后在 json 里写.
#pragma once

#include <array>
#include <string>
#include <vector>

// 一个二维点. 画面坐标, 原点在左上角, 单位是像素.
struct Point {
  double x = 0;
  double y = 0;
};

// 一条计数线
struct Line {
  std::string name;         // "A" / "B" / "C", 画面上显示的名字
  Point a;                  // 线的起点
  Point b;                  // 线的终点
                            // 注意: a 和 b 同时决定了"位置、方向、长度",
                            // 线画到哪就只管那一段, 所以没有 x_min/x_max 了
  std::string count_dir;    // ""     = 两个方向都数(默认)
                            // "pos"  = 只数 负侧→正侧
                            // "neg"  = 只数 正侧→负侧
  std::string group;        // 同方向合并统计用的名字, 不写就自动等于 name
  double margin = 6;        // 死区(像素): 离线不到这么近不判过线, 防止车压线时抖动反复计数
  bool has_label_at = false;
  Point label_at;           // 计数数字画在哪; 不写就自动放在线的上端
};

// "detect" 小节: 送给模型的参数
struct Detect {
  int imgsz = 640;          // 送进模型的图缩到多大. 必须和训练时一致
  double conf = 0.3;        // 置信度阈值: 低于它的框直接丢. 调高更严, 但漏检变多
  double iou = 0.5;         // NMS 阈值: 两个框重叠超过它就只留分数高的那个
};

// "draw" 小节: 画面上画什么、怎么画
// 每个 bool 都是开关, 正式跑测试时关掉, 画面就干净了
struct Draw {
  bool roi = true;          // 是否画出 ROI 多边形轮廓
  bool box = true;          // 是否画车辆检测框
  int box_width = 2;        // 框线粗细
  std::array<int, 3> box_color{0, 255, 0};   // 框的颜色, 注意是 [B, G, R] 不是 RGB
  bool id = true;           // 是否在框上写跟踪编号 "#12"
  bool label = true;        // 是否写置信度
  double font_scale = 0.6;  // 字号缩放
  int text_width = 2;       // 文字粗细
  double counter_scale = 0.9;                 // 计数数字的字号
  std::array<int, 3> text_color{0, 255, 255}; // 计数数字颜色 [B,G,R] = 黄色
  std::array<int, 3> line_color{0, 0, 255};   // 计数线颜色 [B,G,R] = 红色
  int line_width = 2;
  std::array<int, 3> roi_color{0, 200, 255};  // ROI 轮廓颜色
};

// 一整份配置, 对应一个 configs/xxx.json
struct Config {
  std::string name;                       // 视频名, 也是输出文件的前缀
  std::string video;                      // 视频路径, 相对工程根目录
  Detect detect;
  bool has_roi = false;                   // roi 是不是有效(至少 3 个顶点)
  std::vector<Point> roi;                 // 检测区域多边形; 空 = 整帧都算
  std::vector<Line> lines;                // 可以有任意多条计数线
  Draw draw;
  std::string output = "save";            // "save" 存成 mp4 / "show" 开窗口实时播放
  int max_frames = 0;                     // 只跑前多少帧, 0 = 全部
};

// 读一个 json 文件, 返回填好的 Config. 文件坏了会抛异常.
Config load_config(const std::string& path);
