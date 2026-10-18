// 配置的数据结构，字段与 configs/testN.json 一一对应，解析在 config.cpp。

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
                            // a 与 b 同时决定位置、方向与长度;
                            // 计数线的作用范围即线段本身, 因此不需要 x_min/x_max
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
  double conf = 0.4;        // 置信度阈值: 低于它的框直接丢. 调高更严, 但漏检变多
  double iou = 0.3;         // NMS 阈值: 两个框重叠超过它就只留分数高的那个
};

// "track" 小节: 跟踪器的四个阈值.
//
// 为什么这四个要单独拿出来: 它们决定"哪些框可信、哪些框只能救轨迹",
// 和 detect.conf 一起构成两道串联的筛子. 写死在代码里就没法按视频调.
//
// 和 detect.conf 的关系(重要, 配错了第二阶段会全程空转):
//   conf  过滤 -> 框进不了跟踪器
//   low   过滤 -> 框进了跟踪器但被丢掉
//   所以 low 配得比 conf 低是没有意义的, 那些框根本到不了这里.
struct TrackCfg {
  int buffer = 30;           // 轨迹丢了之后还能活几帧(30 帧约 1.2 秒 @24fps)
  double high = 0.45;        // 高分框下限: >= 它的进高分堆, 能建新轨迹
  double low = 0.30;         // 低分框下限: 低于它直接丢; 中间的只能救已有轨迹
  double new_thresh = 0.45;  // 建新轨迹的最低分数. 只在高分堆里判, 通常 == high
  double match = 0.8;        // 第一轮的代价上限(代价 = 1 - IoU). 0.8 相当于 IoU >= 0.2
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
  TrackCfg track;
  bool has_roi = false;                   // roi 是否有效(至少 3 个顶点)
  std::vector<Point> roi;                 // 检测区域多边形; 空 = 整帧都算
  std::vector<Line> lines;                // 可以有任意多条计数线
  Draw draw;
  std::string output = "save";            // "save" 存成 mp4 / "show" 开窗口实时播放
  int max_frames = 0;                     // 只跑前多少帧, 0 = 全部
};

// 读一个 json 文件, 返回填好的 Config. 文件坏了会抛异常.
Config load_config(const std::string& path);

