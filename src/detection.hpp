// 检测框结构体 Detection 的定义，两个推理后端和 tracker 共用。
// 单独成头文件是为了不依赖任何后端，包含它不会连带拉进 ONNX Runtime / RKNN 头文件。

#pragma once

// 一个检测框. 坐标是【原图】坐标, 不是 640 输入图上的坐标.
// 也就是说推理模块返回之前, 已经把 letterbox 的缩放和补边还原回去了.
struct Detection {
  double x1 = 0, y1 = 0, x2 = 0, y2 = 0;   // 左上角 + 右下角
  double score = 0;                        // 置信度 0~1
  int cls = 0;                             // 类别号. 自训模型只有 0 = vehicle
};

