// 检测框的定义. 单独一个头文件, 因为两个推理后端(torch 无关)都要用它.
//
// 为什么不让 detector.hpp 自己定义:
//   ONNX 版和 RKNN 版是两个互斥的头文件, main.cpp 只会包含其中一个.
//   但 tracker.hpp 也要用 Detection —— 如果它去包含 detector.hpp,
//   板子上(用 RKNN)就会连带把 ONNX Runtime 的头文件拉进来, 直接编不过.
//   所以把最基础的结构体挪到这里, 谁需要谁包含, 不带任何后端依赖.
#pragma once

// 一个检测框. 坐标是【原图】坐标, 不是 640 输入图上的坐标.
// 也就是说推理模块返回之前, 已经把 letterbox 的缩放和补边还原回去了.
struct Detection {
  double x1 = 0, y1 = 0, x2 = 0, y2 = 0;   // 左上角 + 右下角
  double score = 0;                        // 置信度 0~1
  int cls = 0;                             // 类别号. 自训模型只有 0 = vehicle
};
