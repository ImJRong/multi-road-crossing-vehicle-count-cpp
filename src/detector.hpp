// YOLOv8 ONNX 推理封装. 预处理、后处理全在这里, main.cpp 只管调用.
//
// 为什么单独拆一个类:
//   上 RK3588 板子时要把 ONNX Runtime 换成 RKNN 的运行时(librknnrt).
//   到时候只改这个文件的实现, main.cpp 一行都不用动.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

#include "detection.hpp"   // Detection 定义在这里, 和 RKNN 版共用

class Detector {
 public:
  // imgsz: 送进模型的边长(必须是 32 的倍数, YOLOv8 的要求)
  // conf : 低于这个分数直接丢
  // iou  : NMS 时两个框重叠超过它就只留一个
  Detector(const std::string& model_path, int imgsz, double conf, double iou);

  // 送一帧进去(任意尺寸), 返回过滤 + NMS 之后的框, 坐标是原图坐标
  std::vector<Detection> detect(const cv::Mat& frame);

 private:
  int imgsz_;
  double conf_;
  double iou_;

  // ONNX Runtime 的三件套, 创建后一直复用.
  // 如果每帧都重新建 Session, 光初始化就要几百毫秒, 比推理本身还慢.
  Ort::Env env_;                          // 运行时环境
  Ort::Session session_{nullptr};         // 加载好的模型
  Ort::MemoryInfo mem_{nullptr};          // 描述内存布局(CPU 连续内存)
};
