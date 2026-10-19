// RKNN 检测器（板端）。接口与 detector.hpp 的 Detector 一致，由 CMake 的 USE_RKNN 决定包含哪个。

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"   // Detection 定义在这里, 和 ONNX 版共用

// 静默开关: 关掉后 detector 不再往终端打印(多路并行时用)
void rknn_set_verbose(bool on);

class RknnDetector {
 public:
  // model_path 指向 .rknn 文件（不是 .onnx）
  RknnDetector(const std::string& model_path, int imgsz, double conf, double iou);
  ~RknnDetector();

  RknnDetector(const RknnDetector&) = delete;
  RknnDetector& operator=(const RknnDetector&) = delete;

  std::vector<Detection> detect(const cv::Mat& frame);

  // 指定使用哪个 NPU 核：0 / 1 / 2，传 -1 表示三核共用（构造时的默认行为）。
  // 多线程并行时应给每个实例分配不同的核，否则所有线程会挤在同一个核上。
  // 须在构造之后、首次 detect 之前调用。
  void set_core(int core_id);

 private:
  struct Impl;                 // 定义在 .cpp 里, 头文件不暴露 rknn_api.h
  std::unique_ptr<Impl> impl_;
  int imgsz_;
  double conf_;
  double iou_;
};

