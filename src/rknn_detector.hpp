// RKNN 版的检测器（板子上用）. 接口和 detector.hpp 里的 Detector 完全一致,
// 区别只在底层：那边是 ONNX Runtime, 这边是瑞芯微的 librknnrt.
//
// 为什么不直接改 detector.hpp：
//   那个文件里写着 Ort::Env / Ort::Session, 板子上没有 ONNX Runtime, 包含它就编不过.
//   所以后端相关的头文件分开, 由 CMake 的 USE_RKNN 开关决定包含哪一个.
//
// 前处理(letterbox)和后处理(分数过滤/NMS/坐标还原)和 ONNX 版一字不差 —— 这部分与后端无关.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"   // Detection 定义在这里, 和 ONNX 版共用

class RknnDetector {
 public:
  // model_path 指向 .rknn 文件（不是 .onnx）
  RknnDetector(const std::string& model_path, int imgsz, double conf, double iou);
  ~RknnDetector();

  RknnDetector(const RknnDetector&) = delete;
  RknnDetector& operator=(const RknnDetector&) = delete;

  std::vector<Detection> detect(const cv::Mat& frame);

 private:
  struct Impl;                 // 定义在 .cpp 里, 头文件不暴露 rknn_api.h
  std::unique_ptr<Impl> impl_;
  int imgsz_;
  double conf_;
  double iou_;
};
