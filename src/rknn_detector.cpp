// RKNN 推理实现（板子用）. 逻辑和 detector.cpp 一一对应, 只把四个调用换掉:
//
//   Ort::Session   加载模型   ->  rknn_init
//   CreateTensor   喂输入     ->  rknn_inputs_set
//   session_.Run   跑          ->  rknn_run
//   GetTensorData  取输出     ->  rknn_outputs_get
//
// 前处理和后处理与 ONNX 版完全相同, 因为它们跟推理后端无关.
//
// 一个必须知道的差异: 数据布局
//   ONNX Runtime 吃 NCHW（先通道、再高、再宽）
//   RKNN 内部也是 NCHW（rknn_api.h 里写明 "internal input format of NPU is NCHW by default"）
//   所以这里也传 NCHW, 免掉驱动再做一次转换.

#include "rknn_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "rknn_api.h"

namespace {

double iou(const Detection& a, const Detection& b) {
  const double iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const double ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0.0;
  const double inter = iw * ih;
  const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

void check(int ret, const char* what) {
  if (ret != 0) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s 失败, 错误码 %d", what, ret);
    throw std::runtime_error(buf);
  }
}

}  // namespace

struct RknnDetector::Impl {
  rknn_context ctx = 0;
  std::vector<unsigned char> model;   // 模型内容要一直留着, rknn_init 之后不能释放
  int n_rows = 5;                     // 每个候选框 5 个数: cx cy w h score
  int n_cols = 8400;                  // 候选框个数
};

RknnDetector::RknnDetector(const std::string& model_path, int imgsz, double conf, double iou)
    : impl_(new Impl()), imgsz_(imgsz), conf_(conf), iou_(iou) {
  // 1. 把 .rknn 整个读进内存
  std::ifstream f(model_path, std::ios::binary);
  if (!f) throw std::runtime_error("打不开模型: " + model_path);
  impl_->model.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  if (impl_->model.empty()) throw std::runtime_error("模型文件是空的: " + model_path);

  // 2. 初始化. 第三个参数是模型大小, 第四个 0 = 不加密
  check(rknn_init(&impl_->ctx, impl_->model.data(),
                  static_cast<uint32_t>(impl_->model.size()), 0, nullptr),
        "rknn_init");

  // 3. 问一下模型有几个输入输出
  rknn_input_output_num io_num;
  std::memset(&io_num, 0, sizeof(io_num));
  check(rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
        "rknn_query(IN_OUT_NUM)");
  std::printf("RKNN 模型: 输入 %u 个, 输出 %u 个\n", io_num.n_input, io_num.n_output);

  // 4. 打印输入的形状, 出问题时第一个要看的就是这个
  rknn_tensor_attr in_attr;
  std::memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  check(rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
        "rknn_query(INPUT_ATTR)");
  std::printf("输入: dims=[");
  for (uint32_t i = 0; i < in_attr.n_dims; ++i) {
    std::printf("%d%s", in_attr.dims[i], i + 1 == in_attr.n_dims ? "" : ", ");
  }
  std::printf("] fmt=%d type=%d\n", static_cast<int>(in_attr.fmt),
              static_cast<int>(in_attr.type));

  // 5. 问输出的形状, 用它来定 n_rows / n_cols, 不写死
  rknn_tensor_attr out_attr;
  std::memset(&out_attr, 0, sizeof(out_attr));
  out_attr.index = 0;
  if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)) == 0) {
    if (out_attr.n_dims >= 3) {
      impl_->n_rows = static_cast<int>(out_attr.dims[out_attr.n_dims - 2]);
      impl_->n_cols = static_cast<int>(out_attr.dims[out_attr.n_dims - 1]);
    }
    std::printf("输出: dims=[");
    for (uint32_t i = 0; i < out_attr.n_dims; ++i) {
      std::printf("%d%s", out_attr.dims[i], i + 1 == out_attr.n_dims ? "" : ", ");
    }
    std::printf("]\n");
  }
  std::printf("按 [%d, %d] 解析输出 (行=坐标和分数, 列=候选框)\n", impl_->n_rows, impl_->n_cols);
}

RknnDetector::~RknnDetector() {
  if (impl_ && impl_->ctx) rknn_destroy(impl_->ctx);
}

std::vector<Detection> RknnDetector::detect(const cv::Mat& frame) {
  // ===== 第 1 步: letterbox, 和 ONNX 版一字不差 =====
  const double scale = std::min(imgsz_ / static_cast<double>(frame.cols),
                                imgsz_ / static_cast<double>(frame.rows));
  const int nw = static_cast<int>(std::round(frame.cols * scale));
  const int nh = static_cast<int>(std::round(frame.rows * scale));
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(nw, nh));

  const int dx = (imgsz_ - nw) / 2;
  const int dy = (imgsz_ - nh) / 2;
  cv::Mat canvas(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  // blobFromImage 一次做完: 除以 255、BGR->RGB（倒数第二个参数 true）、排成 NCHW
  cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(imgsz_, imgsz_),
                                        cv::Scalar(), true, false);

  // ===== 第 2 步: 喂进去、跑 =====
  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = blob.ptr<float>();
  inputs[0].size = static_cast<uint32_t>(blob.total() * sizeof(float));
  inputs[0].pass_through = 0;                  // 0 = 让驱动按下面两项做格式转换
  inputs[0].type = RKNN_TENSOR_FLOAT32;        // 模型要 float32, 0~1
  inputs[0].fmt = RKNN_TENSOR_NCHW;            // 和 NPU 内部布局一致, 免一次转换
  check(rknn_inputs_set(impl_->ctx, 1, inputs), "rknn_inputs_set");
  check(rknn_run(impl_->ctx, nullptr), "rknn_run");

  // ===== 第 3 步: 取输出 =====
  rknn_output outputs[1];
  std::memset(outputs, 0, sizeof(outputs));
  outputs[0].want_float = 1;    // 要 float 结果, 省得自己反量化
  outputs[0].is_prealloc = 0;   // 让库自己分配, 用完调 release
  outputs[0].index = 0;
  check(rknn_outputs_get(impl_->ctx, 1, outputs, nullptr), "rknn_outputs_get");

  const float* out = static_cast<const float*>(outputs[0].buf);
  const int n_cols = impl_->n_cols;
  const int n_rows = impl_->n_rows;

  // ===== 第 4 步: 分数过滤. 布局和 ONNX 一样: out[c * n_cols + i] =====
  std::vector<Detection> boxes;
  for (int i = 0; i < n_cols; ++i) {
    double best = 0;
    int best_c = 0;
    for (int c = 4; c < n_rows; ++c) {
      const double s = out[static_cast<size_t>(c) * n_cols + i];
      if (s > best) {
        best = s;
        best_c = c - 4;
      }
    }
    if (best < conf_) continue;

    const double cx = out[static_cast<size_t>(0) * n_cols + i];
    const double cy = out[static_cast<size_t>(1) * n_cols + i];
    const double w = out[static_cast<size_t>(2) * n_cols + i];
    const double h = out[static_cast<size_t>(3) * n_cols + i];

    Detection d;
    d.x1 = (cx - w / 2 - dx) / scale;
    d.y1 = (cy - h / 2 - dy) / scale;
    d.x2 = (cx + w / 2 - dx) / scale;
    d.y2 = (cy + h / 2 - dy) / scale;
    d.score = best;
    d.cls = best_c;
    boxes.push_back(d);
  }

  rknn_outputs_release(impl_->ctx, 1, outputs);

  // ===== 第 5 步: NMS, 和 ONNX 版一字不差 =====
  std::sort(boxes.begin(), boxes.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  for (const auto& b : boxes) {
    bool drop = false;
    for (const auto& k : keep) {
      if (k.cls == b.cls && iou(b, k) > iou_) {
        drop = true;
        break;
      }
    }
    if (!drop) keep.push_back(b);
  }
  return keep;
}
