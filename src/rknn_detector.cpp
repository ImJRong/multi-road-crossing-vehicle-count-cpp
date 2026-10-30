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
//
// 两种模型, 一份代码:
//   单输出 [1, 5, 8400]              —— 坐标和分数拼在一起, FP16 模型就是这个
//   双输出 [1,4,8400] + [1,1,8400]   —— 拆开的 boxes / scores, INT8 模型必须是这个
//
// 为什么 INT8 要拆:
//   量化把一个张量整体映射到 8 位整数, 一套 scale 管所有数. 坐标是 0~640, 分数是 0~1,
//   共用 2.5 的步长时, 0.87 会被舍成 0 —— 分数全灭, 一个框都出不来.
//   拆开后 boxes 用 2.5 的步长, scores 用 1/255 的步长, 各管各的, 分数就保住了.
//
// 这里的做法: 两个输出各自读, 但后面的过滤和 NMS 完全走同一条路,
// 所以单输出模型和双输出模型的结果是一致可比的.

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

// 打印一个 tensor 的 dims, 排查问题时最常看的就是这行
void print_dims(const char* tag, const rknn_tensor_attr& a) {
  std::printf("%s dims=[", tag);
  for (uint32_t i = 0; i < a.n_dims; ++i) {
    std::printf("%d%s", a.dims[i], i + 1 == a.n_dims ? "" : ", ");
  }
  std::printf("]\n");
}

}  // namespace

struct RknnDetector::Impl {
  rknn_context ctx = 0;
  std::vector<unsigned char> model;   // 模型内容要一直留着, rknn_init 之后不能释放

  int n_output = 1;                   // 模型有几个输出
  int boxes_index = 0;                // 坐标在哪个输出里
  int scores_index = -1;              // 分数在哪个输出里(-1 = 单输出, 分数跟在坐标后面)
  int n_rows = 5;                     // 每个候选框一共 4+类别数 个数
  int n_cols = 8400;                  // 候选框个数
  int n_cls = 1;                      // 类别数
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
  print_dims("输入:", in_attr);
  std::printf("      fmt=%d type=%d\n", static_cast<int>(in_attr.fmt),
              static_cast<int>(in_attr.type));

  // 5. 问输出的形状, 用它来定 n_rows / n_cols, 不写死
  impl_->n_output = static_cast<int>(io_num.n_output);
  int cols = 0;

  if (impl_->n_output == 1) {
    // ---- 单输出: [1, 5, 8400], 前 4 行坐标, 第 5 行起是分数 ----
    rknn_tensor_attr a;
    std::memset(&a, 0, sizeof(a));
    a.index = 0;
    if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) == 0) {
      print_dims("输出[0]:", a);
      if (a.n_dims >= 3) {
        impl_->n_rows = static_cast<int>(a.dims[a.n_dims - 2]);
        cols = static_cast<int>(a.dims[a.n_dims - 1]);
      }
    }
    impl_->n_cls = impl_->n_rows - 4;
    impl_->scores_index = -1;
  } else {
    // ---- 双输出: 一个 [1, 4, 8400] 放坐标, 一个 [1, C, 8400] 放分数 ----
    // 不靠输出顺序猜, 按"行数"认: 4 行的那个是坐标, 其余的是分数.
    int rows_boxes = 0;
    int rows_scores = 0;
    for (int i = 0; i < impl_->n_output; ++i) {
      rknn_tensor_attr a;
      std::memset(&a, 0, sizeof(a));
      a.index = i;
      if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) != 0) continue;
      char tag[32];
      std::snprintf(tag, sizeof(tag), "输出[%d]:", i);
      print_dims(tag, a);
      if (a.n_dims < 3) continue;
      const int r = static_cast<int>(a.dims[a.n_dims - 2]);
      const int c = static_cast<int>(a.dims[a.n_dims - 1]);
      if (c > cols) cols = c;
      if (r == 4) {
        impl_->boxes_index = i;
        rows_boxes = r;
      } else if (r > 0) {
        impl_->scores_index = i;
        rows_scores = r;
      }
    }
    if (impl_->scores_index < 0 || rows_boxes == 0) {
      throw std::runtime_error(
          "这个模型有多个输出, 但认不出哪路是坐标(4 行)、哪路是分数. "
          "预期是 [1,4,8400] + [1,C,8400], 实际形状看上面的打印.");
    }
    impl_->n_rows = rows_boxes + rows_scores;
    impl_->n_cls = rows_scores;
  }

  impl_->n_cols = cols > 0 ? cols : 8400;
  std::printf("按 %s 解析: 类别 %d 个, 候选框 %d 个\n",
              impl_->n_output == 1 ? "单输出 [4+C, N]" : "双输出 boxes/scores", impl_->n_cls,
              impl_->n_cols);
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

  // ===== 第 3 步: 取输出(单输出取 1 个, 双输出取 2 个) =====
  const int n_out = impl_->n_output;
  std::vector<rknn_output> outputs(static_cast<size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    std::memset(&outputs[static_cast<size_t>(i)], 0, sizeof(rknn_output));
    outputs[static_cast<size_t>(i)].index = i;
    outputs[static_cast<size_t>(i)].want_float = 1;    // 要 float, 省得自己反量化
    outputs[static_cast<size_t>(i)].is_prealloc = 0;   // 让库自己分配, 用完调 release
  }
  check(rknn_outputs_get(impl_->ctx, n_out, outputs.data(), nullptr), "rknn_outputs_get");

  // 两种模型在这里归一成同一套读法: coord 前 4 行是 cx cy w h, score 是各类分数
  const float* coord = nullptr;
  const float* score = nullptr;
  if (n_out == 1) {
    // 单输出: 坐标和分数在同一个 buffer 里, 分数从第 4 行开始
    coord = static_cast<const float*>(outputs[0].buf);
    score = coord + static_cast<size_t>(4) * impl_->n_cols;
  } else {
    coord = static_cast<const float*>(outputs[static_cast<size_t>(impl_->boxes_index)].buf);
    score = static_cast<const float*>(outputs[static_cast<size_t>(impl_->scores_index)].buf);
  }

  const int n_cols = impl_->n_cols;
  const int n_cls = impl_->n_cls;

  // ===== 第 4 步: 分数过滤. 每个候选框取分数最高的类别 =====
  std::vector<Detection> boxes;
  boxes.reserve(64);
  for (int i = 0; i < n_cols; ++i) {
    double best = 0;
    int best_c = 0;
    for (int c = 0; c < n_cls; ++c) {
      const double s = score[static_cast<size_t>(c) * n_cols + i];
      if (s > best) {
        best = s;
        best_c = c;
      }
    }
    if (best < conf_) continue;

    const double cx = coord[static_cast<size_t>(0) * n_cols + i];
    const double cy = coord[static_cast<size_t>(1) * n_cols + i];
    const double w = coord[static_cast<size_t>(2) * n_cols + i];
    const double h = coord[static_cast<size_t>(3) * n_cols + i];

    Detection d;
    d.x1 = (cx - w / 2 - dx) / scale;
    d.y1 = (cy - h / 2 - dy) / scale;
    d.x2 = (cx + w / 2 - dx) / scale;
    d.y2 = (cy + h / 2 - dy) / scale;
    d.score = best;
    d.cls = best_c;
    boxes.push_back(d);
  }

  rknn_outputs_release(impl_->ctx, n_out, outputs.data());

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
