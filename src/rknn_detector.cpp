// RKNN 推理实现（板端）。流程与 detector.cpp 一致，仅替换以下四个调用：
//
//   Ort::Session   加载模型   ->  rknn_init
//   CreateTensor   提交输入   ->  rknn_inputs_set
//   session_.Run   执行推理   ->  rknn_run
//   GetTensorData  读取输出   ->  rknn_outputs_get
//
// 前处理与后处理与 ONNX 版相同，这两部分与推理后端无关。
//
// 数据布局差异：
//   ONNX Runtime 使用 NCHW（通道、高、宽）
//   RKNN 内部同样为 NCHW（见 rknn_api.h："internal input format of NPU is NCHW by default"）
//   因此此处直接传入 NCHW，避免驱动侧额外转换。
//
// 支持两种输出结构：
//   单输出 [1, 5, 8400]              —— 坐标与分数拼接，FP16 模型采用此结构
//   双输出 [1,4,8400] + [1,1,8400]   —— 分离的 boxes / scores，INT8 模型采用此结构
//
// INT8 必须拆分的依据：
//   量化将整个张量映射到 8 位整数，同一张量共用一套 scale。坐标值域 0~640，分数值域 0~1，
//   若共用步长 2.5，则 0.87 会被舍入为 0，导致分数全部失效、无法检出目标。
//   拆分后 boxes 使用步长 2.5，scores 使用 1/255，各自独立，分数得以保留。
//
// 两个输出分别读取，后续过滤与 NMS 走同一条路径，
// 因此单输出与双输出模型的结果可直接比较。

#include "rknn_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "rknn_api.h"

// 静默开关: 多路并行时 8 个 detector 各打一套日志会刷屏, 关掉.
namespace {
bool g_verbose = true;
}  // namespace

void rknn_set_verbose(bool on) { g_verbose = on; }

#define VPRINT(...) do { if (g_verbose) std::printf(__VA_ARGS__); } while (0)

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

// 打印张量维度，排查问题时的主要依据
void print_dims(const char* tag, const rknn_tensor_attr& a) {
  VPRINT("%s dims=[", tag);
  for (uint32_t i = 0; i < a.n_dims; ++i) {
    VPRINT("%d%s", a.dims[i], i + 1 == a.n_dims ? "" : ", ");
  }
  VPRINT("]\n");
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

  // head4 模型输出 DFL 距离而非坐标，需在此换算：
  //   head3：输出 cx cy w h，单位为 640 空间像素
  //   head4：输出 l t r b 四个距离，单位为 anchor 网格数，值域 0~15
  // 换算依据：head3 的 boxes 张量 scale=2.5017（每步进 2.5 像素），
  //           head4 的距离张量 scale=0.0469（每步进 0.047），量化精度提升约 53 倍。
  // 识别方式：依据坐标输出的张量名中是否含 "dfl"
  //   head4 为 /model.22/dfl/Reshape_1_output_0，head3 为 /model.22/Mul_2_output_0
  bool boxes_is_dist = false;
  std::vector<float> anchor_x;        // 8400 个 anchor 网格中心 x 坐标
  std::vector<float> anchor_y;        // 8400 个 anchor 网格中心 y 坐标
  std::vector<float> stride_tab;      // 8400 个 stride，按 8 / 16 / 32 分三段

  // 输出张量的量化属性。float32 输出时 out_quant 为 false；
  // int8 输出需按 (q - zp) * scale 反量化后方可使用。
  bool out_quant[2] = {false, false};
  float out_scale[2] = {1.f, 1.f};
  int out_zp[2] = {0, 0};
};

RknnDetector::RknnDetector(const std::string& model_path, int imgsz, double conf, double iou)
    : impl_(new Impl()), imgsz_(imgsz), conf_(conf), iou_(iou) {
  // 1. 将 .rknn 文件完整读入内存
  std::ifstream f(model_path, std::ios::binary);
  if (!f) throw std::runtime_error("打不开模型: " + model_path);
  impl_->model.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  if (impl_->model.empty()) throw std::runtime_error("模型文件是空的: " + model_path);

  // 2. 初始化。参数三为模型大小，参数四为 0 表示不加密
  check(rknn_init(&impl_->ctx, impl_->model.data(),
                  static_cast<uint32_t>(impl_->model.size()), 0, nullptr),
        "rknn_init");

  // RK3588 含 3 个 NPU 核心，默认仅启用一个。此处切换为三核并行以提高推理吞吐。
  // 若调用方稍后用 set_core() 指定单核，会覆盖这里的设置。
  if (rknn_set_core_mask(impl_->ctx, RKNN_NPU_CORE_0_1_2) == 0) {
    VPRINT("NPU: 三核并行已开启\n");
  } else {
    VPRINT("NPU: 三核设置失败, 继续用默认单核\n");
  }

  // 3. 查询模型的输入输出数量
  rknn_input_output_num io_num;
  std::memset(&io_num, 0, sizeof(io_num));
  check(rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
        "rknn_query(IN_OUT_NUM)");
  VPRINT("RKNN 模型: 输入 %u 个, 输出 %u 个\n", io_num.n_input, io_num.n_output);

  // 4. 打印输入形状
  rknn_tensor_attr in_attr;
  std::memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  check(rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
        "rknn_query(INPUT_ATTR)");
  print_dims("输入:", in_attr);
  VPRINT("      fmt=%d type=%d\n", static_cast<int>(in_attr.fmt),
              static_cast<int>(in_attr.type));

  // 5. 查询输出形状，据此确定 n_rows / n_cols，不硬编码
  impl_->n_output = static_cast<int>(io_num.n_output);
  int cols = 0;

  if (impl_->n_output == 1) {
    // 单输出：[1, 5, 8400]，前 4 行为坐标，其后为各类分数
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
    // 双输出：一个 [1, 4, 8400] 存坐标，一个 [1, C, 8400] 存分数
    // 按行数而非输出顺序识别：4 行为坐标，其余为分数。
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

  // 查每路输出的量化参数：2.5.0 默认输出 int8，得自己反量化
  for (int i = 0; i < impl_->n_output && i < 2; ++i) {
    rknn_tensor_attr a;
    std::memset(&a, 0, sizeof(a));
    a.index = i;
    if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) == RKNN_SUCC) {
      if (a.type == RKNN_TENSOR_INT8 && a.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
        impl_->out_quant[i] = true;
        impl_->out_scale[i] = a.scale;
        impl_->out_zp[i] = a.zp;
        VPRINT("输出[%d] int8: scale=%.6f zp=%d fmt=%d w_stride=%u n_elems=%u size=%u dims=[%d,%d,%d,%d]\n", i,
                    static_cast<double>(a.scale), a.zp, static_cast<int>(a.fmt), a.w_stride,
                    a.n_elems, a.size, a.dims[0], a.dims[1], a.dims[2], a.dims[3]);
      } else {
        VPRINT("输出[%d] 是浮点（直接用）\n", i);
      }
    }
  }

  impl_->n_cols = cols > 0 ? cols : 8400;
  VPRINT("按 %s 解析: 类别 %d 个, 候选框 %d 个\n",
              impl_->n_output == 1 ? "单输出 [4+C, N]" : "双输出 boxes/scores", impl_->n_cls,
              impl_->n_cols);

  // ===== 认一下这是 head3 还是 head4，head4 要预先建好 anchor 表 =====
  {
    rknn_tensor_attr a;
    std::memset(&a, 0, sizeof(a));
    a.index = impl_->boxes_index;
    if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) == RKNN_SUCC) {
      // 张量名含 "dfl" 即为 head4（输出距离而非坐标）
      impl_->boxes_is_dist = std::string(a.name).find("dfl") != std::string::npos;
    }
  }

  if (impl_->boxes_is_dist) {
    // 构建 anchor 网格表。三个 stride 层依次拼接：
    //   80x80 = 6400 个，stride 8
    //   40x40 = 1600 个，stride 16
    //   20x20 =  400 个，stride 32
    // 网格中心坐标 = (列号 + 0.5, 行号 + 0.5)，与 ONNX 图中的常量逐元素一致。
    struct SegBlk { int n; int g; float stride; };
    const SegBlk segs[3] = {{6400, 80, 8.f}, {1600, 40, 16.f}, {400, 20, 32.f}};
    const int N = impl_->n_cols;
    impl_->anchor_x.resize(static_cast<size_t>(N));
    impl_->anchor_y.resize(static_cast<size_t>(N));
    impl_->stride_tab.resize(static_cast<size_t>(N));
    int i = 0;
    for (const auto& s : segs) {
      for (int k = 0; k < s.n && i < N; ++k, ++i) {
        impl_->anchor_x[static_cast<size_t>(i)] = static_cast<float>(k % s.g) + 0.5f;
        impl_->anchor_y[static_cast<size_t>(i)] = static_cast<float>(k / s.g) + 0.5f;
        impl_->stride_tab[static_cast<size_t>(i)] = s.stride;
      }
    }
    VPRINT("模型类型: head4（输出 DFL 距离，坐标在 CPU 里换算）\n");
    {
      // 打印各段首尾，确认三个 stride 段的网格都生成对了
      struct { const char* tag; int lo; } chk[3] = {{"seg1(80x80, s8) ", 0},
                                                   {"seg2(40x40,s16) ", 6400},
                                                   {"seg3(20x20,s32) ", 8000}};
      for (const auto& c : chk) {
        const size_t lo = static_cast<size_t>(c.lo);
        const size_t hi = static_cast<size_t>(c.lo + (c.lo == 0 ? 6399 : (c.lo == 6400 ? 1599 : 399)));
        VPRINT("  %s 首(%.1f,%.1f) 尾(%.1f,%.1f) stride=%.0f\n", c.tag,
                    impl_->anchor_x[lo], impl_->anchor_y[lo], impl_->anchor_x[hi],
                    impl_->anchor_y[hi], impl_->stride_tab[lo]);
      }
    }
  } else {
    VPRINT("模型类型: head3（输出已是 cx cy w h 坐标，直接用）\n");
  }
}

RknnDetector::~RknnDetector() {
  if (impl_ && impl_->ctx) rknn_destroy(impl_->ctx);
}

void RknnDetector::set_core(int core_id) {
  // RKNN_NPU_CORE_0 / _1 / _2 分别是 0x1 / 0x2 / 0x4；
  // RKNN_NPU_CORE_AUTO (0x0) 表示交给驱动自动选择，等同于不绑定。
  if (core_id < 0) {
    if (rknn_set_core_mask(impl_->ctx, RKNN_NPU_CORE_AUTO) == 0) {
      VPRINT("NPU: 恢复自动选择核\n");
    }
    return;
  }
  const rknn_core_mask masks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
  const int idx = core_id % 3;
  if (rknn_set_core_mask(impl_->ctx, masks[idx]) == 0) {
    VPRINT("NPU: 已绑定到核 %d\n", idx);
  } else {
    VPRINT("NPU: 绑定核 %d 失败，继续使用默认设置\n", idx);
  }
}

std::vector<Detection> RknnDetector::detect(const cv::Mat& frame) {
  // ===== 第 1 步：letterbox，与 ONNX 版完全一致 =====
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

  // ===== 第 2 步：提交输入并执行推理 =====
  // 1.6.0 运行库处理 float32 输入时会段错误，因此使用 uint8 + NHWC，
  // 即 RKNN 官方示例的标准做法：传入原始像素，归一化与布局转换由驱动完成。
  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = rgb.data;
  inputs[0].size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
  inputs[0].pass_through = 0;                  // 0 = 让驱动按下面两项做格式转换
  inputs[0].type = RKNN_TENSOR_UINT8;          // 原始像素, 0~255
  inputs[0].fmt = RKNN_TENSOR_NHWC;            // 高-宽-通道, 跟内存里的排法一致
  const auto t_in0 = std::chrono::steady_clock::now();
  check(rknn_inputs_set(impl_->ctx, 1, inputs), "rknn_inputs_set");
  const auto t_in1 = std::chrono::steady_clock::now();

  // 分别统计数据传输与 NPU 计算耗时，用于定位性能瓶颈
  const auto t_run0 = std::chrono::steady_clock::now();
  check(rknn_run(impl_->ctx, nullptr), "rknn_run");
  const auto t_run1 = std::chrono::steady_clock::now();
  {
    static double sum_in = 0.0, sum_run = 0.0;
    static int cnt = 0;
    sum_in += std::chrono::duration<double, std::milli>(t_in1 - t_in0).count();
    sum_run += std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    if (++cnt % 100 == 0) {
      VPRINT("[PERF] 喂数据 %.2f ms | NPU 计算 %.2f ms | 合计 %.2f ms（上限 %.1f fps）\n",
                  sum_in / cnt, sum_run / cnt, (sum_in + sum_run) / cnt,
                  1000.0 / ((sum_in + sum_run) / cnt));
    }
  }

  // ===== 第 3 步：读取输出（单输出取 1 个，双输出取 2 个）=====
  const int n_out = impl_->n_output;
  std::vector<rknn_output> outputs(static_cast<size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    std::memset(&outputs[static_cast<size_t>(i)], 0, sizeof(rknn_output));
    outputs[static_cast<size_t>(i)].index = i;
    // 由驱动完成反量化。该路径对 int8 输出同样有效；
    // 手工反量化会与驱动的处理冲突。
    outputs[static_cast<size_t>(i)].want_float = 1;
    outputs[static_cast<size_t>(i)].is_prealloc = 0;   // 由库分配，用完调用 release
  }
  check(rknn_outputs_get(impl_->ctx, n_out, outputs.data(), nullptr), "rknn_outputs_get");

  // 两种模型在此统一为同一读取方式：coord 前 4 行为 cx cy w h，score 为各类分数。
  // int8 输出先反量化到 float 缓冲，后续逻辑无需感知量化。
  std::vector<float> coord_buf;
  std::vector<float> score_buf;

  auto dequant_into = [](const void* src, size_t n, float scale, int zp,
                         std::vector<float>& dst) {
    const auto* p = static_cast<const int8_t*>(src);
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = (static_cast<float>(p[i]) - static_cast<float>(zp)) * scale;
    }
  };

  const float* coord = nullptr;
  const float* score = nullptr;
  if (n_out == 1) {
    coord = static_cast<const float*>(outputs[0].buf);
    score = coord + static_cast<size_t>(4) * impl_->n_cols;
  } else {
    coord = static_cast<const float*>(outputs[static_cast<size_t>(impl_->boxes_index)].buf);
    score = static_cast<const float*>(outputs[static_cast<size_t>(impl_->scores_index)].buf);
  }

  const int n_cols = impl_->n_cols;
  const int n_cls = impl_->n_cls;

  // ===== 第 4 步：分数过滤。每个候选框取分数最高的类别 =====
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

    double x1, y1, x2, y2;   // 640 空间中的左上、右下坐标（未扣除补边）
    if (impl_->boxes_is_dist) {
      // head4：输出为四个距离，转换为边界坐标
      //   coord[0..3] 依次为左 l、上 t、右 r、下 b 距离（单位为网格数，值域 0~15）
      //   x1 = anchor_x - l    x2 = anchor_x + r
      //   y1 = anchor_y - t    y2 = anchor_y + b
      // 再乘 stride 映射到 640 空间。该换算与被移除的 9 个 ONNX 算子等价。
      const size_t idx = static_cast<size_t>(i);
      const double ax = impl_->anchor_x[idx];
      const double ay = impl_->anchor_y[idx];
      const double st = impl_->stride_tab[idx];

      const double l = coord[idx];
      const double t = coord[static_cast<size_t>(1) * n_cols + idx];
      const double r = coord[static_cast<size_t>(2) * n_cols + idx];
      const double b = coord[static_cast<size_t>(3) * n_cols + idx];

      const double ex1 = ax - l;
      const double ey1 = ay - t;
      const double ex2 = ax + r;
      const double ey2 = ay + b;

      x1 = ex1 * st;
      y1 = ey1 * st;
      x2 = ex2 * st;
      y2 = ey2 * st;
    } else {
      // ---- head3: 本来就是 cx cy w h，直接算 ----
      const double cx = coord[static_cast<size_t>(0) * n_cols + i];
      const double cy = coord[static_cast<size_t>(1) * n_cols + i];
      const double w = coord[static_cast<size_t>(2) * n_cols + i];
      const double h = coord[static_cast<size_t>(3) * n_cols + i];
      x1 = cx - w / 2;
      y1 = cy - h / 2;
      x2 = cx + w / 2;
      y2 = cy + h / 2;
    }

    Detection d;
    d.x1 = (x1 - dx) / scale;
    d.y1 = (y1 - dy) / scale;
    d.x2 = (x2 - dx) / scale;
    d.y2 = (y2 - dy) / scale;
    d.score = best;
    d.cls = best_c;
    boxes.push_back(d);
  }

  rknn_outputs_release(impl_->ctx, n_out, outputs.data());

  // ===== 第 5 步：NMS，与 ONNX 版完全一致 =====
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
