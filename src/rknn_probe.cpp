// RKNN 模型探针: 打印每个输入/输出张量的全部量化属性, 并真实跑一帧,
// 分别用 want_float=1 / want_float=0 取输出, 对比数值分布.
// 编译(板子上):
//   g++ -O2 -std=c++17 -I/home/orangepi/code/src /tmp/rknn_probe.cpp -o /tmp/rknn_probe \
//       -lrknnrt $(pkg-config --cflags --libs opencv4)
// 用法:
//   /tmp/rknn_probe <model.rknn> <video.mp4> [frame_no]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

namespace {

const char* dtype_name(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "FLOAT32";
    case RKNN_TENSOR_FLOAT16: return "FLOAT16";
    case RKNN_TENSOR_INT8: return "INT8";
    case RKNN_TENSOR_UINT8: return "UINT8";
    case RKNN_TENSOR_INT16: return "INT16";
    default: return "OTHER";
  }
}

const char* qnt_name(rknn_tensor_qnt_type q) {
  switch (q) {
    case RKNN_TENSOR_QNT_NONE: return "NONE";
    case RKNN_TENSOR_QNT_DFP: return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "AFFINE";
    default: return "OTHER";
  }
}

void print_attr(const char* tag, const rknn_tensor_attr& a) {
  std::printf("%s index=%u name=%s dims=[", tag, a.index, a.name);
  for (uint32_t i = 0; i < a.n_dims; ++i)
    std::printf("%d%s", a.dims[i], i + 1 == a.n_dims ? "" : ",");
  std::printf("] n_elems=%u size=%u fmt=%d type=%s qnt=%s fl=%d zp=%d scale=%.9f w_stride=%u\n",
              a.n_elems, a.size, static_cast<int>(a.fmt), dtype_name(a.type), qnt_name(a.qnt_type),
              static_cast<int>(a.fl), a.zp, static_cast<double>(a.scale), a.w_stride);
}

void stats_float(const char* tag, const float* p, size_t n, size_t show = 12) {
  if (n == 0) return;
  float mn = p[0], mx = p[0];
  double sum = 0;
  size_t nonzero = 0, gt01 = 0, gt05 = 0;
  for (size_t i = 0; i < n; ++i) {
    const float v = p[i];
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += v;
    if (v != 0.f) ++nonzero;
    if (v > 0.01f) ++gt01;
    if (v > 0.5f) ++gt05;
  }
  std::printf("%s n=%zu min=%.6g max=%.6g mean=%.6g nonzero=%zu >0.01=%zu >0.5=%zu\n", tag, n,
              static_cast<double>(mn), static_cast<double>(mx), sum / n, nonzero, gt01, gt05);
  std::printf("%s   head:", tag);
  for (size_t i = 0; i < std::min(n, show); ++i) std::printf(" %.5g", static_cast<double>(p[i]));
  std::printf("\n");
}

void stats_int8(const char* tag, const int8_t* p, size_t n, int zp) {
  if (n == 0) return;
  int mn = p[0], mx = p[0];
  long long sum = 0;
  size_t eq_zp = 0;
  int hist[256] = {0};
  for (size_t i = 0; i < n; ++i) {
    const int v = p[i];
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += v;
    if (v == zp) ++eq_zp;
    ++hist[v + 128];
  }
  std::printf("%s n=%zu min=%d max=%d mean=%.3f ==zp(%d)=%zu\n", tag, n, mn, mx,
              static_cast<double>(sum) / n, zp, eq_zp);
  std::printf("%s   distinct:", tag);
  int shown = 0;
  for (int i = 0; i < 256 && shown < 10; ++i) {
    if (hist[i]) {
      std::printf(" %d:%d", i - 128, hist[i]);
      ++shown;
    }
  }
  std::printf("\n%s   head:", tag);
  for (size_t i = 0; i < std::min<size_t>(n, 12); ++i) std::printf(" %d", p[i]);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("用法: %s <model.rknn> [video.mp4] [frame_no]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string video = argc > 2 ? argv[2] : "";
  const int frame_no = argc > 3 ? std::atoi(argv[3]) : 200;

  FILE* fp = std::fopen(model_path.c_str(), "rb");
  if (!fp) { std::printf("打不开模型\n"); return 1; }
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<unsigned char> buf(static_cast<size_t>(sz));
  if (std::fread(buf.data(), 1, buf.size(), fp) != buf.size()) { std::fclose(fp); return 1; }
  std::fclose(fp);
  std::printf("模型: %s (%ld 字节)\n", model_path.c_str(), sz);

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, buf.data(), static_cast<uint32_t>(buf.size()), 0, nullptr);
  if (ret != 0) { std::printf("rknn_init 失败 %d\n", ret); return 1; }

  rknn_sdk_version ver;
  std::memset(&ver, 0, sizeof(ver));
  if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
    std::printf("runtime api=%s driver=%s\n", ver.api_version, ver.drv_version);

  char custom[4096];
  std::memset(custom, 0, sizeof(custom));
  if (rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, custom, sizeof(custom)) == 0)
    std::printf("custom_string(len=%zu): %.300s\n", std::strlen(custom), custom);

  rknn_input_output_num io;
  std::memset(&io, 0, sizeof(io));
  if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) return 1;
  std::printf("输入 %u 个, 输出 %u 个\n", io.n_input, io.n_output);

  rknn_tensor_attr in;
  std::memset(&in, 0, sizeof(in));
  in.index = 0;
  if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)) == 0) print_attr("[IN ]", in);

  std::vector<rknn_tensor_attr> out(io.n_output);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::memset(&out[i], 0, sizeof(rknn_tensor_attr));
    out[i].index = i;
    if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out[i], sizeof(rknn_tensor_attr)) == 0)
      print_attr("[OUT]", out[i]);

    rknn_tensor_attr nat;
    std::memset(&nat, 0, sizeof(nat));
    nat.index = i;
    if (rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &nat, sizeof(nat)) == 0)
      print_attr("[NAT]", nat);
  }

  // ---- 造一张输入 ----
  const int imgsz = 640;
  cv::Mat frame(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
  if (!video.empty()) {
    cv::VideoCapture cap(video);
    if (cap.isOpened()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, frame_no);
      cv::Mat src;
      if (cap.read(src) && !src.empty()) {
        const double s = std::min(imgsz / static_cast<double>(src.cols),
                                  imgsz / static_cast<double>(src.rows));
        const int nw = static_cast<int>(std::round(src.cols * s));
        const int nh = static_cast<int>(std::round(src.rows * s));
        cv::Mat rs;
        cv::resize(src, rs, cv::Size(nw, nh));
        const int dx = (imgsz - nw) / 2, dy = (imgsz - nh) / 2;
        rs.copyTo(frame(cv::Rect(dx, dy, nw, nh)));
        std::printf("输入图: %s 第 %d 帧 (%dx%d -> %dx%d)\n", video.c_str(), frame_no, src.cols,
                    src.rows, nw, nh);
      }
    }
  }
  cv::Mat rgb;
  cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = rgb.data;
  inputs[0].size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
  inputs[0].pass_through = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  if (rknn_inputs_set(ctx, 1, inputs) != 0) { std::printf("inputs_set 失败\n"); return 1; }
  if (rknn_run(ctx, nullptr) != 0) { std::printf("run 失败\n"); return 1; }

  // ---- A. want_float = 1 ----
  std::printf("\n===== want_float=1 (驱动反量化) =====\n");
  std::vector<rknn_output> outs(io.n_output);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::memset(&outs[i], 0, sizeof(rknn_output));
    outs[i].index = i;
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  if (rknn_outputs_get(ctx, io.n_output, outs.data(), nullptr) == 0) {
    for (uint32_t i = 0; i < io.n_output; ++i) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "[OUT%d float]", i);
      stats_float(tag, static_cast<const float*>(outs[i].buf), outs[i].size / sizeof(float));
    }
    rknn_outputs_release(ctx, io.n_output, outs.data());
  } else {
    std::printf("outputs_get(want_float=1) 失败\n");
  }

  // ---- B. want_float = 0 ----
  std::printf("\n===== want_float=0 (原始 int8) =====\n");
  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::memset(&outs[i], 0, sizeof(rknn_output));
    outs[i].index = i;
    outs[i].want_float = 0;
    outs[i].is_prealloc = 0;
  }
  if (rknn_outputs_get(ctx, io.n_output, outs.data(), nullptr) == 0) {
    for (uint32_t i = 0; i < io.n_output; ++i) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "[OUT%d int8]", i);
      if (out[i].type == RKNN_TENSOR_INT8)
        stats_int8(tag, static_cast<const int8_t*>(outs[i].buf),
                   outs[i].size / sizeof(int8_t), out[i].zp);
      else
        std::printf("%s 不是 int8 (type=%s), 跳过\n", tag, dtype_name(out[i].type));
    }
    rknn_outputs_release(ctx, io.n_output, outs.data());
  } else {
    std::printf("outputs_get(want_float=0) 失败\n");
  }

  rknn_destroy(ctx);
  return 0;
}
