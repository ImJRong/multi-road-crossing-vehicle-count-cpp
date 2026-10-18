#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// 两个框的交并比 IoU = 交集面积 / 并集面积, 范围 0~1.
// NMS 用它判断"这两个框是不是在描述同一辆车".
double iou(const Detection& a, const Detection& b) {
  const double iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const double ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0.0;     // 完全不相交
  const double inter = iw * ih;
  const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);   // 并集 = A + B - 交集
}

}  // namespace

Detector::Detector(const std::string& model_path, int imgsz, double conf, double iou)
    : imgsz_(imgsz),
      conf_(conf),
      iou_(iou),
      env_(ORT_LOGGING_LEVEL_WARNING, "traffic_count"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  Ort::SessionOptions opt;
  opt.SetIntraOpNumThreads(4);   // 单次推理内部用几个线程. 核显机器上 4 个够, 再多反而抢核
  session_ = Ort::Session(env_, model_path.c_str(), opt);
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
  // ===== 第 1 步: letterbox =====
  // 模型只吃固定尺寸的方图(比如 640x640), 但输入是 1280x960 的长方形.
  // 直接拉伸会让车变形, 长宽比一乱, 模型的精度会掉.
  // 所以做法是: 等比缩放 + 四周补灰边, 多出来的地方填 114 灰.
  const double scale = std::min(imgsz_ / static_cast<double>(frame.cols),
                                imgsz_ / static_cast<double>(frame.rows));
  const int nw = static_cast<int>(std::round(frame.cols * scale));
  const int nh = static_cast<int>(std::round(frame.rows * scale));
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(nw, nh));

  // 补边时左右(或上下)各留一半, 所以要记住偏移量 dx/dy.
  // 后面把框坐标还原回原图时, 要先减掉这个偏移.
  const int dx = (imgsz_ - nw) / 2;
  const int dy = (imgsz_ - nh) / 2;
  cv::Mat canvas(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  // blobFromImage 做三件事: 归一化(除以 255)、BGR 转 RGB(最后的 true)、排成 NCHW 四维.
  // 这三步必须和训练时的预处理完全一致, 差一步精度就差很多.
  cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(imgsz_, imgsz_),
                                        cv::Scalar(), true, false);

  // ===== 第 2 步: 推理 =====
  // 输入输出名字必须和 onnx 里的一致, 写错会直接抛异常.
  // 我们这个模型(单类 vehicle)的输出是 [1, 5, 8400]:
  //   5    = 4 个坐标(cx, cy, w, h) + 1 个类别分数
  //   8400 = 模型在图上铺的候选框数量, 固定值, 和 imgsz 相关
  const char* in_names[] = {"images"};
  const char* out_names[] = {"output0"};
  std::vector<int64_t> shape = {1, 3, imgsz_, imgsz_};
  Ort::Value input = Ort::Value::CreateTensor<float>(
      mem_, blob.ptr<float>(), 1 * 3 * imgsz_ * imgsz_, shape.data(), shape.size());
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

  const float* out = outputs[0].GetTensorData<float>();
  const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const int n_cols = static_cast<int>(out_shape[2]);   // 8400: 候选框个数
  const int n_rows = static_cast<int>(out_shape[1]);   // 5: 每个框几个数

  // ===== 第 3 步: 按分数过滤 =====
  // 输出是 [1, 5, 8400], 内存里按行排: 同一个属性的 8400 个值连在一起.
  // 所以取第 i 个框的第 j 个值, 下标是 j * 8400 + i, 不是 i * 5 + j.
  // 这个顺序搞反了, 读出来的全是乱七八糟的数.
  std::vector<Detection> boxes;
  for (int i = 0; i < n_cols; ++i) {
    double best = 0;
    int best_c = 0;
    for (int c = 4; c < n_rows; ++c) {     // 从第 4 行开始是类别分数
      const double s = out[static_cast<size_t>(c) * n_cols + i];
      if (s > best) {
        best = s;
        best_c = c - 4;
      }
    }
    if (best < conf_) continue;            // 分数不够, 这个候选框不要

    // 模型给的是中心点 + 宽高, 要转成左上角 + 右下角.
    // 还原公式是 letterbox 的逆过程: 先减补边偏移, 再除以缩放比.
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

  // ===== 第 4 步: NMS(非极大值抑制) =====
  // 同一辆车会被模型给出好几个框(稍微错开一点), 只需要留最好的一个.
  // 做法: 按分数从高到低排, 依次取; 取到的和已留下的比, 重叠超过阈值就丢掉.
  std::sort(boxes.begin(), boxes.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  for (const auto& b : boxes) {
    bool drop = false;
    for (const auto& k : keep) {
      if (k.cls == b.cls && iou(b, k) > iou_) {   // 只和同类比, 不同类不互相压制
        drop = true;
        break;
      }
    }
    if (!drop) keep.push_back(b);
  }
  return keep;
}
