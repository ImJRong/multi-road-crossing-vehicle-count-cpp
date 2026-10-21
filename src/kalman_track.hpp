// 卡尔曼跟踪：预测轨迹当前帧的位置，再用检测框校正。
// 状态 8 维（cx,cy,w,h 及各自速度），观测 4 维（cx,cy,w,h）。

#pragma once

#include <memory>

#include <opencv2/opencv.hpp>

class KalmanBox {
 public:
  // 新轨迹出现时调一次：用第一个检测框当初始状态，速度先设 0
  void init(const cv::Rect2f& box) {
    kf_ = std::make_shared<cv::KalmanFilter>(8, 4, 0, CV_32F);

    const float w = std::max(1.0f, box.width);
    const float h = std::max(1.0f, box.height);
    // 实测（test1 静止车 721 帧）：检测框中心抖动 σ≈0.8px，框宽抖动 2.7%
    // 所以观测噪声取 2px 足够（留余量）；过程噪声同量级 -> K≈0.5，跟手又不抖
    const float std_meas = 2.0f;        // 观测噪声(px)：实测 0.8，取 2 留余量
    const float std_q = 2.0f;           // 过程噪声(px)：和观测同量级
    const float std_vel = h / 160.0f;   // 速度误差

    // 状态转移：位置 += 速度（一帧走一步）
    cv::setIdentity(kf_->transitionMatrix);
    for (int i = 0; i < 4; ++i) kf_->transitionMatrix.at<float>(i, i + 4) = 1.0f;

    // 观测矩阵：从 8 维状态里取出前 4 维
    kf_->measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i) kf_->measurementMatrix.at<float>(i, i) = 1.0f;

    // 过程噪声：位置那几项和观测噪声同量级，速度那几项小一个数量级
    cv::setIdentity(kf_->processNoiseCov);
    for (int i = 0; i < 4; ++i) kf_->processNoiseCov.at<float>(i, i) = std_q * std_q;
    for (int i = 4; i < 8; ++i) kf_->processNoiseCov.at<float>(i, i) = std_vel * std_vel;

    // 观测噪声：检测框本身的误差
    cv::setIdentity(kf_->measurementNoiseCov);
    for (int i = 0; i < 4; ++i) kf_->measurementNoiseCov.at<float>(i, i) = std_meas * std_meas;

    // 初始协方差：位置按 std_pos，速度完全不知道所以给个大值
    cv::setIdentity(kf_->errorCovPost);
    for (int i = 0; i < 4; ++i) kf_->errorCovPost.at<float>(i, i) = std_meas * std_meas;
    for (int i = 4; i < 8; ++i) kf_->errorCovPost.at<float>(i, i) = 1e4f;

    // 初始状态：位置就是检测框，速度 0。不调 correct，直接写进去。
    kf_->statePost = cv::Mat::zeros(8, 1, CV_32F);
    kf_->statePost.at<float>(0) = box.x + w * 0.5f;
    kf_->statePost.at<float>(1) = box.y + h * 0.5f;
    kf_->statePost.at<float>(2) = w;
    kf_->statePost.at<float>(3) = h;

    pred_ = box;
  }

  // 每帧调一次：按速度往前推一步，返回"这一帧它大概在哪"
  cv::Rect2f predict() {
    if (!kf_) return pred_;
    const cv::Mat st = kf_->predict();
    pred_ = to_rect(st);
    return pred_;
  }

  // 配上检测框之后调：把预测和检测加权融合，写回状态
  void correct(const cv::Rect2f& box) {
    if (!kf_) return;
    cv::Mat z(4, 1, CV_32F);
    z.at<float>(0) = box.x + box.width * 0.5f;
    z.at<float>(1) = box.y + box.height * 0.5f;
    z.at<float>(2) = box.width;
    z.at<float>(3) = box.height;
    kf_->correct(z);
  }

  bool ready() const { return kf_ != nullptr; }

  const cv::Rect2f& pred() const { return pred_; }

  // 滤波后的当前状态（correct 之后的值）。输出框和计数都用它，
  // 不要再在外面叠一层 EMA —— 那是重复平滑，只会多 1~2 帧滞后。
  cv::Rect2f state() const { return kf_ ? to_rect(kf_->statePost) : pred_; }

 private:
  // 8 维状态的前 4 个是 cx, cy, w, h，还原成左上角 + 宽高
  static cv::Rect2f to_rect(const cv::Mat& st) {
    const float cx = st.at<float>(0);
    const float cy = st.at<float>(1);
    const float w = st.at<float>(2);
    const float h = st.at<float>(3);
    return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
  }

  std::shared_ptr<cv::KalmanFilter> kf_;
  cv::Rect2f pred_;
};

