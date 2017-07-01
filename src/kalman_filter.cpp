#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  UpdateCommon(z - H_ * x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  const float px = x_(0);
  const float py = x_(1);
  const float vx = x_(2);
  const float vy = x_(3);
  const float c1 = px * px + py * py;
  const float c2 = sqrt(c1);
  const float error = 1e-5;
  
  // avoid division by 0
  if (c1 < error)
    return;
  
  // use h(x) to calculate predicted state
  VectorXd z_pred(3);
  z_pred(0) = c2;
  z_pred(1) = atan2f(py, px);
  z_pred(2) = (px * vx + py * vy) / c2;

  UpdateCommon(z - z_pred);
}

void KalmanFilter::UpdateCommon(const VectorXd &pred_error) {
  auto y = pred_error;
  y[1] = atan2(sin(y[1]), cos(y[1]));
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K = P_ * Ht * S.inverse();
  
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

