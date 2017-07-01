#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF() :
  is_initialized_(false),
  previous_timestamp_(0),
  R_laser_(2, 2),
  R_radar_(3, 3),
  H_laser_(2, 4)
{
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;
  
  //measurement matrix for laser (lidar)
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    float px = 0;
    float py = 0;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      const float rho = measurement_pack.raw_measurements_[0];
      const float phi = measurement_pack.raw_measurements_[1];
      px = rho * cosf(phi);
      py = rho * sinf(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    }

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << px, py, 0, 0;

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0,   0,
               0, 1, 0,   0,
               0, 0, 100, 0,
               0, 0, 0,   100;

    ekf_.F_ = MatrixXd(4, 4);
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.I_ = MatrixXd::Identity(4, 4);
    
    // timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0,  dt,
             0, 0, 1,  0,
             0, 0, 0,  1;
  
  const float varx = 9;
  const float vary = 9;
  const float dt2 = dt * dt;
  const float dt3 = dt2 * dt;
  const float dt4 = dt3 * dt;
  ekf_.Q_ << (dt4/4*varx), 0,            (dt3/2*varx), 0,
             0,            (dt4/4*vary), 0,            (dt3/2*vary),
             (dt3/2*varx), 0,            (dt2*varx),   0,
             0,            (dt3/2*vary), 0,            (dt2*vary);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
