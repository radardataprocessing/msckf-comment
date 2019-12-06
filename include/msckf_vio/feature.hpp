/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_FEATURE_H
#define MSCKF_VIO_FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"

namespace msckf_vio {

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int FeatureIDType;

  /*
   * @brief OptimizationConfig Configuration parameters
   *    for 3d feature position optimization.
   */
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  // triangulate
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @param cam_states : input camera poses.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion(
      const CamStateServer& cam_states) const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @param cam_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition(
      const CamStateServer& cam_states);


  // An unique identifier for the feature.
  // In case of long time running, the variable
  // type of id is set to FeatureIDType in order
  // to avoid duplication.
  FeatureIDType id;

  // id for next feature
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector4d> > > observations;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

  // Noise for a normalized feature measurement.
  static double observation_noise;

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;

/**
 * to compute an estimate of the position of a tracked feature fj, we employ intersection. To avoid local minima, and for better numerical
 * stability, during this process we use an inverse-depth parametrization of the feature position. In particular, if {Cn} is the camera 
 * frame in which the feature was observed for the first time, then the feature coordinates with respect to the camera at the i-th time
 * instant are:
 * pfj_Ci = R_Cn_to_Ci*pfj_Cn+pCn_Ci
 * the above equation can be rewritten as:
 *                                | Xj_Cn/Zj_Cn | 
 * pfj_Ci = Zj_Cn * (R_Cn_to_Ci * | Yj_Cn/Zj_Cn | + 1/Zj_Cn*pCn_Ci)
 *                                |      1      |
 * 
 *                                | alpha_j |
 *        = Zj_Cn * (R_Cn_to_Ci * | beta_j  | + rho_j * pCn_Ci)
 *                                |    1    |
 * 
 *                  | hi1(alpha_j, beta_j, rho_j) |
 *        = Zj_Cn * | hi2(alpha_j, beta_j, rho_j) | 
 *                  | hi3(alpha_j, beta_j, rho_j) | 
 * 
 * alpha_j = Xj_Cn/Zj_Cn
 * beta_j = Yj_Cn/Zj_Cn
 * rho_j = 1/Zj_Cn
 * 
 * if j means the feature id, i means the camera id
 *                | Xj_Ci |
 * zj_i = 1/Zj_Ci*| Yj_Ci | + nj_i
 * nj_i is the 2*1 image noise vector with covariance matrix Rj_i = sigma*I2x2, which is
 *                                      | hi1(alpha_j, beta_j, rho_j) |
 * zj_i = 1/hi3(alpha_j, beta_j, rho_j)*| hi2(alpha_j, beta_j, rho_j) | + nj_i
 */
void Feature::cost(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

/**
 * (h1/h3)' = (h1'*h3 - h1*h3')/(h3*h3) 
 * (h2/h3)' = (h2'*h3 - h2*h3')/(h3*h3) 
 * h1 = R11*alpha + R12*beta + R13 + rho*T11
 * h2 = R21*alpha + R22*beta + R23 + rho*T21
 * h3 = R31*alpha + R32*beta + R33 + rho*T31
 * the derivatives of h1, h2 and h3 are
 * a(h1)/a(alpha) = R11        a(h1)/a(beta) = R12         a(h1)/a(rho) = T11
 * a(h2)/a(alpha) = R21        a(h2)/a(beta) = R22         a(h2)/a(rho) = T21
 * a(h3)/a(alpha) = R31        a(h3)/a(beta) = R32         a(h3)/a(rho) = T31 
 */
void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Isometry3d is a 4x4 matrix, means the euclidean transformation
  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>(); // leftCols<q> means the first q columns
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;// the estimated pixel coordinate and the measured pixel coordinate

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);
  // w is the weight induced by huber kernel
  return;
}


/*
 * denote the 3D point as C, 2D pixel as x, pose as P, we have
 * x = P*C   use cross multiply, we can get [x]x(PC) = 0
 * | 0  -1  y |   |P(row0)C|
 * | 1   0  -x| * |P(row1)C| = 0
 * |-y   x  0 |   |P(row2)C|
 * -P(row1)C+yP(row2)C = 0   (1)
 * P(row0)C-xP(row2)C = 0    (2)
 * -yP(row0)C+xP(row1)C = 0  (3)
 * (3) can be expressed using (1) and (2)
 */
/**
 * the position of the point in c1 coordinate, denoted as p1, is: [ z1(0)*d1  z1(1)*d1  d1 ].transpose(), which is d1*[ z1(0)  z1(1)  1 ].transpose()
 * the position of the point in c2 coordinate, denoted as p2, is: [ z2(0)*d2  z2(1)*d2  d2 ].transpose(), which is d2*[ z2(0)  z2(1)  1 ].transpose()
 * m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0)
 * d1*m + T = z2*d2
 * m(0)*d1+T(0) = z2(0)*d2    (1)
 * m(1)*d1+T(1) = z2(1)*d2    (2)
 * m(2)*d1+T(2) = d2          (3)
 * substituting from equation (3) to equation (1) and equation(2)
 * m(0)*d1+T(0) = z2(0)*(m(2)*d1+T(2))
 * m(1)*d1+T(1) = z2(1)*(m(2)*d1+T(2))
 * which is
 * (m(0)-z2(0)*m(2))*d1 = z2(0)*T(2)-T(0)
 * (m(1)-z2(1)*m(2))*d1 = z2(1)*T(2)-T(1)
 * 
 * d1*| m(0)-z2(0)*m(2) | = | z2(0)*T(2)-T(0) |
 *    | m(1)-z2(1)*m(2) |   | z2(1)*T(2)-T(1) |
 * 
 * A*d1 = b
 * A.transpose()*A*d1 = A.transpose()*b
 * d1 = (A.transpose()*A).inverse()*A.transpose()*b
 */
void Feature::generateInitialGuess(
    const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  // the comment of the declare is not correct, T_c1_c2 is the transform from c1 coordinate to c2 coordinate
  Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}

/**
 *  
 */
bool Feature::checkMotion(
    const CamStateServer& cam_states) const {

  const StateIDType& first_cam_id = observations.begin()->first;
  const StateIDType& last_cam_id = (--observations.end())->first;

  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = quaternionToRotation(
      cam_states.find(first_cam_id)->second.orientation).transpose();
  first_cam_pose.translation() =
    cam_states.find(first_cam_id)->second.position;

  Eigen::Isometry3d last_cam_pose;
  last_cam_pose.linear() = quaternionToRotation(
      cam_states.find(last_cam_id)->second.orientation).transpose();
  last_cam_pose.translation() =
    cam_states.find(last_cam_id)->second.position;

  /**
   * // Store the observations of the features in the
   * // state_id(key)-image_coordinates(value) manner.
   * std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
   *   Eigen::aligned_allocator<
   *     std::pair<const StateIDType, Eigen::Vector4d> > > observations; 
   * Eigen::Vector4d here stores u0, v0, u1, v1
   */
  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction(
      observations.begin()->second(0),
      observations.begin()->second(1), 1.0);
  feature_direction = feature_direction / feature_direction.norm();// get the direction of feature in the frame who first observe the feature
  feature_direction = first_cam_pose.linear()*feature_direction;// transform the direction from first observing frame to world frame

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  Eigen::Vector3d translation = last_cam_pose.translation() -
    first_cam_pose.translation();
  double parallel_translation =
    translation.transpose()*feature_direction;// get the part of the translation which is parallel to the feature direction
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;// get the part of the translation which is orthogonal to the feature direction

  // if the orthogonal part of the translation is bigger than the threshold, return true
  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)// the default value of translation_threshold is 0.2
    return true;
  else return false;
}


/**
 *  
 */
bool Feature::initializePosition(
    const CamStateServer& cam_states) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  for (auto& m : observations) {
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input cam_states buffer.
    // get the cam_state of the observation, if not, continue to process next observation
    auto cam_state_iter = cam_states.find(m.first);
    if (cam_state_iter == cam_states.end()) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());
    measurements.push_back(m.second.tail<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam0_pose;
    cam0_pose.linear() = quaternionToRotation(
        cam_state_iter->second.orientation).transpose();
    cam0_pose.translation() = cam_state_iter->second.position;

    Eigen::Isometry3d cam1_pose;
    cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();// I think here T_cam0_cam1 means the transform of left and right view of the stereo

    cam_poses.push_back(cam0_pose);
    cam_poses.push_back(cam1_pose);
  }

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  // use the transform matrix from 0th pose to (cam_poses.size - 1)th pose, and the 2D pixel coordinate in the first and last frame
  // to compute the point coordinate in 0th camera frame
  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
      measurements[measurements.size()-1], initial_position);
  // compute (X/Z, Y/Z, 1/Z) under 0th camera frame
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.  outer loop compute jacobian while inner loop just use the result of the outer loop
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      // compute the jacobian of (h1/h3) and (h2/h3) about alpha, beta and rho.
      // h1, h2, h3 is defined in equation(37), alpha is (X/Z), beta is (Y/Z), rho is (1/Z)
      jacobian(cam_poses[i], solution, measurements[i], J, r, w);

      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      // (J.transpose()*J + lambda*Identity)*x = J.transpose()*r      I think it's the same as LM method
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;// update (X/Z, Y/Z, 1/Z) of the feature point in 0th camera frame
      delta_norm = delta.norm();// delta_norm is the delta value of alpha,beta and rho

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }// compute the new reprojection error using the new feature solution

      // check whether the cost is reducing, if yes, the cost is reducing, we can reduce lambda to release the the constrainting region
      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  // convert from (X/Z, Y/Z, 1/Z) to (X, Y, Z)
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  // compute the feature 3D coordinate in every camera frame who is observing it, if the depth in any of the camera is negative, assign
  // is_valid_solution using false 
  bool is_valid_solution = true;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d position =
      pose.linear()*final_position + pose.translation();
    if (position(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }

  // Convert the feature position to the world frame.
  // convert the feature position from the 0th camera frame to world frame
  position = T_c0_w.linear()*final_position + T_c0_w.translation();

  // if is_valid_solution is true, assign is_initialized using true
  if (is_valid_solution)
    is_initialized = true;
  // return is_valid_solution
  return is_valid_solution;
}
} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_H
