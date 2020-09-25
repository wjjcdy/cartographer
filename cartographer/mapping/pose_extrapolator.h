/*
 * Copyright 2017 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
#define CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_

#include <deque>
#include <memory>

#include "cartographer/common/time.h"
#include "cartographer/mapping/imu_tracker.h"
#include "cartographer/mapping/pose_extrapolator_interface.h"
#include "cartographer/sensor/imu_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

// Keep poses for a certain duration to estimate linear and angular velocity.
// Uses the velocities to extrapolate motion. Uses IMU and/or odometry data if
// available to improve the extrapolation.
class PoseExtrapolator : public PoseExtrapolatorInterface {
 public:
  explicit PoseExtrapolator(common::Duration pose_queue_duration,
                            double imu_gravity_time_constant);

  PoseExtrapolator(const PoseExtrapolator&) = delete;
  PoseExtrapolator& operator=(const PoseExtrapolator&) = delete;

  // 采用imu数据初始化推算器， 2d可暂时不考虑
  static std::unique_ptr<PoseExtrapolator> InitializeWithImu(
      common::Duration pose_queue_duration, double imu_gravity_time_constant,
      const sensor::ImuData& imu_data);

  // Returns the time of the last added pose or Time::min() if no pose was added
  // yet.
  // 获取上次的位置和推算位置的时间
  common::Time GetLastPoseTime() const override;
  common::Time GetLastExtrapolatedTime() const override;

  // 添加传感器数据
  void AddPose(common::Time time, const transform::Rigid3d& pose) override;
  void AddImuData(const sensor::ImuData& imu_data) override;
  void AddOdometryData(const sensor::OdometryData& odometry_data) override;

  // 推算出估计位置
  transform::Rigid3d ExtrapolatePose(common::Time time) override;

  // 推算出带重力加速度预测位置
  ExtrapolationResult ExtrapolatePosesWithGravity(
      const std::vector<common::Time>& times) override;

  // Returns the current gravity alignment estimate as a rotation from
  // the tracking frame into a gravity aligned frame
  // 获取重力加速度方向
  Eigen::Quaterniond EstimateGravityOrientation(common::Time time) override;

 private:
  // 从历史位置更新出当前速度 
  void UpdateVelocitiesFromPoses();

  void TrimImuData();
  void TrimOdometryData();

  // 更新推算器
  void AdvanceImuTracker(common::Time time, ImuTracker* imu_tracker) const;

  // 推算出平移矩阵
  Eigen::Quaterniond ExtrapolateRotation(common::Time time,
                                         ImuTracker* imu_tracker) const;
  // 推算出旋转向量
  Eigen::Vector3d ExtrapolateTranslation(common::Time time);
  //时间参数，两次预测估计最小时间差
  const common::Duration pose_queue_duration_;

  // 带时间戳的位置结构体
  struct TimedPose {
    common::Time time;
    transform::Rigid3d pose;
  };
  // 带时间戳的位置队列
  std::deque<TimedPose> timed_pose_queue_;
  // 从位置队列估计的线速度和角速度
  Eigen::Vector3d linear_velocity_from_poses_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity_from_poses_ = Eigen::Vector3d::Zero();

  // 重力加速度常数
  const double gravity_time_constant_;
  // Imu 原始data 队列，一般仅保留两个或最新预测时间之后的所有序列
  std::deque<sensor::ImuData> imu_data_;

  // 全局的航向推算器
  std::unique_ptr<ImuTracker> imu_tracker_;
  // 共享临时航向角推算器
  std::unique_ptr<ImuTracker> odometry_imu_tracker_;
  std::unique_ptr<ImuTracker> extrapolation_imu_tracker_;

  // 推算新的位置缓存
  TimedPose cached_extrapolated_pose_;

  // 里程计队列信息，一般仅保留两个或最新预测时间之后的所有序列
  std::deque<sensor::OdometryData> odometry_data_;
  // 通过里程计估计线速度和角速度
  Eigen::Vector3d linear_velocity_from_odometry_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity_from_odometry_ = Eigen::Vector3d::Zero();
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
