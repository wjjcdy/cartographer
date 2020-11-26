/*
 * Copyright 2016 The Cartographer Authors
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

#include "cartographer/mapping/internal/2d/pose_graph_2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/internal/2d/overlapping_submaps_trimmer_2d.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

static auto* kWorkQueueDelayMetric = metrics::Gauge::Null();
static auto* kWorkQueueSizeMetric = metrics::Gauge::Null();
static auto* kConstraintsSameTrajectoryMetric = metrics::Gauge::Null();
static auto* kConstraintsDifferentTrajectoryMetric = metrics::Gauge::Null();
static auto* kActiveSubmapsMetric = metrics::Gauge::Null();
static auto* kFrozenSubmapsMetric = metrics::Gauge::Null();
static auto* kDeletedSubmapsMetric = metrics::Gauge::Null();

// 构造函数
/*
配置
优化器
线程池
 */
PoseGraph2D::PoseGraph2D(
    const proto::PoseGraphOptions& options,
    std::unique_ptr<optimization::OptimizationProblem2D> optimization_problem,
    common::ThreadPool* thread_pool)
    : options_(options),
      optimization_problem_(std::move(optimization_problem)),
      constraint_builder_(options_.constraint_builder_options(), thread_pool),
      thread_pool_(thread_pool) {
        // 默认没有
  if (options.has_overlapping_submaps_trimmer_2d()) {
    const auto& trimmer_options = options.overlapping_submaps_trimmer_2d();
    AddTrimmer(absl::make_unique<OverlappingSubmapsTrimmer2D>(
        trimmer_options.fresh_submaps_count(),
        trimmer_options.min_covered_area(),
        trimmer_options.min_added_submaps_count()));
  }
}

// 析构函数
PoseGraph2D::~PoseGraph2D() {
  WaitForAllComputations();                     // 等待所有计算结束
  absl::MutexLock locker(&work_queue_mutex_);
  CHECK(work_queue_ == nullptr);
}

// 初始化正在维护的submap为一个submap id
/*
根据优化器里存储的submap id和正在维护的submap进行判断，如果一个都没有，则创建一个0 index的submap
如果上个submapid为正在维护的front， 则返回上一个和新增加的submap id
如果上个submapid为正在维护的back，说明都已经处理过了，则返回上上个和上个的submap id 
*/
std::vector<SubmapId> PoseGraph2D::InitializeGlobalSubmapPoses(
    const int trajectory_id, const common::Time time,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  CHECK(!insertion_submaps.empty());
  // 获取优化器中的submap地址
  const auto& submap_data = optimization_problem_->submap_data();
  // 表明为第一个submap
  if (insertion_submaps.size() == 1) {
    // If we don't already have an entry for the first submap, add one.
    // 如果optimization_problem_的submap data中还没有存储任何id，则添加一个新的
    // 获取存储的submap id的个数
    if (submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
      // trajectory_id初始位置的个数如果大于0，即存在初始位置
      if (data_.initial_trajectory_poses.count(trajectory_id) > 0) {
        // 创建两个trajectory_id的链接， 即有相对位置的trajectory id
        data_.trajectory_connectivity_state.Connect(
            trajectory_id,
            data_.initial_trajectory_poses.at(trajectory_id).to_trajectory_id,
            time);
      }
      // 添加submap的全局位置
      optimization_problem_->AddSubmap(
          trajectory_id, transform::Project2D(
                             ComputeLocalToGlobalTransform(
                                 data_.global_submap_poses_2d, trajectory_id) *
                             insertion_submaps[0]->local_pose()));
    }
    CHECK_EQ(1, submap_data.SizeOfTrajectoryOrZero(trajectory_id));
    // submapid 为0 添加到集合中
    const SubmapId submap_id{trajectory_id, 0};
    CHECK(data_.submap_data.at(submap_id).submap == insertion_submaps.front());
    return {submap_id};
  }
  CHECK_EQ(2, insertion_submaps.size());
  // 获取trajectory_id的最后一个submap id
  const auto end_it = submap_data.EndOfTrajectory(trajectory_id);
  // 检测是否为空（理论上不为空，因为上面为空时，已经初始化添加了一个）
  CHECK(submap_data.BeginOfTrajectory(trajectory_id) != end_it);
  // 取出前一个submap id
  const SubmapId last_submap_id = std::prev(end_it)->id;

  //如果submap data中上一个id为正在维护的submap中的front，说明back为新的还未处理
  if (data_.submap_data.at(last_submap_id).submap ==
      insertion_submaps.front()) {
    // In this case, 'last_submap_id' is the ID of
    // 'insertions_submaps.front()' and 'insertions_submaps.back()' is new.
    // 获取上个submap ID的全局位置
    const auto& first_submap_pose = submap_data.at(last_submap_id).global_pose;
    // 根据上一个submap推出相邻的下一个submap的全局位置
    optimization_problem_->AddSubmap(
        trajectory_id,
        first_submap_pose *
            constraints::ComputeSubmapPose(*insertion_submaps[0]).inverse() *
            constraints::ComputeSubmapPose(*insertion_submaps[1]));
    // 返回上一个和当前新的submap id集合
    // 新的id为上个id+1
    return {last_submap_id,
            SubmapId{trajectory_id, last_submap_id.submap_index + 1}};
  }
  // 表明存储的上一个submapid 为 正在维护的back submap， 则说明已经处理过
  CHECK(data_.submap_data.at(last_submap_id).submap ==
        insertion_submaps.back());
  // 获取上上个submapid
  const SubmapId front_submap_id{trajectory_id,
                                 last_submap_id.submap_index - 1};
  CHECK(data_.submap_data.at(front_submap_id).submap ==
        insertion_submaps.front());
  // 返回前前和前一个的submap
  return {front_submap_id, last_submap_id};
}

// posegraph 添加一个节点
/*
input:
此次match后的处理后的点云数据，包括水平投影的点云，重力方向和在local submap的位置
trajectory id
匹配采用的submap 地图
此次scan 的全局pose
 */
NodeId PoseGraph2D::AppendNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps,
    const transform::Rigid3d& optimized_pose) {
  absl::MutexLock locker(&mutex_);
  // 判断是否需要添加trajectory id
  AddTrajectoryIfNeeded(trajectory_id);
  // 此 trajectory id的轨迹是否存在或更改，暂时只是判断无用
  if (!CanAddWorkItemModifying(trajectory_id)) {
    LOG(WARNING) << "AddNode was called for finished or deleted trajectory.";
  }
  // 添加一个scan id， 返回trajectoryid和对应的scan idex
  const NodeId node_id = data_.trajectory_nodes.Append(
      trajectory_id, TrajectoryNode{constant_data, optimized_pose});
  // 记录轨迹节点个数
  ++data_.num_trajectory_nodes;
  // Test if the 'insertion_submap.back()' is one we never saw before.
  // 只有正在维护的两个submap都存在时，并且第二submap第一次出现
  if (data_.submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
      std::prev(data_.submap_data.EndOfTrajectory(trajectory_id))
              ->data.submap != insertion_submaps.back()) {
    // We grow 'data_.submap_data' as needed. This code assumes that the first
    // time we see a new submap is as 'insertion_submaps.back()'.
    // 添加submap id，
    const SubmapId submap_id =
        data_.submap_data.Append(trajectory_id, InternalSubmapData());
    // 闭环中submap节点采用维护的两个中后一个submap
    data_.submap_data.at(submap_id).submap = insertion_submaps.back();
    LOG(INFO) << "Inserted submap " << submap_id << ".";
    // 被处理的submap个数自加
    kActiveSubmapsMetric->Increment();
  }
  return node_id;
}


// 在全局SLAM中，每帧点云经过local slam匹配后，获取较为准确的scan pose即对应的submap后，调用
/*
input:
此次match后的处理后的点云数据，包括水平投影的点云，重力方向和在local submap的位置
trajectory id
匹配采用的submap 地图
 */
NodeId PoseGraph2D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  // 将点云的原点坐标的local的位置转换为全局位置
  const transform::Rigid3d optimized_pose(
      GetLocalToGlobalTransform(trajectory_id) * constant_data->local_pose);

  // 添加一个节点
  const NodeId node_id = AppendNode(constant_data, trajectory_id,
                                    insertion_submaps, optimized_pose);
  // We have to check this here, because it might have changed by the time we
  // execute the lambda.
  // 查看维护的两个submap的front是否插入完成（两个submap，front用于匹配和更新的）
  // 如果一个submap finished，则需要进行闭环检测
  const bool newly_finished_submap =
      insertion_submaps.front()->insertion_finished();

  // 把计算约束的工作较为workitem完成，是个多线程工作组
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    return ComputeConstraintsForNode(node_id, insertion_submaps,
                                     newly_finished_submap);
  });
  return node_id;
}

// 将一个函数地址加入到一个工作队列中
void PoseGraph2D::AddWorkItem(
    const std::function<WorkItem::Result()>& work_item) {
  absl::MutexLock locker(&work_queue_mutex_);
  // 如果工作队列未被初始化，则先初始化并启动线程执行任务队列
  if (work_queue_ == nullptr) {
    work_queue_ = absl::make_unique<WorkQueue>();
    auto task = absl::make_unique<common::Task>();
    task->SetWorkItem([this]() { DrainWorkQueue(); });
    thread_pool_->Schedule(std::move(task));
  }
  // 将当前任务插入到工作队列中，等待执行
  const auto now = std::chrono::steady_clock::now();
  work_queue_->push_back({now, work_item});
  kWorkQueueSizeMetric->Set(work_queue_->size());
  kWorkQueueDelayMetric->Set(
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - work_queue_->front().time)
          .count());
}

// 是否需要将trajectory_id， 暂时没看懂具体意义？？？？？
void PoseGraph2D::AddTrajectoryIfNeeded(const int trajectory_id) {
  data_.trajectories_state[trajectory_id];
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::FINISHED);
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::DELETED);
  CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
        InternalTrajectoryState::DeletionState::NORMAL);
  // 在整个闭环轨迹链接数据中添加trajectory id信息
  data_.trajectory_connectivity_state.Add(trajectory_id);
  // Make sure we have a sampler for this trajectory.
  // 全局定位采样器要存在，默认为0.003秒，采样一次
  if (!global_localization_samplers_[trajectory_id]) {
    global_localization_samplers_[trajectory_id] =
        absl::make_unique<common::FixedRatioSampler>(
            options_.global_sampling_ratio());
  }
}

//也可考虑imu
void PoseGraph2D::AddImuData(const int trajectory_id,
                             const sensor::ImuData& imu_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddImuData(trajectory_id, imu_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}
//也可考虑odometry、
void PoseGraph2D::AddOdometryData(const int trajectory_id,
                                  const sensor::OdometryData& odometry_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddOdometryData(trajectory_id, odometry_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph2D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddFixedFramePoseData(trajectory_id,
                                                   fixed_frame_pose_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

//暂时没用
void PoseGraph2D::AddLandmarkData(int trajectory_id,
                                  const sensor::LandmarkData& landmark_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      for (const auto& observation : landmark_data.landmark_observations) {
        data_.landmark_nodes[observation.id].landmark_observations.emplace_back(
            PoseGraphInterface::LandmarkNode::LandmarkObservation{
                trajectory_id, landmark_data.time,
                observation.landmark_to_tracking_transform,
                observation.translation_weight, observation.rotation_weight});
      }
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

// 计算指定的nodeid和submapid计算其约束
void PoseGraph2D::ComputeConstraint(const NodeId& node_id,
                                    const SubmapId& submap_id) {
  bool maybe_add_local_constraint = false;         // 声明默认不添加局部约束条件
  bool maybe_add_global_constraint = false;        // 声明默认不添加全局约束条件
  const TrajectoryNode::Data* constant_data;
  const Submap2D* submap;
  {
    absl::MutexLock locker(&mutex_);
    // 判断指定的submapid是否为finished状态，理论上不是不应该计算
    CHECK(data_.submap_data.at(submap_id).state == SubmapState::kFinished);
    // submap如果未插入完成，无需计算
    if (!data_.submap_data.at(submap_id).submap->insertion_finished()) {
      // Uplink server only receives grids when they are finished, so skip
      // constraint search before that.
      return;
    }

    // 获取两个id最新的那个时刻
    const common::Time node_time = GetLatestNodeTime(node_id, submap_id);
    // 最近轨迹链接的时间
    const common::Time last_connection_time =
        data_.trajectory_connectivity_state.LastConnectionTime(
            node_id.trajectory_id, submap_id.trajectory_id);

    // 如果nodeid和submapid在同一个轨迹中，或者更新节点的时间小于全局约束间隔时间
    // 简单说局部位置是较为可靠的，可添加为局部约束
    // 并且在同一scanid所在的轨迹与submap所在的轨迹为同一条，则进入局部约束
    if (node_id.trajectory_id == submap_id.trajectory_id ||
        node_time <
            last_connection_time +
                common::FromSeconds(
                    options_.global_constraint_search_after_n_seconds())) {
      // If the node and the submap belong to the same trajectory or if there
      // has been a recent global constraint that ties that node's trajectory to
      // the submap's trajectory, it suffices to do a match constrained to a
      // local search window.
      // 一条轨迹全局采样一定时间间隔时，可添加全局约束
      maybe_add_local_constraint = true;
    } else if (global_localization_samplers_[node_id.trajectory_id]->Pulse()) {
      maybe_add_global_constraint = true;
    }
    // 获取node id的相关data
    constant_data = data_.trajectory_nodes.at(node_id).constant_data.get();
    // 获取submapid的相关data
    submap = static_cast<const Submap2D*>(
        data_.submap_data.at(submap_id).submap.get());
  }

  if (maybe_add_local_constraint) {
    // 获取相对位置
    const transform::Rigid2d initial_relative_pose =
        optimization_problem_->submap_data()
            .at(submap_id)
            .global_pose.inverse() *
        optimization_problem_->node_data().at(node_id).global_pose_2d;
    // 添加约束，添加约束即是进行闭环匹配，更新约束内容，即相对位置
    constraint_builder_.MaybeAddConstraint(
        submap_id, submap, node_id, constant_data, initial_relative_pose);
  } else if (maybe_add_global_constraint) {
    // 添加全局约束
    constraint_builder_.MaybeAddGlobalConstraint(submap_id, submap, node_id,
                                                 constant_data);
  }
}

// 计算scan的node和submap的相对关系
/*
input:
nodeid
submap
submap是否finished 标志
 */
WorkItem::Result PoseGraph2D::ComputeConstraintsForNode(
    const NodeId& node_id,
    std::vector<std::shared_ptr<const Submap2D>> insertion_submaps,
    const bool newly_finished_submap) {
  std::vector<SubmapId> submap_ids;
  std::vector<SubmapId> finished_submap_ids;
  std::set<NodeId> newly_finished_submap_node_ids;
  {
    absl::MutexLock locker(&mutex_);
    // 获取该nodeid的点云数据和位置信息
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    // 获取submap对应的submapid vector
    // 如果第一个则为index 为0， 否则会返回最新的两个连续的submap id   
    submap_ids = InitializeGlobalSubmapPoses(
        node_id.trajectory_id, constant_data->time, insertion_submaps);

    CHECK_EQ(submap_ids.size(), insertion_submaps.size());
    //front的submap 为match 的submap
    const SubmapId matching_id = submap_ids.front();
    //获取scan 的在submap的local pose
    const transform::Rigid2d local_pose_2d =
        transform::Project2D(constant_data->local_pose *
                             transform::Rigid3d::Rotation(
                                 constant_data->gravity_alignment.inverse()));
    // 获取全局绝对位置，通过优化后的submap 全局位置和submap的local位置求出转移矩阵
    const transform::Rigid2d global_pose_2d =
        optimization_problem_->submap_data().at(matching_id).global_pose *
        constraints::ComputeSubmapPose(*insertion_submaps.front()).inverse() *
        local_pose_2d;
    // 优化器中增加轨迹节点信息
    optimization_problem_->AddTrajectoryNode(
        matching_id.trajectory_id,
        optimization::NodeSpec2D{constant_data->time, local_pose_2d,
                                 global_pose_2d,
                                 constant_data->gravity_alignment});
    // 遍历正在维护的两个submap
    for (size_t i = 0; i < insertion_submaps.size(); ++i) {
      const SubmapId submap_id = submap_ids[i];
      // Even if this was the last node added to 'submap_id', the submap will
      // only be marked as finished in 'data_.submap_data' further below.
      CHECK(data_.submap_data.at(submap_id).state ==
            SubmapState::kNoConstraintSearch);
      // submap_id中的对应的添加nodeid
      data_.submap_data.at(submap_id).node_ids.emplace(node_id);
      //nodeid的scan pose在submap的相对位置 
      const transform::Rigid2d constraint_transform =
          constraints::ComputeSubmapPose(*insertion_submaps[i]).inverse() *
          local_pose_2d;
      // 增加一个约束条件，为submap插入的scan id， submapid和nodeid ，约束为相对位置
      data_.constraints.push_back(
          Constraint{submap_id,
                     node_id,
                     {transform::Embed3D(constraint_transform),
                      options_.matcher_translation_weight(),
                      options_.matcher_rotation_weight()},
                     Constraint::INTRA_SUBMAP});
    }

    // TODO(gaschler): Consider not searching for constraints against
    // trajectories scheduled for deletion.
    // TODO(danielsievers): Add a member variable and avoid having to copy
    // them out here.
    for (const auto& submap_id_data : data_.submap_data) {
      // 判断当前graph中所有的finished状态并存储
      if (submap_id_data.data.state == SubmapState::kFinished) {
        CHECK_EQ(submap_id_data.data.node_ids.count(node_id), 0);
        finished_submap_ids.emplace_back(submap_id_data.id);
      }
    }
    // 如果当前为新finished的
    if (newly_finished_submap) {
      const SubmapId newly_finished_submap_id = submap_ids.front();
      InternalSubmapData& finished_submap_data =
          data_.submap_data.at(newly_finished_submap_id);
      CHECK(finished_submap_data.state == SubmapState::kNoConstraintSearch);
      // 将新的submap设置为finished，表明已经增加约束条件了
      finished_submap_data.state = SubmapState::kFinished;
      newly_finished_submap_node_ids = finished_submap_data.node_ids;
    }
  }

  // 计算scan的nodeid 与所有存储闭环的submapid全部计算下约束
  for (const auto& submap_id : finished_submap_ids) {
    ComputeConstraint(node_id, submap_id);
  }

  // 存在新的finished的submap
  if (newly_finished_submap) {
    const SubmapId newly_finished_submap_id = submap_ids.front();
    // We have a new completed submap, so we look into adding constraints for
    // old nodes.
    // 当出现一个新的submap完成时，需要将graph存储的所有节点与当前submap进行计算约束
    for (const auto& node_id_data : optimization_problem_->node_data()) {
      const NodeId& node_id = node_id_data.id;
      if (newly_finished_submap_node_ids.count(node_id) == 0) {
        ComputeConstraint(node_id, newly_finished_submap_id);
      }
    }
  }
  // 此次创建约束结束
  constraint_builder_.NotifyEndOfNode();
  absl::MutexLock locker(&mutex_);
  // 记录距离上次闭环处理增加的节点个数
  ++num_nodes_since_last_loop_closure_;
  // 当新增加节点个数达到一定时，则调用优化处理结果
  if (options_.optimize_every_n_nodes() > 0 &&
      num_nodes_since_last_loop_closure_ > options_.optimize_every_n_nodes()) {
    return WorkItem::Result::kRunOptimization;
  }
  return WorkItem::Result::kDoNotRunOptimization;
}

//获取node_id和submap_id存储时最新的时刻
common::Time PoseGraph2D::GetLatestNodeTime(const NodeId& node_id,
                                            const SubmapId& submap_id) const {
  // node ID 在轨迹的时刻
  common::Time time = data_.trajectory_nodes.at(node_id).constant_data->time;
  // submapid的submap数据
  const InternalSubmapData& submap_data = data_.submap_data.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *data_.submap_data.at(submap_id).node_ids.rbegin();
    time = std::max(
        time,
        data_.trajectory_nodes.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

//增加约束时更新轨迹连接关系
void PoseGraph2D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  // 约束是nodeID不是submap中的插入的scan
  CHECK_EQ(constraint.tag, Constraint::INTER_SUBMAP);
  // 获取最新时间
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  // 增加相邻关系
  data_.trajectory_connectivity_state.Connect(
      constraint.node_id.trajectory_id, constraint.submap_id.trajectory_id,
      time);
}

//删除需要删除的轨迹
void PoseGraph2D::DeleteTrajectoriesIfNeeded() {
  TrimmingHandle trimming_handle(this);
  // 遍历所有轨迹状态
  for (auto& it : data_.trajectories_state) {
    if (it.second.deletion_state ==
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION) {
      // TODO(gaschler): Consider directly deleting from data_, which may be
      // more complete.
      auto submap_ids = trimming_handle.GetSubmapIds(it.first);
      for (auto& submap_id : submap_ids) {
        trimming_handle.TrimSubmap(submap_id);
      }
      it.second.state = TrajectoryState::DELETED;
      it.second.deletion_state = InternalTrajectoryState::DeletionState::NORMAL;
    }
  }
}

//处理已经添加约束的结果
void PoseGraph2D::HandleWorkQueue(
    const constraints::ConstraintBuilder2D::Result& result) {
      // 将新的约束添加到全局约束队列中
  {
    absl::MutexLock locker(&mutex_);
    data_.constraints.insert(data_.constraints.end(), result.begin(),
                             result.end());
  }
  // 进行全局优化
  RunOptimization();

  //如果设置了全局优化回调函数，则进行调用
  if (global_slam_optimization_callback_) {
    std::map<int, NodeId> trajectory_id_to_last_optimized_node_id;
    std::map<int, SubmapId> trajectory_id_to_last_optimized_submap_id;
    {
      absl::MutexLock locker(&mutex_);
      const auto& submap_data = optimization_problem_->submap_data();
      const auto& node_data = optimization_problem_->node_data();
      //遍历每条轨迹
      for (const int trajectory_id : node_data.trajectory_ids()) {
        if (node_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
            submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
          continue;
        }
        //仅将每条轨迹的最新节点进行存储
        trajectory_id_to_last_optimized_node_id.emplace(
            trajectory_id,
            std::prev(node_data.EndOfTrajectory(trajectory_id))->id);
        trajectory_id_to_last_optimized_submap_id.emplace(
            trajectory_id,
            std::prev(submap_data.EndOfTrajectory(trajectory_id))->id);
      }
    }
    //回调函数传输出去最近优化的submapid和nodeid
    global_slam_optimization_callback_(
        trajectory_id_to_last_optimized_submap_id,
        trajectory_id_to_last_optimized_node_id);
  }

  {
    // 根据约束结果，更新轨迹间的链接关系
    absl::MutexLock locker(&mutex_);
    for (const Constraint& constraint : result) {
      UpdateTrajectoryConnectivity(constraint);
    }
    //删除需要删除的轨迹
    DeleteTrajectoriesIfNeeded();
    TrimmingHandle trimming_handle(this);
    for (auto& trimmer : trimmers_) {
      trimmer->Trim(&trimming_handle);
    }
    trimmers_.erase(
        std::remove_if(trimmers_.begin(), trimmers_.end(),
                       [](std::unique_ptr<PoseGraphTrimmer>& trimmer) {
                         return trimmer->IsFinished();
                       }),
        trimmers_.end());
    
    //闭环后累计节点进行清零
    //表明已经完成了一次闭环
    num_nodes_since_last_loop_closure_ = 0;

    // Update the gauges that count the current number of constraints.
    double inter_constraints_same_trajectory = 0;      // 同一轨迹中约束的个数
    double inter_constraints_different_trajectory = 0; // 不同轨迹中约束的个数
    // 遍历所有约束
    for (const auto& constraint : data_.constraints) {
      // submap插入的nodeid无需判断
      if (constraint.tag ==
          cartographer::mapping::PoseGraph::Constraint::INTRA_SUBMAP) {
        continue;
      }
      // 分别统计个数
      if (constraint.node_id.trajectory_id ==
          constraint.submap_id.trajectory_id) {
        ++inter_constraints_same_trajectory;
      } else {
        ++inter_constraints_different_trajectory;
      }
    }
    kConstraintsSameTrajectoryMetric->Set(inter_constraints_same_trajectory);
    kConstraintsDifferentTrajectoryMetric->Set(
        inter_constraints_different_trajectory);
  }
  // 开启任务队列缓存循环执行功能
  DrainWorkQueue();
}

// 开启任务队列缓存循环执行功能
void PoseGraph2D::DrainWorkQueue() {
  bool process_work_queue = true;
  size_t work_queue_size;
  while (process_work_queue) {
    std::function<WorkItem::Result()> work_item;
    {
      absl::MutexLock locker(&work_queue_mutex_);
      // 工作队列已空，则可直接退出
      if (work_queue_->empty()) {
        work_queue_.reset();
        return;
      }
      // 获取队列中最前的任务
      work_item = work_queue_->front().task;
      work_queue_->pop_front();
      work_queue_size = work_queue_->size();
      kWorkQueueSizeMetric->Set(work_queue_size);
    }
    // 当前工作任务的状态
    process_work_queue = work_item() == WorkItem::Result::kDoNotRunOptimization;
  }
  LOG(INFO) << "Remaining work items in queue: " << work_queue_size;
  // We have to optimize again.
  // 继续执行优化
  constraint_builder_.WhenDone(
      [this](const constraints::ConstraintBuilder2D::Result& result) {
        HandleWorkQueue(result);
      });
}

// 等待所有约束条件计算完成
void PoseGraph2D::WaitForAllComputations() {
  // 获取轨迹总节点个数
  int num_trajectory_nodes;
  {
    absl::MutexLock locker(&mutex_);
    num_trajectory_nodes = data_.num_trajectory_nodes;
  }
  // 获取完成的节点个数
  const int num_finished_nodes_at_start =
      constraint_builder_.GetNumFinishedNodes();

  // 打印临时函数
  auto report_progress = [this, num_trajectory_nodes,
                          num_finished_nodes_at_start]() {
    // Log progress on nodes only when we are actually processing nodes.
    // 如果不相等，显示完成百分比
    if (num_trajectory_nodes != num_finished_nodes_at_start) {
      std::ostringstream progress_info;
      progress_info << "Optimizing: " << std::fixed << std::setprecision(1)
                    << 100. *
                           (constraint_builder_.GetNumFinishedNodes() -
                            num_finished_nodes_at_start) /
                           (num_trajectory_nodes - num_finished_nodes_at_start)
                    << "%...";
      std::cout << "\r\x1b[K" << progress_info.str() << std::flush;
    }
  };

  // First wait for the work queue to drain so that it's safe to schedule
  // a WhenDone() callback.
  // 等待work queue 清空，
  {
    const auto predicate = [this]()
                               EXCLUSIVE_LOCKS_REQUIRED(work_queue_mutex_) {
                                 return work_queue_ == nullptr;
                               };
    absl::MutexLock locker(&work_queue_mutex_);
    while (!work_queue_mutex_.AwaitWithTimeout(
        absl::Condition(&predicate),
        absl::FromChrono(common::FromSeconds(1.)))) {
      report_progress();
    }
  }

  // Now wait for any pending constraint computations to finish.
  absl::MutexLock locker(&mutex_);
  bool notification = false;
  constraint_builder_.WhenDone(
      [this,
       &notification](const constraints::ConstraintBuilder2D::Result& result)
          LOCKS_EXCLUDED(mutex_) {
            absl::MutexLock locker(&mutex_);
            data_.constraints.insert(data_.constraints.end(), result.begin(),
                                     result.end());
            notification = true;
          });
  const auto predicate = [&notification]() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return notification;
  };
  while (!mutex_.AwaitWithTimeout(absl::Condition(&predicate),
                                  absl::FromChrono(common::FromSeconds(1.)))) {
    report_progress();
  }
  CHECK_EQ(constraint_builder_.GetNumFinishedNodes(), num_trajectory_nodes);
  std::cout << "\r\x1b[KOptimizing: Done.     " << std::endl;
}

//删除指定id的trajectory
// 实际上是对制定id的state标志为将要删除
void PoseGraph2D::DeleteTrajectory(const int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    auto it = data_.trajectories_state.find(trajectory_id);
    if (it == data_.trajectories_state.end()) {
      LOG(WARNING) << "Skipping request to delete non-existing trajectory_id: "
                   << trajectory_id;
      return;
    }
    it->second.deletion_state =
        InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;
  }
  // 添加到工作队列中，返回未执行优化
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::ACTIVE);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::DELETED);
    CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
          InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION);
    data_.trajectories_state.at(trajectory_id).deletion_state =
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

// 完成此条轨迹
void PoseGraph2D::FinishTrajectory(const int trajectory_id) {
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFinished(trajectory_id));
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FINISHED;

    // 将此轨迹中每个submap状态均改成完成
    for (const auto& submap : data_.submap_data.trajectory(trajectory_id)) {
      data_.submap_data.at(submap.id).state = SubmapState::kFinished;
    }
    return WorkItem::Result::kRunOptimization;
  });
}

// 判断此条轨迹是否完成
bool PoseGraph2D::IsTrajectoryFinished(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FINISHED;
}

void PoseGraph2D::FreezeTrajectory(const int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    data_.trajectory_connectivity_state.Add(trajectory_id);
  }
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFrozen(trajectory_id));
    // Connect multiple frozen trajectories among each other.
    // This is required for localization against multiple frozen trajectories
    // because we lose inter-trajectory constraints when freezing.
    for (const auto& entry : data_.trajectories_state) {
      const int other_trajectory_id = entry.first;
      if (!IsTrajectoryFrozen(other_trajectory_id)) {
        continue;
      }
      if (data_.trajectory_connectivity_state.TransitivelyConnected(
              trajectory_id, other_trajectory_id)) {
        // Already connected, nothing to do.
        continue;
      }
      data_.trajectory_connectivity_state.Connect(
          trajectory_id, other_trajectory_id, common::FromUniversal(0));
    }
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FROZEN;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

bool PoseGraph2D::IsTrajectoryFrozen(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FROZEN;
}

void PoseGraph2D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose, const proto::Submap& submap) {
  if (!submap.has_submap_2d()) {
    return;
  }

  const SubmapId submap_id = {submap.submap_id().trajectory_id(),
                              submap.submap_id().submap_index()};

  const transform::Rigid2d global_submap_pose_2d =
      transform::Project2D(global_submap_pose);
  {
    absl::MutexLock locker(&mutex_);
    const std::shared_ptr<const Submap2D> submap_ptr =
        std::make_shared<const Submap2D>(submap.submap_2d(),
                                         &conversion_tables_);
    AddTrajectoryIfNeeded(submap_id.trajectory_id);
    if (!CanAddWorkItemModifying(submap_id.trajectory_id)) return;
    data_.submap_data.Insert(submap_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = submap_ptr;
    // Immediately show the submap at the 'global_submap_pose'.
    data_.global_submap_poses_2d.Insert(
        submap_id, optimization::SubmapSpec2D{global_submap_pose_2d});
  }

  // TODO(MichaelGrupp): MapBuilder does freezing before deserializing submaps,
  // so this should be fine.
  if (IsTrajectoryFrozen(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Increment();
  } else {
    kActiveSubmapsMetric->Increment();
  }

  AddWorkItem(
      [this, submap_id, global_submap_pose_2d]() LOCKS_EXCLUDED(mutex_) {
        absl::MutexLock locker(&mutex_);
        data_.submap_data.at(submap_id).state = SubmapState::kFinished;
        optimization_problem_->InsertSubmap(submap_id, global_submap_pose_2d);
        return WorkItem::Result::kDoNotRunOptimization;
      });
}

// 解析addnode
void PoseGraph2D::AddNodeFromProto(const transform::Rigid3d& global_pose,
                                   const proto::Node& node) {
  const NodeId node_id = {node.node_id().trajectory_id(),
                          node.node_id().node_index()};
  std::shared_ptr<const TrajectoryNode::Data> constant_data =
      std::make_shared<const TrajectoryNode::Data>(FromProto(node.node_data()));

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(node_id.trajectory_id);
    if (!CanAddWorkItemModifying(node_id.trajectory_id)) return;
    data_.trajectory_nodes.Insert(node_id,
                                  TrajectoryNode{constant_data, global_pose});
  }

  AddWorkItem([this, node_id, global_pose]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    const auto gravity_alignment_inverse = transform::Rigid3d::Rotation(
        constant_data->gravity_alignment.inverse());
    optimization_problem_->InsertTrajectoryNode(
        node_id,
        optimization::NodeSpec2D{
            constant_data->time,
            transform::Project2D(constant_data->local_pose *
                                 gravity_alignment_inverse),
            transform::Project2D(global_pose * gravity_alignment_inverse),
            constant_data->gravity_alignment});
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph2D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
  TrajectoryData trajectory_data;
  // gravity_constant and imu_calibration are omitted as its not used in 2d

  if (data.has_fixed_frame_origin_in_map()) {
    trajectory_data.fixed_frame_origin_in_map =
        transform::ToRigid3(data.fixed_frame_origin_in_map());

    const int trajectory_id = data.trajectory_id();
    AddWorkItem([this, trajectory_id, trajectory_data]()
                    LOCKS_EXCLUDED(mutex_) {
                      absl::MutexLock locker(&mutex_);
                      if (CanAddWorkItemModifying(trajectory_id)) {
                        optimization_problem_->SetTrajectoryData(
                            trajectory_id, trajectory_data);
                      }
                      return WorkItem::Result::kDoNotRunOptimization;
                    });
  }
}

void PoseGraph2D::AddNodeToSubmap(const NodeId& node_id,
                                  const SubmapId& submap_id) {
  AddWorkItem([this, node_id, submap_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(submap_id.trajectory_id)) {
      data_.submap_data.at(submap_id).node_ids.insert(node_id);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph2D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  AddWorkItem([this, constraints]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    for (const auto& constraint : constraints) {
      CHECK(data_.trajectory_nodes.Contains(constraint.node_id));
      CHECK(data_.submap_data.Contains(constraint.submap_id));
      CHECK(data_.trajectory_nodes.at(constraint.node_id).constant_data !=
            nullptr);
      CHECK(data_.submap_data.at(constraint.submap_id).submap != nullptr);
      switch (constraint.tag) {
        case Constraint::Tag::INTRA_SUBMAP:
          CHECK(data_.submap_data.at(constraint.submap_id)
                    .node_ids.emplace(constraint.node_id)
                    .second);
          break;
        case Constraint::Tag::INTER_SUBMAP:
          UpdateTrajectoryConnectivity(constraint);
          break;
      }
      const Constraint::Pose pose = {
          constraint.pose.zbar_ij *
              transform::Rigid3d::Rotation(
                  data_.trajectory_nodes.at(constraint.node_id)
                      .constant_data->gravity_alignment.inverse()),
          constraint.pose.translation_weight, constraint.pose.rotation_weight};
      data_.constraints.push_back(Constraint{
          constraint.submap_id, constraint.node_id, pose, constraint.tag});
    }
    LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph2D::AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) {
  // C++11 does not allow us to move a unique_ptr into a lambda.
  PoseGraphTrimmer* const trimmer_ptr = trimmer.release();
  AddWorkItem([this, trimmer_ptr]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    trimmers_.emplace_back(trimmer_ptr);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph2D::RunFinalOptimization() {
  {
    AddWorkItem([this]() LOCKS_EXCLUDED(mutex_) {
      absl::MutexLock locker(&mutex_);
      optimization_problem_->SetMaxNumIterations(
          options_.max_num_final_iterations());
      return WorkItem::Result::kRunOptimization;
    });
    AddWorkItem([this]() LOCKS_EXCLUDED(mutex_) {
      absl::MutexLock locker(&mutex_);
      optimization_problem_->SetMaxNumIterations(
          options_.optimization_problem_options()
              .ceres_solver_options()
              .max_num_iterations());
      return WorkItem::Result::kDoNotRunOptimization;
    });
  }
  //等待所有约束条件
  WaitForAllComputations();
}

// 进行优化处理，并将优化结果进行更新
void PoseGraph2D::RunOptimization() {
  // 如果submap为空直接退出
  if (optimization_problem_->submap_data().empty()) {
    return;
  }

  // No other thread is accessing the optimization_problem_,
  // data_.constraints, data_.frozen_trajectories and data_.landmark_nodes
  // when executing the Solve. Solve is time consuming, so not taking the mutex
  // before Solve to avoid blocking foreground processing.
  // 调用优化，由于耗时间，故没加锁，防止阻塞其他线程
  optimization_problem_->Solve(data_.constraints, GetTrajectoryStates(),
                               data_.landmark_nodes);
  absl::MutexLock locker(&mutex_);

  //获取优化后的结果
  const auto& submap_data = optimization_problem_->submap_data();
  const auto& node_data = optimization_problem_->node_data();
  // 
  for (const int trajectory_id : node_data.trajectory_ids()) {
    // 更新所有节点的全局位置
    for (const auto& node : node_data.trajectory(trajectory_id)) {
      auto& mutable_trajectory_node = data_.trajectory_nodes.at(node.id);
      mutable_trajectory_node.global_pose =
          transform::Embed3D(node.data.global_pose_2d) *
          transform::Rigid3d::Rotation(
              mutable_trajectory_node.constant_data->gravity_alignment);
    }

    // Extrapolate all point cloud poses that were not included in the
    // 'optimization_problem_' yet.
    // 获取submap的local到global的转移矩阵
    const auto local_to_new_global =
        ComputeLocalToGlobalTransform(submap_data, trajectory_id);
    // 获取过去的转移矩阵
    const auto local_to_old_global = ComputeLocalToGlobalTransform(
        data_.global_submap_poses_2d, trajectory_id);
    //获取old到新的转移矩阵
    const transform::Rigid3d old_global_to_new_global =
        local_to_new_global * local_to_old_global.inverse();
    // 获取上次最后一个优化的节点
    const NodeId last_optimized_node_id =
        std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
    auto node_it =
        std::next(data_.trajectory_nodes.find(last_optimized_node_id));
    //将后续未优化的节点的全局pose进行转移（其转移矩阵是根据已优化的位置转移量） 
    for (; node_it != data_.trajectory_nodes.EndOfTrajectory(trajectory_id);
         ++node_it) {
      auto& mutable_trajectory_node = data_.trajectory_nodes.at(node_it->id);
      mutable_trajectory_node.global_pose =
          old_global_to_new_global * mutable_trajectory_node.global_pose;
    }
  }

  for (const auto& landmark : optimization_problem_->landmark_data()) {
    data_.landmark_nodes[landmark.first].global_landmark_pose = landmark.second;
  }
  // 将优化后的submap进行替代原队列
  data_.global_submap_poses_2d = submap_data;
}

// 判断该id的是否可modify
bool PoseGraph2D::CanAddWorkItemModifying(int trajectory_id) {
  // 从轨迹状态中找到对应id的状态
  auto it = data_.trajectories_state.find(trajectory_id);
  // 不存在
  if (it == data_.trajectories_state.end()) {
    return true;
  }
  // 存在但是为finished
  if (it->second.state == TrajectoryState::FINISHED) {
    // TODO(gaschler): Replace all FATAL to WARNING after some testing.
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has finished "
                  "but modification is requested, skipping.";
    return false;
  }
  // 将要被删除
  if (it->second.deletion_state !=
      InternalTrajectoryState::DeletionState::NORMAL) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been scheduled for deletion "
                  "but modification is requested, skipping.";
    return false;
  }
  // 已经删除
  if (it->second.state == TrajectoryState::DELETED) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been deleted "
                  "but modification is requested, skipping.";
    return false;
  }
  return true;
}

MapById<NodeId, TrajectoryNode> PoseGraph2D::GetTrajectoryNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_nodes;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph2D::GetTrajectoryNodePoses()
    const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& node_id_data : data_.trajectory_nodes) {
    absl::optional<TrajectoryNodePose::ConstantPoseData> constant_pose_data;
    if (node_id_data.data.constant_data != nullptr) {
      constant_pose_data = TrajectoryNodePose::ConstantPoseData{
          node_id_data.data.constant_data->time,
          node_id_data.data.constant_data->local_pose};
    }
    node_poses.Insert(
        node_id_data.id,
        TrajectoryNodePose{node_id_data.data.global_pose, constant_pose_data});
  }
  return node_poses;
}

std::map<int, PoseGraphInterface::TrajectoryState>
PoseGraph2D::GetTrajectoryStates() const {
  std::map<int, PoseGraphInterface::TrajectoryState> trajectories_state;
  absl::MutexLock locker(&mutex_);
  for (const auto& it : data_.trajectories_state) {
    trajectories_state[it.first] = it.second.state;
  }
  return trajectories_state;
}

std::map<std::string, transform::Rigid3d> PoseGraph2D::GetLandmarkPoses()
    const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& landmark : data_.landmark_nodes) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

void PoseGraph2D::SetLandmarkPose(const std::string& landmark_id,
                                  const transform::Rigid3d& global_pose,
                                  const bool frozen) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    data_.landmark_nodes[landmark_id].global_landmark_pose = global_pose;
    data_.landmark_nodes[landmark_id].frozen = frozen;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

sensor::MapByTime<sensor::ImuData> PoseGraph2D::GetImuData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->imu_data();
}

sensor::MapByTime<sensor::OdometryData> PoseGraph2D::GetOdometryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->odometry_data();
}

std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
PoseGraph2D::GetLandmarkNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.landmark_nodes;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph2D::GetTrajectoryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->trajectory_data();
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph2D::GetFixedFramePoseData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->fixed_frame_pose_data();
}

std::vector<PoseGraphInterface::Constraint> PoseGraph2D::constraints() const {
  std::vector<PoseGraphInterface::Constraint> result;
  absl::MutexLock locker(&mutex_);
  for (const Constraint& constraint : data_.constraints) {
    result.push_back(Constraint{
        constraint.submap_id, constraint.node_id,
        Constraint::Pose{constraint.pose.zbar_ij *
                             transform::Rigid3d::Rotation(
                                 data_.trajectory_nodes.at(constraint.node_id)
                                     .constant_data->gravity_alignment),
                         constraint.pose.translation_weight,
                         constraint.pose.rotation_weight},
        constraint.tag});
  }
  return result;
}

// 为trajectory id设置初始位置
void PoseGraph2D::SetInitialTrajectoryPose(const int from_trajectory_id,
                                           const int to_trajectory_id,
                                           const transform::Rigid3d& pose,
                                           const common::Time time) {
  absl::MutexLock locker(&mutex_);
  // 为一个结构体，表示from_trajectory_id的pose到to_trajectory_id的pose的转换的相对pose
  data_.initial_trajectory_poses[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph2D::GetInterpolatedGlobalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  CHECK_GT(data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = data_.trajectory_nodes.lower_bound(trajectory_id, time);
  if (it == data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)) {
    return data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)
        ->data.global_pose;
  }
  if (it == data_.trajectory_nodes.EndOfTrajectory(trajectory_id)) {
    return std::prev(data_.trajectory_nodes.EndOfTrajectory(trajectory_id))
        ->data.global_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.global_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.global_pose},
             time)
      .transform;
}

// trajectory_id轨迹从local转换到global pose的转移矩阵
transform::Rigid3d PoseGraph2D::GetLocalToGlobalTransform(
    const int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  // 实际是submap的全局位置转换矩阵
  return ComputeLocalToGlobalTransform(data_.global_submap_poses_2d,
                                       trajectory_id);
}

std::vector<std::vector<int>> PoseGraph2D::GetConnectedTrajectories() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.Components();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapData(
    const SubmapId& submap_id) const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock(submap_id);
}

//获取所有submap的map
MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetAllSubmapData() const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock();
}

//获取所有submap的pose
MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph2D::GetAllSubmapPoses() const {
  absl::MutexLock locker(&mutex_);
  MapById<SubmapId, SubmapPose> submap_poses;
  for (const auto& submap_id_data : data_.submap_data) {
    auto submap_data = GetSubmapDataUnderLock(submap_id_data.id);
    submap_poses.Insert(
        submap_id_data.id,
        PoseGraph::SubmapPose{submap_data.submap->num_range_data(),
                              submap_data.pose});
  }
  return submap_poses;
}

// 计算trajectory_id轨迹中pose到全局位置的转移矩阵
transform::Rigid3d PoseGraph2D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec2D>& global_submap_poses,
    const int trajectory_id) const {
  // 查找已经优化过的trajectory id中的submap的位置，如果有可返回存储的值
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = data_.initial_trajectory_poses.find(trajectory_id);
    if (it != data_.initial_trajectory_poses.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;           // 初始为单位矩阵，后续通过相对位置推导每个位置
    } else {
      return transform::Rigid3d::Identity();     // 仅是唯一个，表明为初始submap，转移矩阵为单位矩阵
    }
  }
  // 如果不存在，则返回由上个优化后的submapid的位置的相对位置换算出的位置
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  return transform::Embed3D(
             global_submap_poses.at(last_optimized_submap_id).global_pose) *
         data_.submap_data.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  const auto it = data_.submap_data.find(submap_id);
  if (it == data_.submap_data.end()) {
    return {};
  }
  auto submap = it->data.submap;
  if (data_.global_submap_poses_2d.Contains(submap_id)) {
    // We already have an optimized pose.
    return {submap,
            transform::Embed3D(
                data_.global_submap_poses_2d.at(submap_id).global_pose)};
  }
  // We have to extrapolate.
  return {submap, ComputeLocalToGlobalTransform(data_.global_submap_poses_2d,
                                                submap_id.trajectory_id) *
                      submap->local_pose()};
}

PoseGraph2D::TrimmingHandle::TrimmingHandle(PoseGraph2D* const parent)
    : parent_(parent) {}

int PoseGraph2D::TrimmingHandle::num_submaps(const int trajectory_id) const {
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  return submap_data.SizeOfTrajectoryOrZero(trajectory_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::TrimmingHandle::GetOptimizedSubmapData() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : parent_->data_.submap_data) {
    if (submap_id_data.data.state != SubmapState::kFinished ||
        !parent_->data_.global_submap_poses_2d.Contains(submap_id_data.id)) {
      continue;
    }
    submaps.Insert(
        submap_id_data.id,
        SubmapData{submap_id_data.data.submap,
                   transform::Embed3D(parent_->data_.global_submap_poses_2d
                                          .at(submap_id_data.id)
                                          .global_pose)});
  }
  return submaps;
}

std::vector<SubmapId> PoseGraph2D::TrimmingHandle::GetSubmapIds(
    int trajectory_id) const {
  std::vector<SubmapId> submap_ids;
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  for (const auto& it : submap_data.trajectory(trajectory_id)) {
    submap_ids.push_back(it.id);
  }
  return submap_ids;
}

const MapById<NodeId, TrajectoryNode>&
PoseGraph2D::TrimmingHandle::GetTrajectoryNodes() const {
  return parent_->data_.trajectory_nodes;
}

const std::vector<PoseGraphInterface::Constraint>&
PoseGraph2D::TrimmingHandle::GetConstraints() const {
  return parent_->data_.constraints;
}

bool PoseGraph2D::TrimmingHandle::IsFinished(const int trajectory_id) const {
  return parent_->IsTrajectoryFinished(trajectory_id);
}

void PoseGraph2D::TrimmingHandle::SetTrajectoryState(int trajectory_id,
                                                     TrajectoryState state) {
  parent_->data_.trajectories_state[trajectory_id].state = state;
}

void PoseGraph2D::TrimmingHandle::TrimSubmap(const SubmapId& submap_id) {
  // TODO(hrapp): We have to make sure that the trajectory has been finished
  // if we want to delete the last submaps.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);

  // Compile all nodes that are still INTRA_SUBMAP constrained once the submap
  // with 'submap_id' is gone.
  std::set<NodeId> nodes_to_retain;
  for (const Constraint& constraint : parent_->data_.constraints) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
        constraint.submap_id != submap_id) {
      nodes_to_retain.insert(constraint.node_id);
    }
  }
  // Remove all 'data_.constraints' related to 'submap_id'.
  std::set<NodeId> nodes_to_remove;
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.submap_id == submap_id) {
        if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
            nodes_to_retain.count(constraint.node_id) == 0) {
          // This node will no longer be INTRA_SUBMAP contrained and has to be
          // removed.
          nodes_to_remove.insert(constraint.node_id);
        }
      } else {
        constraints.push_back(constraint);
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }
  // Remove all 'data_.constraints' related to 'nodes_to_remove'.
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (nodes_to_remove.count(constraint.node_id) == 0) {
        constraints.push_back(constraint);
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }

  // Mark the submap with 'submap_id' as trimmed and remove its data.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);
  parent_->data_.submap_data.Trim(submap_id);
  parent_->constraint_builder_.DeleteScanMatcher(submap_id);
  parent_->optimization_problem_->TrimSubmap(submap_id);

  // We have one submap less, update the gauge metrics.
  kDeletedSubmapsMetric->Increment();
  if (parent_->IsTrajectoryFrozen(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Decrement();
  } else {
    kActiveSubmapsMetric->Decrement();
  }

  // Remove the 'nodes_to_remove' from the pose graph and the optimization
  // problem.
  for (const NodeId& node_id : nodes_to_remove) {
    parent_->data_.trajectory_nodes.Trim(node_id);
    parent_->optimization_problem_->TrimTrajectoryNode(node_id);
  }
}

// 获取在锁下的submapid 和submap数据
MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : data_.submap_data) {
    submaps.Insert(submap_id_data.id,
                   GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

// 设置全局优化回调函数
void PoseGraph2D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  global_slam_optimization_callback_ = callback;
}

void PoseGraph2D::RegisterMetrics(metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_2d_pose_graph_work_queue_delay",
      "Age of the oldest entry in the work queue in seconds");
  kWorkQueueDelayMetric = latency->Add({});
  auto* queue_size =
      family_factory->NewGaugeFamily("mapping_2d_pose_graph_work_queue_size",
                                     "Number of items in the work queue");
  kWorkQueueSizeMetric = queue_size->Add({});
  auto* constraints = family_factory->NewGaugeFamily(
      "mapping_2d_pose_graph_constraints",
      "Current number of constraints in the pose graph");
  kConstraintsDifferentTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "different"}});
  kConstraintsSameTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "same"}});
  auto* submaps = family_factory->NewGaugeFamily(
      "mapping_2d_pose_graph_submaps", "Number of submaps in the pose graph.");
  kActiveSubmapsMetric = submaps->Add({{"state", "active"}});
  kFrozenSubmapsMetric = submaps->Add({{"state", "frozen"}});
  kDeletedSubmapsMetric = submaps->Add({{"state", "deleted"}});
}

}  // namespace mapping
}  // namespace cartographer
