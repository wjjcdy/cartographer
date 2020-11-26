/*
 * Copyright 2018 The Cartographer Authors
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

#ifndef CARTOGRAPHER_MAPPING_POSE_GRAPH_DATA_H_
#define CARTOGRAPHER_MAPPING_POSE_GRAPH_DATA_H_

#include <map>
#include <set>
#include <vector>

#include "cartographer/mapping/internal/optimization/optimization_problem_2d.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"
#include "cartographer/mapping/internal/trajectory_connectivity_state.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/submaps.h"

namespace cartographer {
namespace mapping {

// The current state of the submap in the background threads. After this
// transitions to 'kFinished', all nodes are tried to match
// against this submap. Likewise, all new nodes are matched against submaps in
// that state.
// submap 当前在后台的状态，如果状态为kFinshed，即此submap完成不再更新, 则需要所有节点都与这个submap进行匹配
enum class SubmapState { kNoConstraintSearch, kFinished };

struct InternalTrajectoryState {
  enum class DeletionState {
    NORMAL,
    SCHEDULED_FOR_DELETION,
    WAIT_FOR_DELETION
  };

  PoseGraphInterface::TrajectoryState state =
      PoseGraphInterface::TrajectoryState::ACTIVE;
  DeletionState deletion_state = DeletionState::NORMAL;
};

// graph 中存储的submap格式
struct InternalSubmapData {
  std::shared_ptr<const Submap> submap;
  SubmapState state = SubmapState::kNoConstraintSearch;

  // IDs of the nodes that were inserted into this map together with
  // constraints for them. They are not to be matched again when this submap
  // becomes 'kFinished'.
  // 插入该地图的scan ID集合，在此submap称为kFinished时无需进行match
  std::set<NodeId> node_ids;
};

struct PoseGraphData {
  // Submaps get assigned an ID and state as soon as they are seen, even
  // before they take part in the background computations.
  MapById<SubmapId, InternalSubmapData> submap_data;                             // submap 对应的submapID集合

  // Global submap poses currently used for displaying data.
  MapById<SubmapId, optimization::SubmapSpec2D> global_submap_poses_2d;          // submap的全局位置
  MapById<SubmapId, optimization::SubmapSpec3D> global_submap_poses_3d;

  // Data that are currently being shown.
  MapById<NodeId, TrajectoryNode> trajectory_nodes;                              // scanID 与trajectory对应集合

  // Global landmark poses with all observations.
  std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
      landmark_nodes;

  // How our various trajectories are related.
  TrajectoryConnectivityState trajectory_connectivity_state;                    // 轨迹之间的状态
  int num_trajectory_nodes = 0;
  std::map<int, InternalTrajectoryState> trajectories_state;                    // 轨迹状态集合

  // Set of all initial trajectory poses.
  std::map<int, PoseGraph::InitialTrajectoryPose> initial_trajectory_poses;     // 每个trajectory的初始位置

  std::vector<PoseGraphInterface::Constraint> constraints;                      // 约束队列
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_POSE_GRAPH_DATA_H_
