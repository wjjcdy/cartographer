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

#include "cartographer/mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "Eigen/Geometry"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/tsdf_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

float ComputeCandidateScore(const TSDF2D& tsdf,
                            const DiscreteScan2D& discrete_scan,
                            int x_index_offset, int y_index_offset) {
  float candidate_score = 0.f;
  float summed_weight = 0.f;
  for (const Eigen::Array2i& xy_index : discrete_scan) {
    const Eigen::Array2i proposed_xy_index(xy_index.x() + x_index_offset,
                                           xy_index.y() + y_index_offset);
    const std::pair<float, float> tsd_and_weight =
        tsdf.GetTSDAndWeight(proposed_xy_index);
    const float normalized_tsd_score =
        (tsdf.GetMaxCorrespondenceCost() - std::abs(tsd_and_weight.first)) /
        tsdf.GetMaxCorrespondenceCost();
    const float weight = tsd_and_weight.second;
    candidate_score += normalized_tsd_score * weight;
    summed_weight += weight;
  }
  if (summed_weight == 0.f) return 0.f;
  candidate_score /= summed_weight;
  CHECK_GE(candidate_score, 0.f);
  return candidate_score;
}

float ComputeCandidateScore(const ProbabilityGrid& probability_grid,
                            const DiscreteScan2D& discrete_scan,
                            int x_index_offset, int y_index_offset) {
  float candidate_score = 0.f;
  for (const Eigen::Array2i& xy_index : discrete_scan) {
    const Eigen::Array2i proposed_xy_index(xy_index.x() + x_index_offset,
                                           xy_index.y() + y_index_offset);
    const float probability =
        probability_grid.GetProbability(proposed_xy_index);
    candidate_score += probability;
  }
  candidate_score /= static_cast<float>(discrete_scan.size());
  CHECK_GT(candidate_score, 0.f);
  return candidate_score;
}

}  // namespace

RealTimeCorrelativeScanMatcher2D::RealTimeCorrelativeScanMatcher2D(
    const proto::RealTimeCorrelativeScanMatcherOptions& options)
    : options_(options) {}

// 生成平移空间下的平移量，用于相关匹配时的搜索索引
std::vector<Candidate2D>
RealTimeCorrelativeScanMatcher2D::GenerateExhaustiveSearchCandidates(
    const SearchParameters& search_parameters) const {
  int num_candidates = 0;
  //根据角度搜索空间点云个数search_parameters.num_scans
  for (int scan_index = 0; scan_index != search_parameters.num_scans;
       ++scan_index) {
    // 一个keypose的x左右边界，统计个数
    const int num_linear_x_candidates =
        (search_parameters.linear_bounds[scan_index].max_x -
         search_parameters.linear_bounds[scan_index].min_x + 1);
    const int num_linear_y_candidates =
        (search_parameters.linear_bounds[scan_index].max_y -
         search_parameters.linear_bounds[scan_index].min_y + 1);
    // 获取x，y空间边界总个数
    num_candidates += num_linear_x_candidates * num_linear_y_candidates;
  }
  std::vector<Candidate2D> candidates;
  candidates.reserve(num_candidates);
  // 外循环为角度空间的个数
  for (int scan_index = 0; scan_index != search_parameters.num_scans;
       ++scan_index) {
    // 内两层空间即x和y的平移量空间
    for (int x_index_offset = search_parameters.linear_bounds[scan_index].min_x;
         x_index_offset <= search_parameters.linear_bounds[scan_index].max_x;
         ++x_index_offset) {
      for (int y_index_offset =
               search_parameters.linear_bounds[scan_index].min_y;
           y_index_offset <= search_parameters.linear_bounds[scan_index].max_y;
           ++y_index_offset) {
        candidates.emplace_back(scan_index, x_index_offset, y_index_offset,
                                search_parameters);
      }
    }
  }
  CHECK_EQ(candidates.size(), num_candidates);
  return candidates;
}

// 相关性扫描匹配
/*
input:
当前帧估计位置（里程计等提供的初始位置）
当前帧点云（即以激光雷达为坐标系的点云）
栅格地图
匹配后输出位置
 */
double RealTimeCorrelativeScanMatcher2D::Match(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud, const Grid2D& grid,
    transform::Rigid2d* pose_estimate) const {
  CHECK(pose_estimate != nullptr);
  //获取初始位置的角度
  const Eigen::Rotation2Dd initial_rotation = initial_pose_estimate.rotation();
  //将当前点云转换为初始角度坐标系下点云
  const sensor::PointCloud rotated_point_cloud = sensor::TransformPointCloud(
      point_cloud,
      transform::Rigid3f::Rotation(Eigen::AngleAxisf(
          initial_rotation.cast<float>().angle(), Eigen::Vector3f::UnitZ())));
  // 定义搜索的空间，包括平移的窗口和角度窗口
  // options_ 为配置参数，由上层调用传入
  const SearchParameters search_parameters(
      options_.linear_search_window(), options_.angular_search_window(),
      rotated_point_cloud, grid.limits().resolution());
  // 在一定角度搜索空间中（一定窗口内，一定分辨率，如1度分辨率， 搜索范围100度，共100个角度空间），
  // 对当前点云进行旋转变换，放入vector中
  const std::vector<sensor::PointCloud> rotated_scans =
      GenerateRotatedScans(rotated_point_cloud, search_parameters);
  // 将点云进行坐标转换到世界坐标系坐标，即根据当前估计位置（可以为里程计的位置）进行转换
  const std::vector<DiscreteScan2D> discrete_scans = DiscretizeScans(
      grid.limits(), rotated_scans,
      Eigen::Translation2f(initial_pose_estimate.translation().x(),
                           initial_pose_estimate.translation().y()));

  // 初始化在xy平移空间内偏移量，即窗口大小和偏移量
  std::vector<Candidate2D> candidates =
      GenerateExhaustiveSearchCandidates(search_parameters);

  // 计算当前帧在角度、x、y三层窗口中每个状态在grid地图中的匹配的评分
  ScoreCandidates(grid, discrete_scans, search_parameters, &candidates);

  const Candidate2D& best_candidate =
      *std::max_element(candidates.begin(), candidates.end());
  *pose_estimate = transform::Rigid2d(
      {initial_pose_estimate.translation().x() + best_candidate.x,
       initial_pose_estimate.translation().y() + best_candidate.y},
      initial_rotation * Eigen::Rotation2Dd(best_candidate.orientation));
  return best_candidate.score;
}

// 匹配函数，将3层搜索空间每一个位姿进行匹配，并计算置信度
void RealTimeCorrelativeScanMatcher2D::ScoreCandidates(
    const Grid2D& grid, const std::vector<DiscreteScan2D>& discrete_scans,
    const SearchParameters& search_parameters,
    std::vector<Candidate2D>* const candidates) const {
  for (Candidate2D& candidate : *candidates) {
    //栅格地图有两种形式
    switch (grid.GetGridType()) {
      case GridType::PROBABILITY_GRID:
        candidate.score = ComputeCandidateScore(
            static_cast<const ProbabilityGrid&>(grid),
            discrete_scans[candidate.scan_index], candidate.x_index_offset,
            candidate.y_index_offset);
        break;
      case GridType::TSDF:
        candidate.score = ComputeCandidateScore(
            static_cast<const TSDF2D&>(grid),
            discrete_scans[candidate.scan_index], candidate.x_index_offset,
            candidate.y_index_offset);
        break;
    }
    candidate.score *=
        std::exp(-common::Pow2(std::hypot(candidate.x, candidate.y) *
                                   options_.translation_delta_cost_weight() +
                               std::abs(candidate.orientation) *
                                   options_.rotation_delta_cost_weight()));
  }
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
