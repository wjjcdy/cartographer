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

#include "cartographer/mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>

#include "Eigen/Geometry"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

// A collection of values which can be added and later removed, and the maximum
// of the current values in the collection can be retrieved 恢复.
// All of it in (amortized) O(1).
class SlidingWindowMaximum {
 public:
  // 相当于一个队列，且按从大到小顺序存储，如果队列比当前值小的全部剔除
  void AddValue(const float value) {
    while (!non_ascending_maxima_.empty() &&
           value > non_ascending_maxima_.back()) {
      non_ascending_maxima_.pop_back();
    }
    non_ascending_maxima_.push_back(value);
  }

  // 如果当前值在队列头，则剔除
  void RemoveValue(const float value) {
    // DCHECK for performance, since this is done for every value in the
    // precomputation grid.
    DCHECK(!non_ascending_maxima_.empty());
    DCHECK_LE(value, non_ascending_maxima_.front());
    if (value == non_ascending_maxima_.front()) {
      non_ascending_maxima_.pop_front();
    }
  }

  // 显然队列头为最大值
  float GetMaximum() const {
    // DCHECK for performance, since this is done for every value in the
    // precomputation grid.
    DCHECK_GT(non_ascending_maxima_.size(), 0);
    return non_ascending_maxima_.front();
  }

  void CheckIsEmpty() const { CHECK_EQ(non_ascending_maxima_.size(), 0); }

 private:
  // Maximum of the current sliding window at the front. Then the maximum of the
  // remaining window that came after this values first occurrence, and so on.
  // 滑动的窗口，是个队列
  std::deque<float> non_ascending_maxima_;
};

}  // namespace

proto::FastCorrelativeScanMatcherOptions2D
CreateFastCorrelativeScanMatcherOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::FastCorrelativeScanMatcherOptions2D options;
  options.set_linear_search_window(
      parameter_dictionary->GetDouble("linear_search_window"));
  options.set_angular_search_window(
      parameter_dictionary->GetDouble("angular_search_window"));
  options.set_branch_and_bound_depth(
      parameter_dictionary->GetInt("branch_and_bound_depth"));
  return options;
}

// 预处理的栅格地图构造函数
// 大小在x和y方向均放大width个
// 原地图起点的偏移则平移width
// 其预处理是将原图分辨率按照层次变低， 低分辨率的地图，一位置相当于原图高分率那块位置的中心值，但并不是最大值，将其赋值成最大值
// 如此每层地图index位置存储的都是上层层高分辨最大边界值
PrecomputationGrid2D::PrecomputationGrid2D(
    const Grid2D& grid, const CellLimits& limits, const int width,
    std::vector<float>* reusable_intermediate_grid)
    : offset_(-width + 1, -width + 1),
      wide_limits_(limits.num_x_cells + width - 1,
                   limits.num_y_cells + width - 1),
      min_score_(1.f - grid.GetMaxCorrespondenceCost()),
      max_score_(1.f - grid.GetMinCorrespondenceCost()),
      cells_(wide_limits_.num_x_cells * wide_limits_.num_y_cells) {
  CHECK_GE(width, 1);
  CHECK_GE(limits.num_x_cells, 1);
  CHECK_GE(limits.num_y_cells, 1);
  const int stride = wide_limits_.num_x_cells;
  // First we compute the maximum probability for each (x0, y) achieved in the
  // span defined by x0 <= x < x0 + width.
  std::vector<float>& intermediate = *reusable_intermediate_grid;
  // 重定义大小，宽度放大，高度不放大
  intermediate.resize(wide_limits_.num_x_cells * limits.num_y_cells);
  for (int y = 0; y != limits.num_y_cells; ++y) {
    SlidingWindowMaximum current_values;
    // 加入每一行第一个概率
    current_values.AddValue(
        1.f - std::abs(grid.GetCorrespondenceCost(Eigen::Array2i(0, y))));

    // 方法是将每width的数据取其最大值并保存
    //扩展外前width个数据的处理， 
    for (int x = -width + 1; x != 0; ++x) {
      //将最大的概率值赋值
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      // 
      if (x + width < limits.num_x_cells) {
        current_values.AddValue(1.f - std::abs(grid.GetCorrespondenceCost(
                                          Eigen::Array2i(x + width, y))));
      }
    }

    // 扩展内原数据处理
    for (int x = 0; x < limits.num_x_cells - width; ++x) {
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      current_values.RemoveValue(
          1.f - std::abs(grid.GetCorrespondenceCost(Eigen::Array2i(x, y))));
      current_values.AddValue(1.f - std::abs(grid.GetCorrespondenceCost(
                                        Eigen::Array2i(x + width, y))));
    }
    // 最后width数据
    for (int x = std::max(limits.num_x_cells - width, 0);
         x != limits.num_x_cells; ++x) {
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      current_values.RemoveValue(
          1.f - std::abs(grid.GetCorrespondenceCost(Eigen::Array2i(x, y))));
    }
    // 判读是否为空，理论上不应该为空
    current_values.CheckIsEmpty();
  }
  // For each (x, y), we compute the maximum probability in the width x width
  // region starting at each (x, y) and precompute the resulting bound on the
  // score.
  for (int x = 0; x != wide_limits_.num_x_cells; ++x) {
    SlidingWindowMaximum current_values;
    current_values.AddValue(intermediate[x]);
    for (int y = -width + 1; y != 0; ++y) {
      cells_[x + (y + width - 1) * stride] =
          ComputeCellValue(current_values.GetMaximum());
      if (y + width < limits.num_y_cells) {
        current_values.AddValue(intermediate[x + (y + width) * stride]);
      }
    }
    for (int y = 0; y < limits.num_y_cells - width; ++y) {
      cells_[x + (y + width - 1) * stride] =
          ComputeCellValue(current_values.GetMaximum());
      current_values.RemoveValue(intermediate[x + y * stride]);
      current_values.AddValue(intermediate[x + (y + width) * stride]);
    }
    for (int y = std::max(limits.num_y_cells - width, 0);
         y != limits.num_y_cells; ++y) {
      cells_[x + (y + width - 1) * stride] =
          ComputeCellValue(current_values.GetMaximum());
      current_values.RemoveValue(intermediate[x + y * stride]);
    }
    current_values.CheckIsEmpty();
  }
}

//将概率换算成0~255之间整数表示
uint8 PrecomputationGrid2D::ComputeCellValue(const float probability) const {
  const int cell_value = common::RoundToInt(
      (probability - min_score_) * (255.f / (max_score_ - min_score_)));
  CHECK_GE(cell_value, 0);
  CHECK_LE(cell_value, 255);
  return cell_value;
}

// 预处理grid地图堆栈构造函数
// 相当于一个堆栈，其堆栈了存储同一个地图但分辨率不同，低分辨率地图value，采用对应高分辨地图中子格中最高分辨率
PrecomputationGridStack2D::PrecomputationGridStack2D(
    const Grid2D& grid,
    const proto::FastCorrelativeScanMatcherOptions2D& options) {
  CHECK_GE(options.branch_and_bound_depth(), 1);
  // 获取分支边界搜索层参数， 获取grid地图放大的最大宽度
  const int max_width = 1 << (options.branch_and_bound_depth() - 1);
  // precomputation_grids_ 根据参数开辟搜索层数
  precomputation_grids_.reserve(options.branch_and_bound_depth());
  std::vector<float> reusable_intermediate_grid;
  // 赋值原来grid limit参数
  const CellLimits limits = grid.limits().cell_limits();
  // 开辟一个vector，其大小为，应该是每层存储的的grid，空间开辟意义不大，每层都会再次resize
  reusable_intermediate_grid.reserve((limits.num_x_cells + max_width - 1) *
                                     limits.num_y_cells);
  // 构建
  for (int i = 0; i != options.branch_and_bound_depth(); ++i) {
    //后续因为需要用来采样的为1,2,4,8,16......
    //队列中最前的为分辨率最高的地图
    //队列末尾则为分辨率最低的地图
    //故需对原图片进行采样，保证第一个采样位置不变，需要对原图进行扩展，而width则扩展和偏移量
    //层顶采样间隔最小，即为最高分辨率地图
    const int width = 1 << i;
    precomputation_grids_.emplace_back(grid, limits, width,
                                       &reusable_intermediate_grid);
  }
}


//构造函数
// input: 栅格图， 配置参数
// 栅格地图进行预先处理
FastCorrelativeScanMatcher2D::FastCorrelativeScanMatcher2D(
    const Grid2D& grid,
    const proto::FastCorrelativeScanMatcherOptions2D& options)
    : options_(options),
      limits_(grid.limits()),
      precomputation_grid_stack_(
          absl::make_unique<PrecomputationGridStack2D>(grid, options)) {}

FastCorrelativeScanMatcher2D::~FastCorrelativeScanMatcher2D() {}

// 匹配函数
/*
input:
当前帧估计位置（里程计等提供的初始位置）
当前帧点云（即以激光雷达为坐标系的点云）
最小置信度
（grid在构造函数已经传递）

output：
置信度清单
匹配后输出位置
 */
bool FastCorrelativeScanMatcher2D::Match(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud, const float min_score, float* score,
    transform::Rigid2d* pose_estimate) const {
  //根据配置窗口大小和点云距离范围，栅格分辨率获取量化遍历空间
  const SearchParameters search_parameters(options_.linear_search_window(),
                                           options_.angular_search_window(),
                                           point_cloud, limits_.resolution());

  // 根据遍历参数进行匹配
  return MatchWithSearchParameters(search_parameters, initial_pose_estimate,
                                   point_cloud, min_score, score,
                                   pose_estimate);
}


// 此匹配为全范围暴力匹配，而对应上一个函数，则是对预测位置附近的窗口内进行匹配
bool FastCorrelativeScanMatcher2D::MatchFullSubmap(
    const sensor::PointCloud& point_cloud, float min_score, float* score,
    transform::Rigid2d* pose_estimate) const {
  // Compute a search window around the center of the submap that includes it
  // fully.
  // 在一个1e6大小遍历平移窗口， 360度内遍历角度，即全遍历
  const SearchParameters search_parameters(
      1e6 * limits_.resolution(),  // Linear search window, 1e6 cells/direction.
      M_PI,  // Angular search window, 180 degrees in both directions.
      point_cloud, limits_.resolution());
  // 遍历的起始位置为地图的中心位置
  const transform::Rigid2d center = transform::Rigid2d::Translation(
      limits_.max() - 0.5 * limits_.resolution() *
                          Eigen::Vector2d(limits_.cell_limits().num_y_cells,
                                          limits_.cell_limits().num_x_cells));
  // 根据遍历参数进行匹配
  return MatchWithSearchParameters(search_parameters, center, point_cloud,
                                   min_score, score, pose_estimate);
}

bool FastCorrelativeScanMatcher2D::MatchWithSearchParameters(
    SearchParameters search_parameters,
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud, float min_score, float* score,
    transform::Rigid2d* pose_estimate) const {
  CHECK(score != nullptr);
  CHECK(pose_estimate != nullptr);

  const Eigen::Rotation2Dd initial_rotation = initial_pose_estimate.rotation();
  // 将点云旋转至初始位置（即估计位置）航向方向上
  const sensor::PointCloud rotated_point_cloud = sensor::TransformPointCloud(
      point_cloud,
      transform::Rigid3f::Rotation(Eigen::AngleAxisf(
          initial_rotation.cast<float>().angle(), Eigen::Vector3f::UnitZ())));
  // 根据将角度窗口按照一定分辨率划分，并根据每一个旋转角度将点云旋转，生成N个点云
  const std::vector<sensor::PointCloud> rotated_scans =
      GenerateRotatedScans(rotated_point_cloud, search_parameters);
  
  // 将所有点云转换到初始位置上
  const std::vector<DiscreteScan2D> discrete_scans = DiscretizeScans(
      limits_, rotated_scans,
      Eigen::Translation2f(initial_pose_estimate.translation().x(),
                           initial_pose_estimate.translation().y()));

  // 修复下所有点云的大小在空间的大小，即边界
  search_parameters.ShrinkToFit(discrete_scans, limits_.cell_limits());

  //获取低分辨率的量化列表（和标准相关方法对比），并且计算匹配评分结果，并进行了排序
  const std::vector<Candidate2D> lowest_resolution_candidates =
      ComputeLowestResolutionCandidates(discrete_scans, search_parameters);

  // 分支边界搜索最佳匹配
  const Candidate2D best_candidate = BranchAndBound(
      discrete_scans, search_parameters, lowest_resolution_candidates,
      precomputation_grid_stack_->max_depth(), min_score);
  if (best_candidate.score > min_score) {
    *score = best_candidate.score;
    *pose_estimate = transform::Rigid2d(
        {initial_pose_estimate.translation().x() + best_candidate.x,
         initial_pose_estimate.translation().y() + best_candidate.y},
        initial_rotation * Eigen::Rotation2Dd(best_candidate.orientation));
    return true;
  }
  return false;
}

//获取低分辨率的量化列表
std::vector<Candidate2D>
FastCorrelativeScanMatcher2D::ComputeLowestResolutionCandidates(
    const std::vector<DiscreteScan2D>& discrete_scans,
    const SearchParameters& search_parameters) const {
  // 获取低分辨率的量化列表
  std::vector<Candidate2D> lowest_resolution_candidates =
      GenerateLowestResolutionCandidates(search_parameters);

  // 标准相关匹配，并返回匹配结果，其置信度按照从大到小排序
  // 显然此处只是对低分辨率进行了相关匹配，低分辨率具体看共几层。
  ScoreCandidates(
      precomputation_grid_stack_->Get(precomputation_grid_stack_->max_depth()),
      discrete_scans, search_parameters, &lowest_resolution_candidates);
  // 返回从大到小排序后的最低分辨率解
  return lowest_resolution_candidates;
}

//生成低分辨率的量化列表
std::vector<Candidate2D>
FastCorrelativeScanMatcher2D::GenerateLowestResolutionCandidates(
    const SearchParameters& search_parameters) const {
  // 应该是根据搜索的层数，作为分辨率，然后逐步变细腻。假设4层，则分辨率为1<<4,则每隔16个点进行采样
  // 即最底层的地图则为最低分辨率参数
  const int linear_step_size = 1 << precomputation_grid_stack_->max_depth();
  //获取X，Y量化个数， 采样间隔为linear_step_size， 其中scan_index表示每个旋转角度对应的点云
  // 只是在表示每个旋转量对应的参数是一样的，这里却对每一个进行遍历，浪费大量时间？？？？？？？
  int num_candidates = 0;
  for (int scan_index = 0; scan_index != search_parameters.num_scans;
       ++scan_index) {
    const int num_lowest_resolution_linear_x_candidates =
        (search_parameters.linear_bounds[scan_index].max_x -
         search_parameters.linear_bounds[scan_index].min_x + linear_step_size) /
        linear_step_size;
    const int num_lowest_resolution_linear_y_candidates =
        (search_parameters.linear_bounds[scan_index].max_y -
         search_parameters.linear_bounds[scan_index].min_y + linear_step_size) /
        linear_step_size;
    num_candidates += num_lowest_resolution_linear_x_candidates *
                      num_lowest_resolution_linear_y_candidates;
  }
  std::vector<Candidate2D> candidates;
  // 最低分辨率采样所有可能性
  candidates.reserve(num_candidates);
  for (int scan_index = 0; scan_index != search_parameters.num_scans;
       ++scan_index) {
    for (int x_index_offset = search_parameters.linear_bounds[scan_index].min_x;
         x_index_offset <= search_parameters.linear_bounds[scan_index].max_x;
         x_index_offset += linear_step_size) {
      for (int y_index_offset =
               search_parameters.linear_bounds[scan_index].min_y;
           y_index_offset <= search_parameters.linear_bounds[scan_index].max_y;
           y_index_offset += linear_step_size) {
        candidates.emplace_back(scan_index, x_index_offset, y_index_offset,
                                search_parameters);
      }
    }
  }
  CHECK_EQ(candidates.size(), num_candidates);
  return candidates;
}

// 计算置信度
// 统计一帧点云概率和并归一化，获取置信度评分并排序
void FastCorrelativeScanMatcher2D::ScoreCandidates(
    const PrecomputationGrid2D& precomputation_grid,
    const std::vector<DiscreteScan2D>& discrete_scans,
    const SearchParameters& search_parameters,
    std::vector<Candidate2D>* const candidates) const {
  for (Candidate2D& candidate : *candidates) {
    int sum = 0;
    for (const Eigen::Array2i& xy_index :
         discrete_scans[candidate.scan_index]) {
      const Eigen::Array2i proposed_xy_index(
          xy_index.x() + candidate.x_index_offset,
          xy_index.y() + candidate.y_index_offset);
      sum += precomputation_grid.GetValue(proposed_xy_index);
    }
    candidate.score = precomputation_grid.ToScore(
        sum / static_cast<float>(discrete_scans[candidate.scan_index].size()));
  }
  // 按照评分大小排序，从大到小排序（ std::greater 表示改变排序方式）
  std::sort(candidates->begin(), candidates->end(),
            std::greater<Candidate2D>());
}


//分支边界
/*
input:
不同角度下的点云
搜索参数
所有可能的位置点
搜索层数
满足最小的评分数，即分支的下边界
 */
/*
分支边界思想：
1.地图分辨率，顶层为最低分辨率，底层为原分辨率
2.采用最底层分辨率进行暴力遍历，将可能位置匹配评分由高到底排序
3.评分小于分支下边界的，直接裁剪，并给下边界赋初值best_high_resolution_candidate.score
4.从最高的评分的顶层开始分析，并计算下层直到获取最后一层结果，即为真实匹配结果，并保留最大score，并更新下边界best_high_resolution_candidate.score
5.每一层均与下边界对比，若此层节点score小于下边界，则抛弃此节点及其所有子节点
6.以此类推获取最佳score和pose
 */
Candidate2D FastCorrelativeScanMatcher2D::BranchAndBound(
    const std::vector<DiscreteScan2D>& discrete_scans,
    const SearchParameters& search_parameters,
    const std::vector<Candidate2D>& candidates, const int candidate_depth,
    float min_score) const {
  // 如果没分层，则直接返回评分最高的结果，即到达元分辨率层
  if (candidate_depth == 0) {
    // Return the best candidate.， 已经拍过序，故第一个则为最佳匹配
    return *candidates.begin();
  }

  Candidate2D best_high_resolution_candidate(0, 0, 0, search_parameters);
  best_high_resolution_candidate.score = min_score;
  for (const Candidate2D& candidate : candidates) {
    // 小于分支下边界，可直接结束，即裁剪此枝叶，因为顶层已经按评分结果从大小排序，后面只能更小
    if (candidate.score <= min_score) {
      break;
    }
    std::vector<Candidate2D> higher_resolution_candidates;
    // 由于地图分辨率为2的层数次方, 因此下一层高分辨为2的层数-1 次方
    // 获取此层下一层的间隔
    const int half_width = 1 << (candidate_depth - 1);
    for (int x_offset : {0, half_width}) {
      // x 到达遍历边界
      if (candidate.x_index_offset + x_offset >
          search_parameters.linear_bounds[candidate.scan_index].max_x) {
        break;
      }
      for (int y_offset : {0, half_width}) {
        // y到达遍历边界
        if (candidate.y_index_offset + y_offset >
            search_parameters.linear_bounds[candidate.scan_index].max_y) {
          break;
        }
        // 将此层的下一层更高分辨的坐标列表
        higher_resolution_candidates.emplace_back(
            candidate.scan_index, candidate.x_index_offset + x_offset,
            candidate.y_index_offset + y_offset, search_parameters);
      }
    }
    // 计算更高层的评分
    ScoreCandidates(precomputation_grid_stack_->Get(candidate_depth - 1),
                    discrete_scans, search_parameters,
                    &higher_resolution_candidates);
    // 取最高评分的的pose集合，并且更高评分的结果列表，继续分支，直到子节点，即原分辨率地图
    best_high_resolution_candidate = std::max(
        best_high_resolution_candidate,
        BranchAndBound(discrete_scans, search_parameters,
                       higher_resolution_candidates, candidate_depth - 1,
                       best_high_resolution_candidate.score));
  }
  return best_high_resolution_candidate;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
