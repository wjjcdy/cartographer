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

#include "cartographer/mapping/internal/2d/scan_matching/occupied_space_cost_function_2d.h"

#include "cartographer/mapping/probability_values.h"
#include "ceres/cubic_interpolation.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

// Computes a cost for matching the 'point_cloud' to the 'grid' with
// a 'pose'. The cost increases with poorer correspondence of the grid and the
// point observation (e.g. points falling into less occupied space).
// 构造代价函数结构体
class OccupiedSpaceCostFunction2D {
 public:
  // 输入：权重， 点云， 栅格地图
  OccupiedSpaceCostFunction2D(const double scaling_factor,
                              const sensor::PointCloud& point_cloud,
                              const Grid2D& grid)
      : scaling_factor_(scaling_factor),
        point_cloud_(point_cloud),
        grid_(grid) {}
  
  // 类型模板
  template <typename T>
  // pose为输入待优化量， residual为参差
  bool operator()(const T* const pose, T* residual) const {
    // 平移矩阵
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    // 旋转向量
    Eigen::Rotation2D<T> rotation(pose[2]);
    // 旋转矩阵
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    // 2维转移矩阵， 即当前位置在世界坐标系下的转移矩阵
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    // 重新定义grid
    const GridArrayAdapter adapter(grid_);
    // 这里将构造时传入的概率栅格图（local submap）加载到一个双三次插值器中
    // Grid2D还可以利用BiCubicInterpolator实现双三次插值，它相对于双线性插值的优点是能实现自动求导
    ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);
    const MapLimits& limits = grid_.limits();

    for (size_t i = 0; i < point_cloud_.size(); ++i) {
      // Note that this is a 2D point. The third component is a scaling factor.
      const Eigen::Matrix<T, 3, 1> point((T(point_cloud_[i].position.x())),
                                         (T(point_cloud_[i].position.y())),
                                         T(1.));
      // 将点云转换为世界坐标
      const Eigen::Matrix<T, 3, 1> world = transform * point;
      // 迭代评价函数
      // 将坐标转换为栅格坐标，双三次插值器自动计算中对应坐标的value
      interpolator.Evaluate(
          (limits.max().x() - world[0]) / limits.resolution() - 0.5 +
              static_cast<double>(kPadding),
          (limits.max().y() - world[1]) / limits.resolution() - 0.5 +
              static_cast<double>(kPadding),
          &residual[i]);
      // 所有参差加入同一权重
      residual[i] = scaling_factor_ * residual[i];
    }
    return true;
  }

 private:
  // ????????????, 整数中最大值，猜测应该是限制最大搜索范围
  static constexpr int kPadding = INT_MAX / 4;
  // 栅格地图矩阵转换,或者认为是复制
  class GridArrayAdapter {
   public:
    enum { DATA_DIMENSION = 1 };

    // 构造函数
    explicit GridArrayAdapter(const Grid2D& grid) : grid_(grid) {}
    // 不在栅格地图范围内，则返回最大相关代价值
    void GetValue(const int row, const int column, double* const value) const {
      if (row < kPadding || column < kPadding || row >= NumRows() - kPadding ||
          column >= NumCols() - kPadding) {
        *value = kMaxCorrespondenceCost;
      } else {
        // 由于求取最小二乘，因此要取概率值的对应代价=（1 - probality）
        *value = static_cast<double>(grid_.GetCorrespondenceCost(
            Eigen::Array2i(column - kPadding, row - kPadding)));
      }
    }
    // 重新定义栅格地图的大小，即增加偏移
    int NumRows() const {
      return grid_.limits().cell_limits().num_y_cells + 2 * kPadding;
    }

    int NumCols() const {
      return grid_.limits().cell_limits().num_x_cells + 2 * kPadding;
    }

   private:
    const Grid2D& grid_;
  };

  // ?????????????????????????
  OccupiedSpaceCostFunction2D(const OccupiedSpaceCostFunction2D&) = delete;
  OccupiedSpaceCostFunction2D& operator=(const OccupiedSpaceCostFunction2D&) =
      delete;

  const double scaling_factor_;
  const sensor::PointCloud& point_cloud_;
  const Grid2D& grid_;
};

}  // namespace

// 栅格图匹配代价函数
// scaling_factor 为权重
ceres::CostFunction* CreateOccupiedSpaceCostFunction2D(
    const double scaling_factor, const sensor::PointCloud& point_cloud,
    const Grid2D& grid) {
      // 输出维度自动
      // 输入维度为3，即优化的目标参数为3
  return new ceres::AutoDiffCostFunction<OccupiedSpaceCostFunction2D,
                                         ceres::DYNAMIC /* residuals */,
                                         3 /* pose variables */>(
      new OccupiedSpaceCostFunction2D(scaling_factor, point_cloud, grid),
      point_cloud.size());
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
