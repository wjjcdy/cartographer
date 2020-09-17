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
#include "cartographer/mapping/2d/probability_grid.h"

#include <limits>

#include "absl/memory/memory.h"
#include "cartographer/mapping/probability_values.h"
#include "cartographer/mapping/submaps.h"

namespace cartographer {
namespace mapping {

ProbabilityGrid::ProbabilityGrid(const MapLimits& limits,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(limits, kMinCorrespondenceCost, kMaxCorrespondenceCost,
             conversion_tables),
      conversion_tables_(conversion_tables) {}

ProbabilityGrid::ProbabilityGrid(const proto::Grid2D& proto,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(proto, conversion_tables), conversion_tables_(conversion_tables) {
  CHECK(proto.has_probability_grid_2d());
}

// Sets the probability of the cell at 'cell_index' to the given
// 'probability'. Only allowed if the cell was unknown before.
// 设置对应index的概率值， 在cell单元存储的则是概率对应的free 占有率， 对应的uint16整形数
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability) {
  // 取cell_index在栅格地图中对应索引的单元的地址
  // ToFlatIndex, 二维坐标转换为1维索引
  // mutable_correspondence_cost_cells() 返回的grid中grid地图
  uint16& cell =
      (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
  CHECK_EQ(cell, kUnknownProbabilityValue);
  // 将概率值转换对应的整型数，然后存入对应的索引的单元中
  cell =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));

  // 返回的是有效概率值的矩形框，每更新一个cell， 则更新矩形框box， 采用eigen库extend，自动更新有效框
  mutable_known_cells_box()->extend(cell_index.matrix());
}

// Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds()
// to the probability of the cell at 'cell_index' if the cell has not already
// been updated. Multiple updates of the same cell will be ignored until
// FinishUpdate() is called. Returns true if the cell was updated.
//
// If this is the first call to ApplyOdds() for the specified cell, its value
// will be set to probability corresponding to 'odds'.
// 将index的位置进行更新的。
// 采用的方法是
bool ProbabilityGrid::ApplyLookupTable(const Eigen::Array2i& cell_index,
                                       const std::vector<uint16>& table) {
  DCHECK_EQ(table.size(), kUpdateMarker);
  // 2维坐标转换为1维索引
  const int flat_index = ToFlatIndex(cell_index);
  // 对应单元位置
  uint16* cell = &(*mutable_correspondence_cost_cells())[flat_index];
  // 判断是否超出32768 ， 正常范围应该是1~32767
  if (*cell >= kUpdateMarker) {
    return false;
  }

  // 需更新的index放入队列，用于后续finish时，将kUpdateMarker减去，因table表格都加上了个kUpdateMarker，暂时不理解
  mutable_update_indices()->push_back(flat_index);
  // 通过查表进行更新单元
  *cell = table[*cell];
  DCHECK_GE(*cell, kUpdateMarker);
  mutable_known_cells_box()->extend(cell_index.matrix());
  return true;
}

//获取栅格图类型， 为概率地图
GridType ProbabilityGrid::GetGridType() const {
  return GridType::PROBABILITY_GRID;
}

// Returns the probability of the cell with 'cell_index'.
// 返回对应的index的占有概率值
float ProbabilityGrid::GetProbability(const Eigen::Array2i& cell_index) const {
  if (!limits().Contains(cell_index)) return kMinProbability;
  return CorrespondenceCostToProbability(ValueToCorrespondenceCost(
      correspondence_cost_cells()[ToFlatIndex(cell_index)]));
}

proto::Grid2D ProbabilityGrid::ToProto() const {
  proto::Grid2D result;
  result = Grid2D::ToProto();
  result.mutable_probability_grid_2d();
  return result;
}

//
std::unique_ptr<Grid2D> ProbabilityGrid::ComputeCroppedGrid() const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  // 获取目前有效grid的原点偏移和xy边界
  ComputeCroppedLimits(&offset, &cell_limits);
  const double resolution = limits().resolution();
  //获取 有效grid对应的最大的x和y长度
  const Eigen::Vector2d max =
      limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
  // 根据有效的大小重新更新grid地图
  std::unique_ptr<ProbabilityGrid> cropped_grid =
      absl::make_unique<ProbabilityGrid>(
          MapLimits(resolution, max, cell_limits), conversion_tables_);

  // 迭代cell_limits 中所有栅格单元， 将栅格内容根据offset进行移动
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) continue;
    cropped_grid->SetProbability(xy_index, GetProbability(xy_index + offset));
  }

  return std::unique_ptr<Grid2D>(cropped_grid.release());
}

// 序列化相关函数，暂不用关心
bool ProbabilityGrid::DrawToSubmapTexture(
    proto::SubmapQuery::Response::SubmapTexture* const texture,
    transform::Rigid3d local_pose) const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  ComputeCroppedLimits(&offset, &cell_limits);

  std::string cells;
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) {
      cells.push_back(0 /* unknown log odds value */);
      cells.push_back(0 /* alpha */);
      continue;
    }
    // We would like to add 'delta' but this is not possible using a value and
    // alpha. We use premultiplied alpha, so when 'delta' is positive we can
    // add it by setting 'alpha' to zero. If it is negative, we set 'value' to
    // zero, and use 'alpha' to subtract. This is only correct when the pixel
    // is currently white, so walls will look too gray. This should be hard to
    // detect visually for the user, though.
    const int delta =
        128 - ProbabilityToLogOddsInteger(GetProbability(xy_index + offset));
    const uint8 alpha = delta > 0 ? 0 : -delta;
    const uint8 value = delta > 0 ? delta : 0;
    cells.push_back(value);
    cells.push_back((value || alpha) ? alpha : 1);
  }

  common::FastGzipString(cells, texture->mutable_cells());
  texture->set_width(cell_limits.num_x_cells);
  texture->set_height(cell_limits.num_y_cells);
  const double resolution = limits().resolution();
  texture->set_resolution(resolution);
  const double max_x = limits().max().x() - resolution * offset.y();
  const double max_y = limits().max().y() - resolution * offset.x();
  *texture->mutable_slice_pose() = transform::ToProto(
      local_pose.inverse() *
      transform::Rigid3d::Translation(Eigen::Vector3d(max_x, max_y, 0.)));

  return true;
}

}  // namespace mapping
}  // namespace cartographer
