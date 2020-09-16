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
#include "cartographer/mapping/2d/grid_2d.h"

namespace cartographer {
namespace mapping {
namespace {

// 从proto提取最小free概率值
float MinCorrespondenceCostFromProto(const proto::Grid2D& proto) {
  if (proto.min_correspondence_cost() == 0.f &&
      proto.max_correspondence_cost() == 0.f) {
    LOG(WARNING) << "proto::Grid2D: min_correspondence_cost "
                    "is initialized with 0 indicating an older version of the "
                    "protobuf format. Loading default values.";
    return kMinCorrespondenceCost;
  } else {
    return proto.min_correspondence_cost();
  }
}

// 从proto提取最大free的概率值
float MaxCorrespondenceCostFromProto(const proto::Grid2D& proto) {
  if (proto.min_correspondence_cost() == 0.f &&
      proto.max_correspondence_cost() == 0.f) {
    LOG(WARNING) << "proto::Grid2D: max_correspondence_cost "
                    "is initialized with 0 indicating an older version of the "
                    "protobuf format. Loading default values.";
    return kMaxCorrespondenceCost;
  } else {
    return proto.max_correspondence_cost();
  }
}
}  // namespace

// 配置grid的参数
proto::GridOptions2D CreateGridOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::GridOptions2D options;
  const std::string grid_type_string =
      parameter_dictionary->GetString("grid_type");
  proto::GridOptions2D_GridType grid_type;
  CHECK(proto::GridOptions2D_GridType_Parse(grid_type_string, &grid_type))
      << "Unknown GridOptions2D_GridType kind: " << grid_type_string;
  options.set_grid_type(grid_type);
  options.set_resolution(parameter_dictionary->GetDouble("resolution"));
  return options;
}

// 构造函数，根据输入参数进行配置
// 构造栅格地图， 栅格地图存储的都是free的概率
  // MapLimits包含属性：分辨率， x和y方向上的栅格个数，x和y最大值
  // min_correspondence_cost: 栅格概率代价最小值即free最小值，即hit最大值
  // max_correspondence_cost: 栅格概率代价最大值，即free最大值，即hit最小值
Grid2D::Grid2D(const MapLimits& limits, float min_correspondence_cost,
               float max_correspondence_cost,
               ValueConversionTables* conversion_tables)
    : limits_(limits),
      correspondence_cost_cells_(
          limits_.cell_limits().num_x_cells * limits_.cell_limits().num_y_cells,
          kUnknownCorrespondenceValue),
      min_correspondence_cost_(min_correspondence_cost),
      max_correspondence_cost_(max_correspondence_cost),
      // 实际存储为uint16的整数，此为与cost的转换表格
      // 获取转换表格
      value_to_correspondence_cost_table_(conversion_tables->GetConversionTable(
          max_correspondence_cost, min_correspondence_cost,
          max_correspondence_cost)) {
  CHECK_LT(min_correspondence_cost_, max_correspondence_cost_);
}

// 构造函数，直接从proto中解析参数构建
Grid2D::Grid2D(const proto::Grid2D& proto,
               ValueConversionTables* conversion_tables)
    : limits_(proto.limits()),
      correspondence_cost_cells_(),
      min_correspondence_cost_(MinCorrespondenceCostFromProto(proto)),
      max_correspondence_cost_(MaxCorrespondenceCostFromProto(proto)),
      value_to_correspondence_cost_table_(conversion_tables->GetConversionTable(
          max_correspondence_cost_, min_correspondence_cost_,
          max_correspondence_cost_)) {
  CHECK_LT(min_correspondence_cost_, max_correspondence_cost_);
  // 是否已知已知区域栅格地图大小, 就获取其大小
  if (proto.has_known_cells_box()) {
    const auto& box = proto.known_cells_box();
    known_cells_box_ =
        Eigen::AlignedBox2i(Eigen::Vector2i(box.min_x(), box.min_y()),
                            Eigen::Vector2i(box.max_x(), box.max_y()));
  }
  correspondence_cost_cells_.reserve(proto.cells_size());
  for (const auto& cell : proto.cells()) {
    CHECK_LE(cell, std::numeric_limits<uint16>::max());
    correspondence_cost_cells_.push_back(cell);
  }
}

// Finishes the update sequence.
// 将需要概率更新的index进行更新，
// update_indices_ ?????? 没看到在哪赋值
void Grid2D::FinishUpdate() {
  while (!update_indices_.empty()) {
    DCHECK_GE(correspondence_cost_cells_[update_indices_.back()],
              kUpdateMarker);
    // 更新的方式减去kUpdateMarker=1 << 15
    correspondence_cost_cells_[update_indices_.back()] -= kUpdateMarker;
    update_indices_.pop_back();
  }
}

// Fills in 'offset' and 'limits' to define a subregion of that contains all
// known cells.
// 计算全部已知概率的空间的大小
void Grid2D::ComputeCroppedLimits(Eigen::Array2i* const offset,
                                  CellLimits* const limits) const {
  // 如果是空的，则输出一个1*1的边界
  if (known_cells_box_.isEmpty()) {
    *offset = Eigen::Array2i::Zero();
    *limits = CellLimits(1, 1);
    return;
  }
  *offset = known_cells_box_.min().array();
  *limits = CellLimits(known_cells_box_.sizes().x() + 1,
                       known_cells_box_.sizes().y() + 1);
}

// Grows the map as necessary to include 'point'. This changes the meaning of
// these coordinates going forward. This method must be called immediately
// after 'FinishUpdate', before any calls to 'ApplyLookupTable'.
// 更新的地图大小参数， 应该是新加入的laser导致地图变大，需要动态更新栅格地图大小
// 采用更新其大小
void Grid2D::GrowLimits(const Eigen::Vector2f& point) {
  GrowLimits(point, {mutable_correspondence_cost_cells()},
             {kUnknownCorrespondenceValue});
}

// 更新地图大小，同时将原grid中数据按照位置放入新放大的grid中
void Grid2D::GrowLimits(const Eigen::Vector2f& point,
                        const std::vector<std::vector<uint16>*>& grids,
                        const std::vector<uint16>& grids_unknown_cell_values) {
  CHECK(update_indices_.empty());
  //如果当前的存在point不在范围内，即需要更新，采用迭代方法放大地图边界，
  while (!limits_.Contains(limits_.GetCellIndex(point))) {
    //获取原来的地图大小的中心坐标，即栅格索引
    const int x_offset = limits_.cell_limits().num_x_cells / 2;
    const int y_offset = limits_.cell_limits().num_y_cells / 2;
    // grid最大值更新原来的一半， 地图总大小放大一倍。 即从地图中心位置上下左右均放大原大小一半
    const MapLimits new_limits(
        limits_.resolution(),
        limits_.max() +
            limits_.resolution() * Eigen::Vector2d(y_offset, x_offset),
        CellLimits(2 * limits_.cell_limits().num_x_cells,
                   2 * limits_.cell_limits().num_y_cells));
    // 行数，用于转换1维索引
    const int stride = new_limits.cell_limits().num_x_cells;
    //新的offset
    const int offset = x_offset + stride * y_offset;
    //新大小
    const int new_size = new_limits.cell_limits().num_x_cells *
                         new_limits.cell_limits().num_y_cells;

    //更新grid概率，即将原来的概率赋值在新的grid中
    for (size_t grid_index = 0; grid_index < grids.size(); ++grid_index) {
      std::vector<uint16> new_cells(new_size,
                                    grids_unknown_cell_values[grid_index]);
      for (int i = 0; i < limits_.cell_limits().num_y_cells; ++i) {
        for (int j = 0; j < limits_.cell_limits().num_x_cells; ++j) {
          new_cells[offset + j + i * stride] =
              (*grids[grid_index])[j + i * limits_.cell_limits().num_x_cells];
        }
      }
      *grids[grid_index] = new_cells;
    }
    limits_ = new_limits;
    // 重新计算有效栅格空间边界，即由于起点发生改变，则矩形框的坐标需进行转换
    if (!known_cells_box_.isEmpty()) {
      known_cells_box_.translate(Eigen::Vector2i(x_offset, y_offset));
    }
  }
}

proto::Grid2D Grid2D::ToProto() const {
  proto::Grid2D result;
  *result.mutable_limits() = mapping::ToProto(limits_);
  *result.mutable_cells() = {correspondence_cost_cells_.begin(),
                             correspondence_cost_cells_.end()};
  CHECK(update_indices().empty()) << "Serializing a grid during an update is "
                                     "not supported. Finish the update first.";
  if (!known_cells_box().isEmpty()) {
    auto* const box = result.mutable_known_cells_box();
    box->set_max_x(known_cells_box().max().x());
    box->set_max_y(known_cells_box().max().y());
    box->set_min_x(known_cells_box().min().x());
    box->set_min_y(known_cells_box().min().y());
  }
  result.set_min_correspondence_cost(min_correspondence_cost_);
  result.set_max_correspondence_cost(max_correspondence_cost_);
  return result;
}

}  // namespace mapping
}  // namespace cartographer
