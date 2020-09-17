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

#include "cartographer/mapping/2d/submap_2d.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "Eigen/Geometry"
#include "absl/memory/memory.h"
#include "cartographer/common/port.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/2d/tsdf_range_data_inserter_2d.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

proto::SubmapsOptions2D CreateSubmapsOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::SubmapsOptions2D options;
  options.set_num_range_data(
      parameter_dictionary->GetNonNegativeInt("num_range_data"));
  *options.mutable_grid_options_2d() = CreateGridOptions2D(
      parameter_dictionary->GetDictionary("grid_options_2d").get());
  *options.mutable_range_data_inserter_options() =
      CreateRangeDataInserterOptions(
          parameter_dictionary->GetDictionary("range_data_inserter").get());

  bool valid_range_data_inserter_grid_combination = false;
  const proto::GridOptions2D_GridType& grid_type =
      options.grid_options_2d().grid_type();
  const proto::RangeDataInserterOptions_RangeDataInserterType&
      range_data_inserter_type =
          options.range_data_inserter_options().range_data_inserter_type();
  if (grid_type == proto::GridOptions2D::PROBABILITY_GRID &&
      range_data_inserter_type ==
          proto::RangeDataInserterOptions::PROBABILITY_GRID_INSERTER_2D) {
    valid_range_data_inserter_grid_combination = true;
  }
  if (grid_type == proto::GridOptions2D::TSDF &&
      range_data_inserter_type ==
          proto::RangeDataInserterOptions::TSDF_INSERTER_2D) {
    valid_range_data_inserter_grid_combination = true;
  }
  CHECK(valid_range_data_inserter_grid_combination)
      << "Invalid combination grid_type " << grid_type
      << " with range_data_inserter_type " << range_data_inserter_type;
  CHECK_GT(options.num_range_data(), 0);
  return options;
}

// 构造函数，初始位置的朝向均设置为0
Submap2D::Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
                   ValueConversionTables* conversion_tables)
    : Submap(transform::Rigid3d::Translation(
          Eigen::Vector3d(origin.x(), origin.y(), 0.))),
      conversion_tables_(conversion_tables) {
  grid_ = std::move(grid);
}

// 从proto中构造submap
Submap2D::Submap2D(const proto::Submap2D& proto,
                   ValueConversionTables* conversion_tables)
    : Submap(transform::ToRigid3(proto.local_pose())),
      conversion_tables_(conversion_tables) {
  if (proto.has_grid()) {
    if (proto.grid().has_probability_grid_2d()) {
      grid_ =
          absl::make_unique<ProbabilityGrid>(proto.grid(), conversion_tables_);
    } else if (proto.grid().has_tsdf_2d()) {
      grid_ = absl::make_unique<TSDF2D>(proto.grid(), conversion_tables_);
    } else {
      LOG(FATAL) << "proto::Submap2D has grid with unknown type.";
    }
  }
  set_num_range_data(proto.num_range_data());
  set_insertion_finished(proto.finished());
}

//转换为proto
proto::Submap Submap2D::ToProto(const bool include_grid_data) const {
  proto::Submap proto;
  auto* const submap_2d = proto.mutable_submap_2d();
  *submap_2d->mutable_local_pose() = transform::ToProto(local_pose());
  submap_2d->set_num_range_data(num_range_data());
  submap_2d->set_finished(insertion_finished());
  if (include_grid_data) {
    CHECK(grid_);
    *submap_2d->mutable_grid() = grid_->ToProto();
  }
  return proto;
}

// 从proto中转换submap
void Submap2D::UpdateFromProto(const proto::Submap& proto) {
  CHECK(proto.has_submap_2d());
  const auto& submap_2d = proto.submap_2d();
  set_num_range_data(submap_2d.num_range_data());
  set_insertion_finished(submap_2d.finished());
  if (proto.submap_2d().has_grid()) {
    if (proto.submap_2d().grid().has_probability_grid_2d()) {
      grid_ = absl::make_unique<ProbabilityGrid>(proto.submap_2d().grid(),
                                                 conversion_tables_);
    } else if (proto.submap_2d().grid().has_tsdf_2d()) {
      grid_ = absl::make_unique<TSDF2D>(proto.submap_2d().grid(),
                                        conversion_tables_);
    } else {
      LOG(FATAL) << "proto::Submap2D has grid with unknown type.";
    }
  }
}

void Submap2D::ToResponseProto(
    const transform::Rigid3d&,
    proto::SubmapQuery::Response* const response) const {
  if (!grid_) return;
  response->set_submap_version(num_range_data());
  proto::SubmapQuery::Response::SubmapTexture* const texture =
      response->add_textures();
  grid()->DrawToSubmapTexture(texture, local_pose());
}

//在本submap中插入新的激光数据，参数：1.激光数据； 2. 激光队列
void Submap2D::InsertRangeData(
    const sensor::RangeData& range_data,
    const RangeDataInserterInterface* range_data_inserter) {
  CHECK(grid_);
  CHECK(!insertion_finished());
  // 激光数据帧插入grid中， 采用虚函数接口调用， 根据不同地图类型具体实现
  range_data_inserter->Insert(range_data, grid_.get());
  // 激光数据帧个数自加1
  set_num_range_data(num_range_data() + 1);
}

// 设置submap插入结束标志位，即不再接收新的激光帧
void Submap2D::Finish() {
  CHECK(grid_);
  CHECK(!insertion_finished());
  // 根据参数将grid进行修整和裁剪，作为当前submap的栅格图
  // 主要是因为地图扩展的原因，地图大小和内部数据进行了一定了调整， 注： 地图类型不同采用不同的方法
  grid_ = grid_->ComputeCroppedGrid();
  // 设置结束标志位
  set_insertion_finished(true);
}

// ActiveSubmaps2D构造函数，仅传入配置参数
// 构造时同时创建了插入器
ActiveSubmaps2D::ActiveSubmaps2D(const proto::SubmapsOptions2D& options)
    : options_(options), range_data_inserter_(CreateRangeDataInserter()) {}

// 获取ActiveSubmaps2D维护的两个submap2d
std::vector<std::shared_ptr<const Submap2D>> ActiveSubmaps2D::submaps() const {
  return std::vector<std::shared_ptr<const Submap2D>>(submaps_.begin(),
                                                      submaps_.end());
}

//ActiveSubmaps2D插入一个新的rangedata
std::vector<std::shared_ptr<const Submap2D>> ActiveSubmaps2D::InsertRangeData(
    const sensor::RangeData& range_data) {
  // 如果第一次，即无任何submap2d时
  // 或者如果new的submap的内部含有的激光个数达到配置的阈值
  // 则需要增加一个新的submap，其submap2d的初始位置，为新range的激光坐标
  // 注意这是在插入前先进行了判断，也就说上次循环已经满足阈值条件
  if (submaps_.empty() ||
      submaps_.back()->num_range_data() == options_.num_range_data()) {
    AddSubmap(range_data.origin.head<2>());
  }

  //两个submap全部插入新的range_data，表明old的submap2d包含了所有new submap2d内容
  for (auto& submap : submaps_) {
    submap->InsertRangeData(range_data, range_data_inserter_.get());
  }
  // 如果old的range的数量达到配置的阈值两倍，也表明new的数量也到达了阈值
  // 则 将old的submap进行结束封装，表明submap结束，设置submap2d 结束标志位，同时也进行裁剪仅保留有效value的边界
  if (submaps_.front()->num_range_data() == 2 * options_.num_range_data()) {
    submaps_.front()->Finish();
  }
  return submaps();
}

// 创建插入器接口，同时因为根据类型继承，具体实现根据地图类型选择
// 由于配置采用probability map， 故实际插入器在probability里实现
// 插入器仅根据配置参数进行构造
std::unique_ptr<RangeDataInserterInterface>
ActiveSubmaps2D::CreateRangeDataInserter() {
  switch (options_.range_data_inserter_options().range_data_inserter_type()) {
    case proto::RangeDataInserterOptions::PROBABILITY_GRID_INSERTER_2D:
      return absl::make_unique<ProbabilityGridRangeDataInserter2D>(
          options_.range_data_inserter_options()
              .probability_grid_range_data_inserter_options_2d());
    case proto::RangeDataInserterOptions::TSDF_INSERTER_2D:
      return absl::make_unique<TSDFRangeDataInserter2D>(
          options_.range_data_inserter_options()
              .tsdf_range_data_inserter_options_2d());
    default:
      LOG(FATAL) << "Unknown RangeDataInserterType.";
  }
}

// 初始化一个空的grid地图，作为submap2d初始空白地图
// 尽关心概率地图
std::unique_ptr<GridInterface> ActiveSubmaps2D::CreateGrid(
    const Eigen::Vector2f& origin) {
  // 初始化默认大小100*100像素点
  constexpr int kInitialSubmapSize = 100;
  float resolution = options_.grid_options_2d().resolution();
  switch (options_.grid_options_2d().grid_type()) {
    case proto::GridOptions2D::PROBABILITY_GRID:
      return absl::make_unique<ProbabilityGrid>(
          MapLimits(resolution,
                    origin.cast<double>() + 0.5 * kInitialSubmapSize *
                                                resolution *
                                                Eigen::Vector2d::Ones(),
                    CellLimits(kInitialSubmapSize, kInitialSubmapSize)),
          &conversion_tables_);
    case proto::GridOptions2D::TSDF:
      return absl::make_unique<TSDF2D>(
          MapLimits(resolution,
                    origin.cast<double>() + 0.5 * kInitialSubmapSize *
                                                resolution *
                                                Eigen::Vector2d::Ones(),
                    CellLimits(kInitialSubmapSize, kInitialSubmapSize)),
          options_.range_data_inserter_options()
              .tsdf_range_data_inserter_options_2d()
              .truncation_distance(),
          options_.range_data_inserter_options()
              .tsdf_range_data_inserter_options_2d()
              .maximum_weight(),
          &conversion_tables_);
    default:
      LOG(FATAL) << "Unknown GridType.";
  }
}

// 添加一个新的submap
void ActiveSubmaps2D::AddSubmap(const Eigen::Vector2f& origin) {
  // 如果已经存在两个submap，即old和new均存在
  // 则剔除old的submap2d
  if (submaps_.size() >= 2) {
    // This will crop the finished Submap before inserting a new Submap to
    // reduce peak memory usage a bit.
    // 等待插入结束标志位，即也是submap裁剪结束标志
    CHECK(submaps_.front()->insertion_finished());
    // 剔除old的submap2d
    submaps_.erase(submaps_.begin());
  }
  // 插入一个新的submap2d
  submaps_.push_back(absl::make_unique<Submap2D>(
      origin,
      std::unique_ptr<Grid2D>(
          static_cast<Grid2D*>(CreateGrid(origin).release())),
      &conversion_tables_));
}

}  // namespace mapping
}  // namespace cartographer
