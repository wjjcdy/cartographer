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

#ifndef CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
#define CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/map_limits.h"
#include "cartographer/mapping/proto/2d/submaps_options_2d.pb.h"
#include "cartographer/mapping/proto/serialization.pb.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "cartographer/mapping/submaps.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

proto::SubmapsOptions2D CreateSubmapsOptions2D(
    common::LuaParameterDictionary* parameter_dictionary);

class Submap2D : public Submap {
 public:
  // submap构造函数， 包含1.submap初始位置， 2.grid栅格地图，即坐标和对应占有概率， 3. 浮点数与到uint16转换表格
  Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
           ValueConversionTables* conversion_tables);
  // 隐式构造函数，仅有一个参数，后面参数有默认值
  explicit Submap2D(const proto::Submap2D& proto,
                    ValueConversionTables* conversion_tables);

  proto::Submap ToProto(bool include_grid_data) const override;
  void UpdateFromProto(const proto::Submap& proto) override;

  void ToResponseProto(const transform::Rigid3d& global_submap_pose,
                       proto::SubmapQuery::Response* response) const override;

  // 返回栅格地图
  const Grid2D* grid() const { return grid_.get(); }

  // Insert 'range_data' into this submap using 'range_data_inserter'. The
  // submap must not be finished yet.
  // 利用RangeDataInserterInterface来插入并更新概率图
  void InsertRangeData(const sensor::RangeData& range_data,
                       const RangeDataInserterInterface* range_data_inserter);
  void Finish();

 private:
  std::unique_ptr<Grid2D> grid_;    // 存储概率地图
  ValueConversionTables* conversion_tables_;  // 浮点数与到uint16转换表格，估计用于概率图转换成整型进行计算
};

// The first active submap will be created on the insertion of the first range
// data. Except during this initialization when no or only one single submap
// exists, there are always two submaps into which range data is inserted: an
// old submap that is used for matching, and a new one, which will be used for
// matching next, that is being initialized.
//
// Once a certain number of range data have been inserted, the new submap is
// considered initialized: the old submap is no longer changed, the "new" submap
// is now the "old" submap and is used for scan-to-map matching. Moreover, a
// "new" submap gets created. The "old" submap is forgotten by this object.
// ActiveSubmaps2D类维护了两个submap,一个old 和一个new，
// old用于matching，而new用于插入
// 当new submap达到一定数量时，则停止插入。
// 抛弃old，将new变为old
// 创建一个新的new submap
class ActiveSubmaps2D {
 public:
  explicit ActiveSubmaps2D(const proto::SubmapsOptions2D& options);

  ActiveSubmaps2D(const ActiveSubmaps2D&) = delete;
  ActiveSubmaps2D& operator=(const ActiveSubmaps2D&) = delete;

  // Inserts 'range_data' into the Submap collection.
  std::vector<std::shared_ptr<const Submap2D>> InsertRangeData(
      const sensor::RangeData& range_data);

  std::vector<std::shared_ptr<const Submap2D>> submaps() const;

 private:
  // 创建插入器接口
  std::unique_ptr<RangeDataInserterInterface> CreateRangeDataInserter();
  // 创建grid地图接口
  std::unique_ptr<GridInterface> CreateGrid(const Eigen::Vector2f& origin);
  void FinishSubmap();
  // 加入一个新的submap
  void AddSubmap(const Eigen::Vector2f& origin);

  const proto::SubmapsOptions2D options_;                              // 配置信息
  std::vector<std::shared_ptr<Submap2D>> submaps_;                     // 维护submap2d的列表
  std::unique_ptr<RangeDataInserterInterface> range_data_inserter_;    // 插入器接口
  ValueConversionTables conversion_tables_;                            // 浮点数与到uint16转换表格
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
