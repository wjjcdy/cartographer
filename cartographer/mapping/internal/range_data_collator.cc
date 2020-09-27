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

#include "cartographer/mapping/internal/range_data_collator.h"

#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/mapping/local_slam_result_data.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

// 距离传感器插入处理，形成集合
sensor::TimedPointCloudOriginData RangeDataCollator::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& timed_point_cloud_data) {
  CHECK_NE(expected_sensor_ids_.count(sensor_id), 0);
  // TODO(gaschler): These two cases can probably be one.
  // 此传感器类型数据已有
  if (id_to_pending_data_.count(sensor_id) != 0) {
    current_start_ = current_end_;
    // When we have two messages of the same sensor, move forward the older of
    // the two (do not send out current).
    // 采用旧的时间戳
    current_end_ = id_to_pending_data_.at(sensor_id).time;
    auto result = CropAndMerge();
    // 用新的数据替换原有的数据
    id_to_pending_data_.emplace(sensor_id, timed_point_cloud_data);
    return result;
  }
  // 直接插入
  id_to_pending_data_.emplace(sensor_id, timed_point_cloud_data);
  // 若现在收到的数据类型未全，即期望收到种类未全，直接退出，无需融合
  if (expected_sensor_ids_.size() != id_to_pending_data_.size()) {
    return {};
  }
  current_start_ = current_end_;
  // We have messages from all sensors, move forward to oldest.
  common::Time oldest_timestamp = common::Time::max();
  // 找传感器数据中最早的时间戳
  for (const auto& pair : id_to_pending_data_) {
    oldest_timestamp = std::min(oldest_timestamp, pair.second.time);
  }
  current_end_ = oldest_timestamp;
  return CropAndMerge();
}

//
sensor::TimedPointCloudOriginData RangeDataCollator::CropAndMerge() {
  //定义，输出的集合时间戳，是融合收齐后，按照最早时间戳的数据
  sensor::TimedPointCloudOriginData result{current_end_, {}, {}};
  bool warned_for_dropped_points = false;
  // 遍历收集的所有传感器类型数据
  for (auto it = id_to_pending_data_.begin();
       it != id_to_pending_data_.end();) {
    sensor::TimedPointCloudData& data = it->second;       // 带原点位置和时间戳
    sensor::TimedPointCloud& ranges = it->second.ranges;  // 仅点云（每一个点也都带有测量时间戳）

    auto overlap_begin = ranges.begin();                  // 记录在所有点云中开始时间戳的位置，即上时刻集合的时间戳
    while (overlap_begin < ranges.end() &&
           data.time + common::FromSeconds((*overlap_begin).time) <
               current_start_) {
      ++overlap_begin;
    }

    auto overlap_end = overlap_begin;                     // 记录所有点云结束时间戳的位置，即此时刻集合的时间戳
    while (overlap_end < ranges.end() &&
           data.time + common::FromSeconds((*overlap_end).time) <=
               current_end_) {
      ++overlap_end;
    }
                                                         // 如果某个传感器点云前面时间戳早于当前集合定义的时间戳，则丢弃
    if (ranges.begin() < overlap_begin && !warned_for_dropped_points) {
      LOG(WARNING) << "Dropped " << std::distance(ranges.begin(), overlap_begin)
                   << " earlier points.";
      warned_for_dropped_points = true;
    }

    // Copy overlapping range.
    if (overlap_begin < overlap_end) {
      std::size_t origin_index = result.origins.size();    //获取下个插入的index，即当前集合的个数
      result.origins.push_back(data.origin);               // 插入原点坐标
      const float time_correction =                        // 获取此传感器时间与集合时间戳的误差，
          static_cast<float>(common::ToSeconds(data.time - current_end_));
      for (auto overlap_it = overlap_begin; overlap_it != overlap_end;
           ++overlap_it) {
        sensor::TimedPointCloudOriginData::RangeMeasurement point{*overlap_it,
                                                                  origin_index};
        // current_end_ + point_time[3]_after == in_timestamp +
        // point_time[3]_before
        point.point_time.time += time_correction;          // 针对每个点时间戳进行修正
        result.ranges.push_back(point);                     
      }
    }

    // Drop buffered points until overlap_end. 
    if (overlap_end == ranges.end()) {
      it = id_to_pending_data_.erase(it);
    } else if (overlap_end == ranges.begin()) {
      ++it;
    } else {   
      data = sensor::TimedPointCloudData{
          data.time, data.origin,
          sensor::TimedPointCloud(overlap_end, ranges.end())};  //感觉没用？？？？？,即使删除了，data也无用了
      ++it;
    }
  }

  // 对集合中所有点云，进行按照时间顺序排序
  std::sort(result.ranges.begin(), result.ranges.end(),
            [](const sensor::TimedPointCloudOriginData::RangeMeasurement& a,
               const sensor::TimedPointCloudOriginData::RangeMeasurement& b) {
              return a.point_time.time < b.point_time.time;
            });
  return result;
}

}  // namespace mapping
}  // namespace cartographer
