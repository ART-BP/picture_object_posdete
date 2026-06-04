# App 模块说明

`app/` 是目标识别、点云融合和导航跟随的业务节点目录。核心流程是：

1. 接收图像和点云。
2. 根据文本指令检测目标。
3. 使用 SAM/检测框生成目标区域。
4. 将雷达点投影到图像中，筛出目标点云。
5. 计算目标中心点和最近表面点。
6. 发布目标几何 JSON、调试图像和目标点云。
7. 在跟随任务中向 `move_base` 发送目标点。

## 主要文件

| 文件 | 说明 |
|---|---|
| `object_detection.py` | 主业务节点，支持 GDINO/YoloE、SAM、点云融合、跟随、恢复动作 |
| `object_detection_gdino.py` | GDINO 单模型版本 |
| `object_detection_undistory.py` | 对去畸变图像处理的版本 |
| `only_detection.py` | 只做识别和融合发布，不做导航跟随 |
| `params_load.py` | 读取 `app/config/config.cfg` |
| `recovery.py` | 目标丢失后的恢复动作状态机 |
| `config/config.cfg` | 运行参数配置 |

## 配置文件

默认读取：

```bash
app/config/config.cfg
```

也可以通过环境变量指定：

```bash
export OBJECTNAV_CONFIG=/path/to/config.cfg
```

主要配置项：

| 参数 | 说明 |
|---|---|
| `model` | 检测模型，支持 `gdino` 或 YoloE 相关模型名 |
| `caption` | 默认目标描述 |
| `box_threshold` | 检测框置信度阈值 |
| `text_threshold` | GDINO 文本阈值 |
| `topic_image` | 图像话题 |
| `topic_points` | 点云话题 |
| `sync_slop` | 图像和点云近似同步时间窗口 |
| `sync_queue_size` | 同步队列大小 |
| `max_lidarimage_delay` | 丢弃过旧帧的时间阈值 |
| `max_tolerate_delay` | 结果发布最大允许延迟，`0` 表示不启用 |
| `max_infer_fps` | 最大推理频率，`0` 表示不限制 |
| `min_points` | 目标点云最少点数 |
| `mask_dilate_px` | 目标 mask 膨胀像素 |
| `cluster_grid_size` | 目标点云聚类栅格大小 |
| `min_goal_dist_m` | 跟随时和目标保持的最小距离 |
| `goal_frame` | 导航目标坐标系，通常为 `map` |
| `base_frame` | 机器人底盘坐标系，通常为 `base_link` |
| `enable_debug_overlay` | 是否发布调试图像 |
| `use_bbox_mask_only` | 是否只使用检测框 mask |
| `save_debug_images` | 是否保存调试图 |
| `output_dir` | 调试图保存目录 |

## 输入话题

图像和点云话题由 `config.cfg` 配置：

```ini
topic_image = /camera/go2/front/image_raw
topic_points = /lidar_points
```

指令话题：

- Topic: `/object_cmd`
- Type: `std_msgs/String`
- Payload: JSON 字符串

## 命令协议

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `task` | string | 是 | 任务类型 |
| `caption` | string | 否 | 目标文本描述 |

支持的 `task`：

| task | 行为 |
|---|---|
| `follow` | 持续识别目标，并持续发送跟随导航目标 |
| `recognition` | 持续识别，并发布目标几何结果 |
| `recognition_once` | 单次识别，并发布一次目标几何结果 |
| `follow_once` | 单次识别，并发送一次导航目标 |
| `cancel` | 取消当前任务，并取消 `move_base` 目标 |

常用命令：

```bash
# 取消当前任务
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"cancel\"}'"

# 持续跟随
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow\",\"caption\":\"person in green\"}'"

# 持续识别并发布结果
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"recognition\",\"caption\":\"black box\"}'"

# 单次识别
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"recognition_once\",\"caption\":\"black box\"}'"

# 单次前往
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow_once\",\"caption\":\"black box\"}'"
```

## 输出话题

| Topic | Type | 说明 |
|---|---|---|
| `/fusion_lidar_camera/object_depth_json` | `std_msgs/String` | 目标几何结果 JSON |
| `/fusion_lidar_camera/image` | `sensor_msgs/Image` | 调试图像 |
| `/fusion_lidar_camera/object_points` | `sensor_msgs/PointCloud2` | 目标点云 |

## 输出 JSON

`/fusion_lidar_camera/object_depth_json` 的 payload 是 JSON 字符串。

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `stamp` | float | 图像时间戳，单位秒 |
| `frame_id` | string | 输入点云坐标系 |
| `base_frame` | string | 底盘坐标系 |
| `global_frame` | string | 全局坐标系 |
| `caption` | string | 当前目标描述 |
| `bbox_xyxy` | list[float] | 检测框 `[x1, y1, x2, y2]` |
| `gdino_score` | float | 检测置信度 |
| `num_points` | int | 目标点云数量 |
| `centroid_xyz_m` | list[float] | 目标中心点，全局坐标系，单位米 |
| `nearest_surface_xyz_m` | list[float] | 最近表面点，全局坐标系，单位米 |
| `nearest_surface_dist_m` | float | 最近表面点到机器人距离，单位米 |

示例：

```json
{
  "stamp": 1713412345.12,
  "frame_id": "lidar",
  "base_frame": "base_link",
  "global_frame": "map",
  "caption": "black box",
  "bbox_xyxy": [312.2, 118.7, 557.4, 420.6],
  "gdino_score": 0.89,
  "num_points": 146,
  "centroid_xyz_m": [2.31, -0.46, 0.12],
  "nearest_surface_xyz_m": [1.88, -0.39, 0.08],
  "nearest_surface_dist_m": 1.92
}
```

当点数不足、TF 失败或几何结果无效时，不发布该 JSON。

## 标定参数记录

当前主流程使用 `camdepthfusion/camera_op/config/param_camera.yaml` 中的 `fisheye` 相机参数，并在 `camdepthfusion/project_cloudpoints/points_project.py` 中读取雷达到相机的外参。

内参记录：

```yaml
fisheye:
  camera_name: go2_front_fisheye
  camera_matrix:
    rows: 3
    cols: 3
    data: [1203.762044004368, 0.0, 981.7904792654031, 0.0, 1203.7009720218682, 525.2625697472332, 0.0, 0.0, 1.0]
  distortion_model: fisheye
  distortion_coefficients:
    rows: 1
    cols: 4
    data: [-0.06940178268945467, -0.05259276838826166, 0.060392401913685174, -0.03652503468416535]
  rectification_matrix:
    rows: 3
    cols: 3
    data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  projection_matrix:
    rows: 3
    cols: 4
    data: [824.000458318671, 0.0, 990.3299740527029, 0.0, 0.0, 823.9586532610845, 526.1425913864505, 0.0, 0.0, 0.0, 1.0, 0.0]
```

外参记录：

```yaml
Rcl: [-0.999957, 0.007163, 0.005963, -0.005932, 0.004424, -0.999973, -0.007190, -0.999965, -0.004381]
Pcl: [0.017952, -0.097494, -0.175946]
```

## 运行

示例：

```bash
python app/object_detection.py
```

运行前需要保证：

- ROS master 已启动。
- 图像话题和点云话题正常发布。
- `move_base` 可用，或只使用识别类任务。
- GDINO、SAM、YoloE 等模型依赖已准备好。
