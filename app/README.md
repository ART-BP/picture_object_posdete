# App 模块说明（Object Navigation）

`app/` 负责目标指令解析、视觉检测与分割、点云融合、几何结果发布，以及可选的 `move_base` 跟随控制。

## 命令协议（`/object_cmd`）

- Topic: `/object_cmd`
- Type: `std_msgs/String`
- Payload: JSON 字符串

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `task` | string | 是 | `follow` / `recognition` / `follow_once` / `cancel` |
| `caption` | string | 否 | 目标文本描述；`cancel` 可省略 |

任务行为：

| task | 行为 |
|---|---|
| `follow` | 持续识别并持续发送跟随目标 |
| `recognition` | 单次识别并发布几何结果 |
| `follow_once` | 单次识别并发送一次导航目标 |
| `cancel` | 取消当前任务并取消 `move_base` 目标 |

常用命令：

```bash
# 取消任务
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"cancel\"}'"

# 持续跟随（直到取消）
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow\",\"caption\":\"person in green\"}'"

# 单次识别（发布一次几何结果）
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"recognition\",\"caption\":\"white wall\"}'"

# 单次前往（发送一次目标）
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow_once\",\"caption\":\"white wall\"}'"
```

## 输出协议（`/fusion_lidar_camera/object_depth_json`）

- Topic: `/fusion_lidar_camera/object_depth_json`
- Type: `std_msgs/String`
- Payload: JSON

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `stamp` | float | 时间戳（秒） |
| `frame_id` | string | 点云坐标系 |
| `caption` | string | 当前目标文本 |
| `bbox_xyxy` | list[float] | 检测框 `[x1, y1, x2, y2]` |
| `gdino_score` | float | 检测置信度 |
| `num_points` | int | 目标点数 |
| `centroid_xy_m` | list[float] or null | 目标中心（米） |
| `nearest_surface_xy_m` | list[float] or null | 最近表面点（米） |
| `nearest_surface_dist_m` | float or null | 最近表面距离（米） |

示例：

```json
{
  "stamp": 1713412345.12,
  "frame_id": "lidar",
  "caption": "white wall",
  "bbox_xyxy": [312.2, 118.7, 557.4, 420.6],
  "gdino_score": 0.89,
  "num_points": 146,
  "centroid_xy_m": [2.31, -0.46],
  "nearest_surface_xy_m": [1.88, -0.39],
  "nearest_surface_dist_m": 1.92
}
```

当点数不足或几何估计失败时，以下字段为 `null`：

- `centroid_xy_m`
- `nearest_surface_xy_m`
- `nearest_surface_dist_m`

## 标定参数（当前记录）

内参（fisheye）：

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

外参：

```yaml
Rcl: [-0.999957, 0.007163, 0.005963, -0.005932, 0.004424, -0.999973, -0.007190, -0.999965, -0.004381]
Pcl: [0.017952, -0.097494, -0.175946]
```
