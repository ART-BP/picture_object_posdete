# Object Detection + LiDAR-Camera Fusion

本仓库是一个基于 ROS 的“视觉语义目标检索 + 激光几何定位 + move_base 跟随”工程。  
核心入口是 [`app/object_nav.py`](app/object_nav.py)。

---

## 1. 工程在做什么

给机器人一个文本目标（例如 `black box`、`person in green`）后，系统会：

1. 在相机图像里做文本条件检测（GroundingDINO 或 YOLOE）。
2. 对目标框做实例分割（MobileSAM，或使用 bbox 矩形掩膜）。
3. 将 LiDAR 点投影到图像平面，与目标 mask 对齐，筛出目标对应三维点。
4. 估计目标中心点与最近表面点（2D 平面几何）。
5. 发布结构化 JSON 结果，并在跟随任务下发送 `move_base` 目标点。

这让系统具备“用自然语言指定目标并进行几何驱动导航”的能力。

---

## 2. 核心入口：`app/object_nav.py`

`object_nav.py` 是主线业务节点，节点名为 `fusion_lidar_camera_node`，负责：

1. 读取运行配置（`app/config.cfg` 或 `OBJECTNAV_CONFIG` 指定路径）。
2. 管理任务状态机（`Notask / Follow / Recognize / Follow_once`）。
3. 同步图像与点云（`ApproximateTimeSynchronizer`）。
4. 异步执行重计算链路（检测 + 分割 + 点云融合）并只保留最新帧。
5. 输出调试图、目标点云、目标几何 JSON。
6. 调用 `move_base` 跟随目标。
7. 在目标丢失时触发恢复策略（前进 / 原地旋转 / 取消任务）。

---

## 3. 端到端主流程（按实际代码）

### 3.1 启动阶段

1. 读取配置：[`app/config.cfg`](app/config.cfg)。
2. 加载相机内参与畸变：`camdepthfusion/param_camera.yaml` 中 `rational_polynomial`。
3. 初始化检测模型：
   - `gdino` -> [`GroundingDINO/gdino.py`](GroundingDINO/gdino.py)
   - 其他值（如 `yoloev8n`）-> [`yoloe/yoloe.py`](yoloe/yoloe.py)
4. 初始化分割模型：[`MobileSAM/sam.py`](MobileSAM/sam.py)。
5. 建立 `move_base` action client 和 TF listener。

### 3.2 输入门控与排队

`synced_callback` 先做轻量门控，不直接做重计算：

1. 丢弃过旧帧（`max_lidarimage_delay`）。
2. 仅在 `Follow / Follow_once / Recognize` 三种任务状态下继续。
3. `Follow` 模式按 `max_infer_fps` 限频。
4. 把任务封装成 job 放入长度 1 的队列，始终保留“最新帧”。

### 3.3 Worker 处理（重计算主链路）

`_process_job` 会做：

1. TF 对齐检查：确认在 `frame_stamp` 时刻能查到 `goal_frame <- base_frame`。
2. 视觉检测：选置信度最高 bbox。
3. mask 生成：
   - 默认：SAM 按 bbox 分割。
   - 可选：`use_bbox_mask_only=true` 时直接用矩形 mask。
4. 点云读取：`cloudpoints_handle._read_xyz`。
5. LiDAR -> 图像投影：`points_project.project_lidar_to_image_with_rational_polynomial`。
6. 根据 `mask[v,u]` 取目标点云 `object_xyz`。
7. 用 `cluster_2d_center_nearest_surface` 求：
   - `centroid_xy_m`
   - `nearest_surface_xy_m`
8. 发布 JSON 到 `/fusion_lidar_camera/object_depth_json`。
9. `Recognize` 模式在一帧后回到 `Notask`。
10. `Follow`/`Follow_once` 下若结果有效则发 `move_base` 目标。

---

## 4. 任务协议与状态机

通过 `object_cmd`（`std_msgs/String`，JSON 字符串）控制：

```json
{"task":"follow","caption":"black box"}
```

`task` 支持：

1. `follow`：持续跟随。
2. `recognition`：单次识别并输出结果，不持续跟随。
3. `follow_once`：仅触发一次跟随计算。
4. `cancel`：取消导航并清空待处理队列。

示例命令见 [`app/cmd.txt`](app/cmd.txt)。

---

## 5. 话题与数据接口

### 5.1 订阅

1. `topic_image`（默认 `/camera/go2/front/image_raw`）
2. `topic_points`（默认 `/lidar_points`）
3. `object_cmd`

注：配置里保留了 `topic_visual_points` 字段，但当前主链路同步使用的是 `topic_points`。

### 5.2 发布

1. `/fusion_lidar_camera/object_depth_json`（主要业务输出，`std_msgs/String`）
2. `/fusion_lidar_camera/debug_image`（调试可视化）
3. `/fusion_lidar_camera/object_points`（目标点云）

### 5.3 JSON 输出字段

主要字段包括：

1. `stamp`, `frame_id`, `caption`
2. `bbox_xyxy`, `gdino_score`, `num_points`
3. `centroid_xy_m`（无效时为 `null`）
4. `nearest_surface_xy_m`（无效时为 `null`）
5. `nearest_surface_dist_m`

---

## 6. 几何与导航策略

### 6.1 点云-图像融合

1. 使用 [`camdepthfusion/points_project.py`](camdepthfusion/points_project.py) 中外参 `R/T` 将 LiDAR 点转换到相机坐标。
2. 按 `rational_polynomial` 畸变模型投影到像素坐标。
3. 用目标 mask 在像素平面筛选对应三维点。

### 6.2 聚类与目标点

`cluster_2d_center_nearest_surface`（[`cloudpoints_handle.py`](camdepthfusion/cloudpoints_handle.py)）在 XY 平面做网格聚类，输出：

1. 主连通簇中位数中心（用于目标方位）。
2. 最近表面点（用于安全距离与靠近控制）。

### 6.3 跟随目标生成

在机器人底盘坐标下计算目标方向，再结合当前 `base_link` 在 `map` 中姿态，生成 `MoveBaseGoal`：

1. 目标太近（`min_goal_dist_m`）时不下发。
2. 姿态朝向目标中心方向。
3. 发送到 `move_base` action server。

---

## 7. 目标丢失恢复机制

恢复逻辑在 [`app/recovery.py`](app/recovery.py)，按“丢失持续时长”分阶段触发：

1. `move_after_sec`：沿最后目标运动方向前进。
2. `rotate_after_sec`：原地旋转（默认按上次方向的符号决定旋转方向）。
3. `cancel_after_sec`：取消任务并回到 `Notask`。

---

## 8. 关键配置（`app/config.cfg`）

建议重点关注：

1. 模型与语义：
   - `model` (`gdino` / `yoloe...`)
   - `caption`
   - `box_threshold`, `text_threshold`
2. 同步与时延门控：
   - `sync_slop`, `sync_queue_size`
   - `max_lidarimage_delay`, `max_tolerate_delay`
   - `max_infer_fps`
3. 融合与聚类：
   - `min_points`
   - `mask_dilate_px`（当前实现为 `erode`，会收缩 mask 边缘）
   - `cluster_grid_size`
   - 稀疏/稠密聚类阈值组
4. 导航与 TF：
   - `goal_frame`, `base_frame`, `tf_timeout_s`
   - `min_goal_dist_m`
5. 恢复策略：
   - `recovery_move_after_s`, `recovery_rotate_after_s`, `recovery_cancel_after_s`
   - `recovery_forward_dist_m`, `recovery_rotate_deg`, `recovery_tick_hz`
6. 调试：
   - `enable_debug_overlay`
   - `use_bbox_mask_only`
   - `save_debug_images`, `output_dir`

---

## 9. 运行方式

### 9.1 前置条件

1. ROS1（代码风格按 Noetic）
2. 可用的 `move_base` 与 TF 树（至少 `goal_frame <- base_frame` 可查）
3. 模型权重存在：
   - `GroundingDINO/weights/groundingdino_swint_ogc.pth`
   - `MobileSAM/weights/mobile_sam.pt`
   - `yoloe/weights/*.pt`（若使用 YOLOE）

### 9.2 启动

在仓库根目录：

```bash
PYTHONPATH=. python3 app/object_nav.py
```

或：

```bash
PYTHONPATH=. python3 -m app.object_nav
```

### 9.3 发送任务

```bash
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"recognition\",\"caption\":\"person\"}'"
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow\",\"caption\":\"black box\"}'"
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"cancel\",\"caption\":\"black box\"}'"
```

---

## 10. 目录导览

1. [`app/`](app)
   - 业务主节点、配置、恢复控制、历史版本。
2. [`camdepthfusion/`](camdepthfusion)
   - 相机参数读取、点云解析、投影模型、调试测试脚本。
3. [`GroundingDINO/`](GroundingDINO)
   - 文本检测模型与封装。
4. [`MobileSAM/`](MobileSAM)
   - 分割模型与封装。
5. [`yoloe/`](yoloe)
   - 备选文本检测模型封装。
6. [`bag/`](bag)
   - 数据记录目录（若有）。

---

## 11. 相关脚本说明

1. [`app/object_nav.py`](app/object_nav.py)：主入口（推荐）。
2. [`app/object_timenav.py`](app/object_timenav.py)：含详细计时日志的变体。
3. [`camdepthfusion/test.py`](camdepthfusion/test.py)：相机-激光投影联调节点。
4. [`camdepthfusion/test_visual.py`](camdepthfusion/test_visual.py)：投影可视化验证脚本。

---

## 12. 工程边界与默认假设

1. 外参 `R/T` 当前来自 `points_project.py` 常量，未在 `object_nav.py` 中动态标定。
2. 结果质量高度依赖：
   - 相机标定参数
   - LiDAR-相机外参
   - 文本提示词有效性
3. `move_base` 不可用时，节点仍可做感知与 JSON 输出，但无法执行跟随动作。

