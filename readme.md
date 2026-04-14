# Object Detection + LiDAR-Camera Fusion

本项目将 GroundingDINO（文本检测）和 MobileSAM（实例分割）与相机-激光点云融合在一起，输出目标的二维空间位置，并可直接下发 `move_base` 跟随目标。

---

## 1. 项目整体功能

项目核心能力分为两部分：

1. **感知**：根据文本描述（caption）在图像中检测目标并分割目标区域。  
2. **几何定位与控制**：利用点云中的 `x/y/z/u/v` 对齐信息，把图像目标映射到三维点云，估计目标中心与最近表面点，再发布 JSON 结果并可发送导航目标。

当前主流程脚本是 `app/objectnav.py`。

---

## 2. 整体基本原理（端到端流程）

以 `objectnav.py` 为准，单帧处理链路如下：

1. **同步输入**：`ApproximateTimeSynchronizer` 同步图像与点云。
2. **任务门控**：仅在 `FOLLOW` 或 `RECON` 状态处理帧，并支持按命令时间戳过滤旧帧。
3. **视觉检测**：GroundingDINO 按 `caption` 输出候选框，取最高置信度目标框。
4. **目标分割**：优先用 MobileSAM 依据目标框得到 mask；可切换为仅 bbox mask。
5. **点云解析与筛选**：
   - 从 `PointCloud2` 读取 `x/y/z/u/v`；
   - 过滤 NaN；
   - 过滤不在图像范围内的 `u/v`；
   - 通过 `mask[v, u]` 保留落在目标上的点。
6. **目标几何估计**：
   - 对目标点的 `x/y` 进行网格聚类；
   - 取主聚类中位数为目标中心；
   - 取离传感器原点最近点为目标表面点。
7. **结果输出**：
   - 发布 `/fusion_lidar_camera/object_depth_json`；
   - 可选发布调试图像；
   - 若有效则发送 `move_base` 跟随目标。

---

## 3. 目录与分模块说明

### 3.1 主流程与业务脚本（`app/`）

- `app/objectnav.py`  
  当前推荐主节点。支持识别/跟随任务状态机、TF 变换、跟随目标发送、调试覆盖图、推理限频、bbox-only 掩膜、点云向量化解析。

- `app/fusionLidarCamera0414.py`  
  较新的融合版本，逻辑与 `objectnav.py` 接近，但结构和参数略少/不同（例如无命令时间戳门控）。

- `app/fusionLidarCamera0413.py`  
  更早版本，流程更简化，主要用于演进对照。

- `app/fusionLidarCamera copy.py`  
  历史拷贝版本，用于实验保留。

- `app/getmask_object.py`  
  单独的“检测+分割并保存结果图”工具脚本，便于快速验证视觉模型效果。

- `app/test_gdino.py`  
  GroundingDINO 基础连通性/耗时测试。

- `app/test_time.py`  
  GroundingDINO + MobileSAM 串联耗时测试，结果写入 `time_data.txt`。

- `app/cmd.txt`  
  `object_cmd` 的常用发布命令样例。

### 3.2 模型封装模块

- `GroundingDINO/gdino.py`  
  对 GroundingDINO 的 Python 封装，提供 `predict`、`annotate`、bbox 信息构建等能力。默认模型权重：
  - `GroundingDINO/weights/groundingdino_swint_ogc.pth`

- `MobileSAM/sam.py`  
  对 MobileSAM 的封装，提供按框/按点分割、最佳 mask 选择、目标渲染、裁剪保存等能力。默认权重：
  - `MobileSAM/weights/mobile_sam.pt`

- `GroundingDINO/gdinoros.py`  
  GroundingDINO 的 ROS 节点封装（单视觉检测链路），可独立发布检测可视化和检测框 JSON。

---

## 4. `objectnav.py` 模块细节

### 4.1 状态机与任务控制

- `NOTASK=0`：空闲  
- `FOLLOW=1`：持续跟随  
- `RECON=2`：单次识别（处理一帧后自动回到 `NOTASK`）

通过订阅 `object_cmd`（`std_msgs/String`，JSON 字符串）控制任务和目标文本。

### 4.2 话题输入输出

订阅：

- `~topic_image`（默认 `/camera/go2/front/image_raw`）
- `~topic_visual_points`（默认 `/visual_points/cam_front_lidar`）
- `object_cmd`（任务控制）

发布：

- `/fusion_lidar_camera/object_depth_json`：目标几何结果 JSON（主输出）
- `/fusion_lidar_camera/debug_image`：调试可视化图（可选）
- `/fusion_lidar_camera/object_points`：当前版本在 `objectnav.py` 中已注释（保留代码）

### 4.3 点云-图像对齐关键前提

点云消息必须包含字段：

- `x, y, z`
- `u, v`（字段名可通过参数配置或候选名自动匹配：`u/pixel_u/img_u/uv_u/col` 与 `v/pixel_v/img_v/uv_v/row`）

否则节点会报错并拒绝处理该帧。

### 4.4 `_read_xyzuv` 的向量化实现（CPU）

当前实现采用 NumPy 向量化解析 `PointCloud2`，核心目标是减少 Python 逐点循环开销：

1. 根据 `PointField` 构造结构化 `dtype`（处理 `offset/datatype/endian/point_step`）。
2. 以 `row_step/point_step` 作为 stride 将二进制 buffer 映射为二维点阵视图。
3. 批量取出 `x/y/z/u/v` 并做有限值过滤。
4. 按图像边界过滤 `u/v`，输出 `xyzuv_inside` 及 `inside_ratio`。

这部分是 CPU 优化，不依赖额外 GPU 框架，兼顾性能和可维护性。

### 4.5 几何估计与跟随目标生成

目标点集合 `object_xyz[:, :2]` 会进入网格聚类：

- 网格尺寸：`cluster_grid_size`
- 稀疏/稠密场景采用不同最小阈值
- 主连通簇中位数作为 `centroid_xy_m`
- 离原点最近点作为 `nearest_surface_xy_m`

跟随目标策略：

- 在目标最近表面点基础上后退 `follow_distance_m`
- 最小保持 `min_goal_dist_m`
- 用 TF 将机器人坐标系下目标转换到 `goal_frame`
- 朝向目标中心，发送 `MoveBaseGoal`

---

## 5. 参数说明（`objectnav.py`）

常用参数如下（均为 ROS 私有参数 `~`）：

- `topic_image`：图像话题
- `topic_visual_points`：带 `u/v` 的点云话题
- `caption`：检测文本
- `box_threshold` / `text_threshold`：GroundingDINO 阈值
- `sync_slop`：图像-点云近似同步容忍时间
- `min_points`：最小目标点数
- `mask_dilate_px`：mask 膨胀像素
- `u_field` / `v_field`：点云字段名
- `swap_uv`：是否交换 `u/v` 维度（不同标定链路常见）
- `cluster_grid_size`：聚类网格尺寸
- `cluster_dense_threshold`：稠密点云阈值
- `cluster_min_cell_sparse` / `cluster_min_points_sparse`：稀疏场景聚类阈值
- `cluster_min_cell_dense` / `cluster_min_points_dense`：稠密场景聚类阈值
- `follow_distance_m`：跟随保持距离
- `min_goal_dist_m`：最小目标距离
- `goal_frame` / `base_frame`：导航与机器人基坐标系
- `tf_timeout_s`：TF 查询超时
- `max_infer_fps`：`FOLLOW` 模式最大推理频率（0 表示不限）
- `enable_debug_overlay`：是否启用调试叠加图
- `use_bbox_mask_only`：是否绕过 SAM，仅使用检测框矩形 mask
- `debug_max_points`：调试图最多绘制点数
- `save_debug_images` / `output_dir`：本地保存调试图配置

---

## 6. 命令协议（`object_cmd`）

消息类型：`std_msgs/String`，内容为 JSON。

字段：

- `task`：`follow` / `recognition` / `cancel`
- `caption`：目标文本描述

示例：

```bash
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"recognition\",\"caption\":\"white wall\"}'"
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"follow\",\"caption\":\"black box\"}'"
rostopic pub -1 /object_cmd std_msgs/String "data: '{\"task\":\"cancel\",\"caption\":\"black box\"}'"
```

---

## 7. 运行方式（建议）

### 7.1 环境准备

至少需要以下依赖环境：

- ROS（`rospy`, `sensor_msgs`, `message_filters`, `tf`, `actionlib`, `move_base_msgs`）
- Python：`numpy`, `opencv-python`, `torch`, `supervision`
- 模型权重文件：
  - `GroundingDINO/weights/groundingdino_swint_ogc.pth`
  - `MobileSAM/weights/mobile_sam.pt`

### 7.2 启动主节点

在仓库根目录执行：

```bash
PYTHONPATH=. python3 app/objectnav.py
```

说明：

- 若你通过 ROS launch/rosrun 管理，也可将 `PYTHONPATH` 与参数配置放到启动文件中。
- 默认设备策略是：`torch.cuda.is_available()` 为真时使用 GPU，否则回退 CPU。

### 7.3 推理精度/速度相关环境变量

常见可调项：

- `MOBILESAM_USE_AMP`：`1/0`，控制 MobileSAM AMP（默认开启）
- GroundingDINO 侧的 AMP / resize 也可通过测试脚本中的环境变量组合验证（参见 `app/test_time.py`）

---

## 8. 性能与工程注意事项

- 本工程中 **视觉模型推理通常是主要耗时项**，点云过滤是次要项。  
- `_read_xyzuv` 已改为 NumPy 向量化 CPU 实现，可显著降低 Python for-loop 解析开销。  
- 如果点云字段缺失或 `u/v` 不匹配，优先检查：
  - `u_field/v_field` 参数
  - `swap_uv` 是否正确
  - 点云投影链路是否发布了图像坐标字段  
- `FOLLOW` 模式建议设置 `max_infer_fps`，避免视觉模型占满算力。

---

## 9. 推荐使用哪个脚本

日常开发/集成建议优先使用：

- `app/objectnav.py`（主线版本）

历史对照可参考：

- `app/fusionLidarCamera0414.py`
- `app/fusionLidarCamera0413.py`
- `app/fusionLidarCamera copy.py`
