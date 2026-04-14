#!/usr/bin/env python3
import json
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

from GroundingDINO.gdino import GroundingDINO
from MobileSAM.sam import Sam

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import tf
from tf.transformations import quaternion_from_euler

NOTASK = 0
FOLLOW = 1
RECON = 2


class FusionLidarCameraNode:
    def __init__(self) -> None:
        """Initialize model instances, ROS params, pubs/subs, and time synchronizer."""
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_visual_points = rospy.get_param("~topic_visual_points", "/visual_points/cam_front_lidar")
        self.caption = rospy.get_param("~caption", "black box")
        self.box_threshold = float(rospy.get_param("~box_threshold", 0.45))
        self.text_threshold = float(rospy.get_param("~text_threshold", 0.35))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_points = int(rospy.get_param("~min_points", 5))
        self.mask_dilate_px = int(rospy.get_param("~mask_dilate_px", 0))
        self.u_field = rospy.get_param("~u_field", "u")
        self.v_field = rospy.get_param("~v_field", "v")
        self.swap_uv = bool(rospy.get_param("~swap_uv", True))
        self.cluster_grid_size = float(rospy.get_param("~cluster_grid_size", 0.1))

        self.cluster_dense_threshold = int(rospy.get_param("~cluster_dense_threshold", 500))
        self.cluster_min_cell_sparse = int(rospy.get_param("~cluster_min_cell_sparse", 2))
        self.cluster_min_points_sparse = int(rospy.get_param("~cluster_min_points_sparse", 5))
        self.cluster_min_cell_dense = int(rospy.get_param("~cluster_min_cell_dense", 5))
        self.cluster_min_points_dense = int(rospy.get_param("~cluster_min_points_dense", 10))
        self.follow_distance_m = float(rospy.get_param("~follow_distance_m", 0.5))
        self.min_goal_dist_m = float(rospy.get_param("~min_goal_dist_m", 0.2))
        self.goal_frame = rospy.get_param("~goal_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.03))
        self.max_infer_fps = float(rospy.get_param("~max_infer_fps", 0.0))
        self.enable_debug_overlay = bool(rospy.get_param("~enable_debug_overlay", False))
        self.use_bbox_mask_only = bool(rospy.get_param("~use_bbox_mask_only", False))
        self.debug_max_points = int(rospy.get_param("~debug_max_points", 500))
        self.save_debug_images = bool(rospy.get_param("~save_debug_images", False))
        self.output_dir = rospy.get_param(
            "~output_dir",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_fusion"),
        )
        self.run = NOTASK
        self.cmd_stamp = rospy.Time(0)
        self.last_infer_stamp_sec = 0.0

        self.gdino_model = GroundingDINO()
        self.sam_model = Sam()
        self.tf_listener = tf.TransformListener()

        self.pub_debug_image = rospy.Publisher("/fusion_lidar_camera/debug_image", Image, queue_size=1, latch=True)
        # self.pub_object_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)
        self.pub_depth_json = rospy.Publisher("/fusion_lidar_camera/object_depth_json", String, latch=True, queue_size=2)

        #  注册雷达和图像同步
        self.sub_image = Subscriber(self.topic_image, Image)
        self.sub_visual_points = Subscriber(self.topic_visual_points, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            fs=[self.sub_image, self.sub_visual_points],
            queue_size=5,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        #  move_base 目标点发送
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        try:
            rospy.loginfo("Waiting for move_base action server...")
            self.client.wait_for_server(rospy.Duration(2.0))
            rospy.loginfo("Connected to move_base action server.")
        except rospy.ROSException as e:
            rospy.logerr("Failed to connect to move_base action server: %s", str(e))

        #  订阅跟踪命令
        self.sub_cmd = rospy.Subscriber("object_cmd", String, self.cmd_callback, queue_size=10)

        if self.save_debug_images:
            os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo(
            "fusionLidarCamera ready: image=%s, points=%s, max_infer_fps=%.2f, debug=%s, bbox_mask_only=%s",
            self.topic_image,
            self.topic_visual_points,
            self.max_infer_fps,
            str(self.enable_debug_overlay),
            str(self.use_bbox_mask_only),
        )

    def cmd_callback(self, msg: String) -> None:
        """Parse task command JSON and switch node running mode."""
        try:
            tmsg = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("Invalid object_cmd JSON: %s, raw=%s", str(exc), msg.data)
            return

        self.caption = tmsg.get("caption", self.caption)
        task = str(tmsg.get("task", "none")).lower()

        if task == "follow":
            self.run = FOLLOW
            self.cmd_stamp = rospy.Time.now()
        elif task == "recognition":
            self.run = RECON
            self.cmd_stamp = rospy.Time.now()
        elif task == "cancel":
            self.run = NOTASK
            self.cmd_stamp = rospy.Time(0)
            self.client.cancel_goal()
        else:
            self.run = NOTASK
        rospy.loginfo(
            "Received cmd: caption=%s task=%s run=%d cmd_stamp=%.6f",
            self.caption,
            task,
            self.run,
            self._as_float_seconds(self.cmd_stamp),
        )

    def _cluster_params(self, num_points: int) -> Tuple[int, int]:
        """Select clustering thresholds by point count."""
        if num_points >= self.cluster_dense_threshold:
            return self.cluster_min_cell_dense, self.cluster_min_points_dense
        return self.cluster_min_cell_sparse, self.cluster_min_points_sparse

    def _send_follow_goal(self, center_xy: List[float], surface_xy: List[float]) -> None:
        """Send goal: nearest surface point minus follow distance, facing object center."""
        center = np.asarray(center_xy, dtype=np.float32)[:2]
        surface = np.asarray(surface_xy, dtype=np.float32)[:2]
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(surface))):
            rospy.logwarn("skip follow goal: invalid center/surface")
            return

        surface_dist = float(np.linalg.norm(surface))
        if surface_dist < 1e-4:
            rospy.logwarn("skip follow goal: surface too close to origin")
            return

        unit_surface = surface / surface_dist
        goal_dist = max(self.min_goal_dist_m, surface_dist - self.follow_distance_m)
        goal_dist = min(goal_dist, surface_dist)
        goal_xy = unit_surface * goal_dist

        yaw_rel = float(np.arctan2(center[1] - goal_xy[1], center[0] - goal_xy[0]))

        cos_yaw = math.cos(self.base_yaw)
        sin_yaw = math.sin(self.base_yaw)
        goal_map_x = float(self.trans[0] + cos_yaw * goal_xy[0] - sin_yaw * goal_xy[1])
        goal_map_y = float(self.trans[1] + sin_yaw * goal_xy[0] + cos_yaw * goal_xy[1])
        yaw_map = self.base_yaw + yaw_rel

        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = goal_map_x
        goal.target_pose.pose.position.y = goal_map_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)

        self.client.send_goal(goal)
        rospy.loginfo("发布命令时间为： %s", str(goal.target_pose.header.stamp))
        rospy.loginfo(
            "sent follow goal (%s): goal=[%.3f, %.3f] center_xy=[%.3f, %.3f] surface_xy=[%.3f, %.3f] "
            "surface_dist=%.3f follow_distance=%.3f yaw_rel=%.3f yaw_map=%.3f",
            self.goal_frame,
            goal_map_x,
            goal_map_y,
            center[0],
            center[1],
            surface[0],
            surface[1],
            surface_dist,
            self.follow_distance_m,
            yaw_rel,
            yaw_map,
        )


    def cluster_2d_center_nearest_surface(
        self,
        points_xy: np.ndarray,
        grid_size: float = 0.1,
        min_points_per_cell: int = 2,
        min_cluster_points: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster 2D points and return (median center, nearest surface point)."""
        if points_xy is None:
            nan_xy = np.array([np.nan, np.nan], dtype=np.float32)
            return nan_xy, nan_xy.copy()

        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"Expected Nx2 points, got shape={pts.shape}")
        if pts.shape[1] > 2:
            pts = pts[:, :2]
        if pts.shape[0] == 0:
            nan_xy = np.array([np.nan, np.nan], dtype=np.float32)
            return nan_xy, nan_xy.copy()

        if grid_size <= 0:
            raise ValueError(f"grid_size must be > 0, got {grid_size}")

        cell_points = np.floor(pts / float(grid_size)).astype(np.int32)
        unique_cells, counts = np.unique(cell_points, axis=0, return_counts=True)

        dense_mask = counts >= max(1, int(min_points_per_cell))
        if not np.any(dense_mask):
            center = np.median(pts, axis=0).astype(np.float32)
            # Same ordering as norm(), but avoids sqrt for better speed.
            nearest_idx = int(np.argmin(np.sum(pts * pts, axis=1)))
            nearest_xy = pts[nearest_idx].astype(np.float32)
            return center, nearest_xy

        dense_cells = unique_cells[dense_mask]
        dense_set = {tuple(c.tolist()) for c in dense_cells}
        cell_count = {tuple(c.tolist()): int(n) for c, n in zip(unique_cells, counts)}

        visited = set()
        best_component_cells = set()
        best_component_points = 0

        for cell_arr in dense_cells:
            start = tuple(cell_arr.tolist())
            if start in visited:
                continue

            stack = [start]
            visited.add(start)
            component_cells = set()
            component_points = 0

            while stack:
                cx, cy = stack.pop()
                cell = (cx, cy)
                component_cells.add(cell)
                component_points += cell_count.get(cell, 0)

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nxt = (cx + dx, cy + dy)
                        if nxt in dense_set and nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)

            if component_points > best_component_points:
                best_component_points = component_points
                best_component_cells = component_cells

        if not best_component_cells:
            cluster_pts = pts
        else:
            keep = np.fromiter(
                (tuple(c.tolist()) in best_component_cells for c in cell_points),
                dtype=np.bool_,
                count=cell_points.shape[0],
            )
            cluster_pts = pts[keep]
            if cluster_pts.shape[0] < max(1, int(min_cluster_points)):
                cluster_pts = pts

        center = np.median(cluster_pts, axis=0).astype(np.float32)
        # Same ordering as norm(), but avoids sqrt for better speed.
        nearest_idx = int(np.argmin(np.sum(cluster_pts * cluster_pts, axis=1)))
        nearest_xy = cluster_pts[nearest_idx].astype(np.float32)
        return center, nearest_xy


    def ros_image_to_cv2(self, ros_image: Image) -> np.ndarray:
        """Convert a ROS Image message into an OpenCV BGR image."""
        data = np.frombuffer(ros_image.data, dtype=np.uint8)
        h, w = ros_image.height, ros_image.width
        enc = (ros_image.encoding or "").lower()

        if h > 0 and w > 0:
            if enc in ("bgr8", "rgb8"):
                expected = h * w * 3
                if data.size >= expected:
                    frame = data[:expected].reshape((h, w, 3))
                    if enc == "rgb8":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return frame
            elif enc in ("bgra8", "rgba8"):
                expected = h * w * 4
                if data.size >= expected:
                    frame = data[:expected].reshape((h, w, 4))
                    if enc == "rgba8":
                        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif enc in ("mono8", "8uc1"):
                expected = h * w
                if data.size >= expected:
                    gray = data[:expected].reshape((h, w))
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(
                "Unsupported image data "
                f"(encoding={ros_image.encoding}, h={h}, w={w}, bytes={len(ros_image.data)})"
            )
        return decoded

    def cv2_to_ros_image(self, image_bgr: np.ndarray, header) -> Image:
        """Convert an OpenCV BGR image into a ROS Image message with the given header."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape={image_bgr.shape}")

        h, w, c = image_bgr.shape
        ros_image = Image()
        ros_image.header = header
        ros_image.height = h
        ros_image.width = w
        ros_image.encoding = "bgr8"
        ros_image.is_bigendian = 0
        ros_image.step = w * c
        ros_image.data = image_bgr.tobytes()
        return ros_image

    @staticmethod
    def _as_float_seconds(stamp) -> float:
        """Normalize ROS time-like objects to float seconds."""
        if hasattr(stamp, "to_sec"):
            return float(stamp.to_sec())
        return float(stamp)

    def _should_process(self) -> bool:
        """Return True when the node should process this synced frame."""
        return True if self.run == RECON or self.run == FOLLOW else False

    def _resolve_uv_fields(self, cloud_msg: PointCloud2) -> Tuple[str, str]:
        """Resolve u/v field names from PointCloud2 and validate x/y/z presence."""
        field_names = {f.name for f in cloud_msg.fields}

        u_candidates = [self.u_field, "u", "pixel_u", "img_u", "uv_u", "col"]
        v_candidates = [self.v_field, "v", "pixel_v", "img_v", "uv_v", "row"]

        u_name = next((n for n in u_candidates if n in field_names), "")
        v_name = next((n for n in v_candidates if n in field_names), "")
        if not u_name or not v_name:
            raise ValueError(
                f"PointCloud2 lacks u/v fields. available={sorted(field_names)} "
                f"tried_u={u_candidates} tried_v={v_candidates}"
            )

        for axis in ("x", "y", "z"):
            if axis not in field_names:
                raise ValueError(f"PointCloud2 missing '{axis}' field. available={sorted(field_names)}")
        return u_name, v_name

    def _read_xyzuv(self, cloud_msg: PointCloud2, h: int, w: int) -> Tuple[np.ndarray, float]:
        """Vectorized PointCloud2 decode: return in-bound finite (x,y,z,u,v) and inside ratio."""
        u_name, v_name = self._resolve_uv_fields(cloud_msg)
        field_map = {f.name: f for f in cloud_msg.fields}
        dtype_map = {
            PointField.INT8: "i1",
            PointField.UINT8: "u1",
            PointField.INT16: "i2",
            PointField.UINT16: "u2",
            PointField.INT32: "i4",
            PointField.UINT32: "u4",
            PointField.FLOAT32: "f4",
            PointField.FLOAT64: "f8",
        }

        names: List[str] = []
        formats: List[str] = []
        offsets: List[int] = []
        endian = ">" if cloud_msg.is_bigendian else "<"
        for name in ("x", "y", "z", u_name, v_name):
            field = field_map[name]
            if int(field.count) != 1:
                raise ValueError(f"Field '{name}' has count={field.count}, only scalar fields are supported")
            base_fmt = dtype_map.get(field.datatype)
            if base_fmt is None:
                raise ValueError(f"Unsupported PointField datatype={field.datatype} for field '{name}'")
            names.append(name)
            formats.append(endian + base_fmt)
            offsets.append(int(field.offset))

        point_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": int(cloud_msg.point_step),
            }
        )

        width = int(cloud_msg.width)
        height = int(cloud_msg.height)
        n_points = width * height
        if n_points <= 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0

        expected_bytes = int(cloud_msg.row_step) * height
        if len(cloud_msg.data) < expected_bytes:
            raise ValueError(
                f"PointCloud2 data too short: len(data)={len(cloud_msg.data)} expected>={expected_bytes}"
            )

        points = np.ndarray(
            shape=(height, width),
            dtype=point_dtype,
            buffer=cloud_msg.data,
            strides=(int(cloud_msg.row_step), int(cloud_msg.point_step)),
        )

        x = np.asarray(points["x"], dtype=np.float32).reshape(-1)
        y = np.asarray(points["y"], dtype=np.float32).reshape(-1)
        z = np.asarray(points["z"], dtype=np.float32).reshape(-1)
        u = np.asarray(points[u_name], dtype=np.float32).reshape(-1)
        v = np.asarray(points[v_name], dtype=np.float32).reshape(-1)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0

        x = x[finite]
        y = y[finite]
        z = z[finite]
        u = u[finite]
        v = v[finite]

        inside = (u >= 0.0) & (u < float(w)) & (v >= 0.0) & (v < float(h))
        inside_count = int(np.count_nonzero(inside))
        if inside_count == 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0

        xyzuv_inside = np.stack((x[inside], y[inside], z[inside], u[inside], v[inside]), axis=1).astype(
            np.float32, copy=False
        )
        ratio = float(inside_count) / float(finite_count)
        inside_ratio = ratio if 0.0 < ratio < 1.0 else 0.0
        return xyzuv_inside, inside_ratio
    
    def _to_pixel_uv(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert uv fields to integer pixel indices with optional u/v swap."""
        if self.swap_uv:
            u_raw = uv[:, 1]
            v_raw = uv[:, 0]
        else:
            u_raw = uv[:, 0]
            v_raw = uv[:, 1]
        return np.rint(u_raw).astype(np.int32), np.rint(v_raw).astype(np.int32)


    @staticmethod
    def _build_cloud_xyz(header, xyz: np.ndarray) -> PointCloud2:
        """Build a PointCloud2 containing only x/y/z fields from an Nx3 array."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, xyz.astype(np.float32).tolist())

    @staticmethod
    def _bbox_mask(shape_hw: Tuple[int, int], box_xyxy: np.ndarray) -> np.ndarray:
        """Build a rectangular boolean mask from one xyxy box."""
        h, w = shape_hw
        x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        mask = np.zeros((h, w), dtype=bool)
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
        return mask

    def _get_payload(
        self,
        stamp_sec: float,
        frame_id: str,
        caption: str,
        box_xyxy: List[float],
        gdino_score: float,
        center: np.ndarray,
        nearest_surface_xy: np.ndarray,
        num_points: int,
    ) -> dict:
        """Build JSON payload for object center and nearest surface point."""
        center_arr = np.asarray(center, dtype=np.float32) if center is not None else None
        nearest_arr = (
            np.asarray(nearest_surface_xy, dtype=np.float32)
            if nearest_surface_xy is not None
            else None
        )
        valid_center = (
            center_arr is not None
            and center_arr.shape[0] >= 2
            and np.all(np.isfinite(center_arr[:2]))
        )
        valid_nearest = (
            nearest_arr is not None
            and nearest_arr.shape[0] >= 2
            and np.all(np.isfinite(nearest_arr[:2]))
        )
        if num_points < self.min_points or not valid_center or not valid_nearest:
            return {
                "stamp": stamp_sec,
                "frame_id": frame_id,
                "caption": caption,
                "bbox_xyxy": box_xyxy,
                "gdino_score": gdino_score,
                "num_points": int(num_points),
                "centroid_xy_m": None,
                "nearest_surface_xy_m": None,
                "nearest_surface_dist_m": None,
            }

        nearest_dist = float(np.linalg.norm(nearest_arr[:2]))
        return {
            "stamp": stamp_sec,
            "frame_id": frame_id,
            "caption": caption,
            "bbox_xyxy": box_xyxy,
            "gdino_score": gdino_score,
            "num_points": int(num_points),
            "centroid_xy_m": [float(center_arr[0]), float(center_arr[1])],
            "nearest_surface_xy_m": [float(nearest_arr[0]), float(nearest_arr[1])],
            "nearest_surface_dist_m": nearest_dist,
        }
    
    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        """Run full fusion pipeline on synced image+visual-points messages."""
        if not self._should_process():
            return
        
        frame_stamp = image_msg.header.stamp if image_msg.header.stamp != rospy.Time(0) else cloud_msg.header.stamp
        frame_stamp_sec = self._as_float_seconds(frame_stamp) if frame_stamp != rospy.Time(0) else rospy.Time.now().to_sec()

        # Gate by command timestamp first, so one-shot RECON is not consumed by old frames.
        if self.cmd_stamp != rospy.Time(0) and frame_stamp != rospy.Time(0) and frame_stamp < self.cmd_stamp:
            rospy.logdebug(
                "skip old synced frame: frame=%.6f < cmd=%.6f",
                self._as_float_seconds(frame_stamp),
                self._as_float_seconds(self.cmd_stamp),
            )
            return

        # Follow mode can be throttled to reduce heavy GDINO+SAM load.
        if self.run == FOLLOW and self.max_infer_fps > 0.0:
            min_dt = 1.0 / self.max_infer_fps
            if self.last_infer_stamp_sec > 0.0 and (frame_stamp_sec - self.last_infer_stamp_sec) < min_dt:
                return
            self.last_infer_stamp_sec = frame_stamp_sec

        if self.run == RECON:
            self.run = NOTASK
        try:
            try:
                self.tf_listener.waitForTransform(
                    self.goal_frame,
                    self.base_frame,
                    frame_stamp,
                    rospy.Duration(self.tf_timeout_s),
                )
                self.trans, rot = self.tf_listener.lookupTransform(
                    self.goal_frame,
                    self.base_frame,
                    frame_stamp,
                )
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                rospy.logwarn(
                    "skip follow goal: transform %s <- %s unavailable at frame_stamp=%.6f: %s",
                    self.goal_frame,
                    self.base_frame,
                    self._as_float_seconds(frame_stamp),
                    str(exc),
                )
                return

            self.base_yaw = float(tf.transformations.euler_from_quaternion(rot)[2])


            image = self.ros_image_to_cv2(image_msg)

            detections, labels = self.gdino_model.predict(
                image=image,
                caption=self.caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            if len(detections.xyxy) == 0:
                rospy.logwarn("No detections for caption='%s'", self.caption)
                return

            conf = (
                np.asarray(detections.confidence, dtype=np.float32)
                if getattr(detections, "confidence", None) is not None
                else np.zeros((len(detections.xyxy),), dtype=np.float32)
            )
            best_idx = int(np.argmax(conf))
            box_xyxy = detections.xyxy[best_idx]
            gdino_score = float(conf[best_idx]) if best_idx < len(conf) else 0.0

            if self.use_bbox_mask_only:
                mask = self._bbox_mask(image.shape[:2], box_xyxy)
            else:
                mask, _, _ = self.sam_model.get_mask_by_box(
                    box_xyxy=box_xyxy,
                    image=image,
                    image_format="BGR",
                    multimask_output=False,
                )
            if self.mask_dilate_px > 0:
                k = self.mask_dilate_px * 2 + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            h, w = mask.shape[:2]
            xyzuv_inside, inside_ratio = self._read_xyzuv(cloud_msg, h, w)
            if xyzuv_inside.shape[0] == 0:
                rospy.logwarn("No points from %s", self.topic_visual_points)
                return

            u_inside, v_inside = self._to_pixel_uv(xyzuv_inside[:, 3:5])

            if xyzuv_inside.shape[0] == 0:
                rospy.logwarn("No projected points inside image bounds")
                return


            on_object = mask[v_inside, u_inside]
            object_xyzuv = xyzuv_inside[on_object]
            object_xyz = object_xyzuv[:, :3] if object_xyzuv.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

            if object_xyz.shape[0] < self.min_points:
                rospy.logwarn(
                    "Object points too few: %d < min_points=%d",
                    object_xyz.shape[0],
                    self.min_points,
                )

            # self.pub_object_points.publish(self._build_cloud_xyz(cloud_msg.header, object_xyz))

            if object_xyz.shape[0] > 0:
                min_points_per_cell, min_cluster_points = self._cluster_params(object_xyz.shape[0])
                center, nearest_surface_xy = self.cluster_2d_center_nearest_surface(
                    object_xyz[:, :2],
                    grid_size=self.cluster_grid_size,
                    min_points_per_cell=min_points_per_cell,
                    min_cluster_points=min_cluster_points,
                )
            else:
                center = np.array([np.nan, np.nan], dtype=np.float32)
                nearest_surface_xy = np.array([np.nan, np.nan], dtype=np.float32)

            payload = self._get_payload(
                stamp_sec=self._as_float_seconds(image_msg.header.stamp),
                frame_id=cloud_msg.header.frame_id,
                caption=self.caption,
                box_xyxy=[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])],
                gdino_score=gdino_score,
                center=center,
                nearest_surface_xy=nearest_surface_xy,
                num_points=object_xyz.shape[0],
            )
            self.pub_depth_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))


            # debug
            need_debug = self.enable_debug_overlay and (
                self.save_debug_images or self.pub_debug_image.get_num_connections() > 0
            )
            if need_debug:
                annotated = self.gdino_model.annotate(image, detections, labels)
                debug = annotated
                debug[mask] = (debug[mask] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)
                if xyzuv_inside.shape[0] > 0:
                    all_step = max(1, xyzuv_inside.shape[0] // self.debug_max_points)
                    all_uv = np.stack([u_inside, v_inside], axis=1)[::all_step]
                    for uu, vv in all_uv:
                        cv2.circle(debug, (int(uu), int(vv)), 1, (255, 200, 0), -1)
                if object_xyzuv.shape[0] > 0:
                    sample_step = max(1, object_xyzuv.shape[0] // self.debug_max_points)
                    obj_u = u_inside[on_object]
                    obj_v = v_inside[on_object]
                    sample_uv = np.stack([obj_u, obj_v], axis=1)[::sample_step].astype(np.int32)
                    for uu, vv in sample_uv:
                        cv2.circle(debug, (int(uu), int(vv)), 1, (0, 0, 255), -1)

                line1 = f"center={payload['centroid_xy_m']}"
                line2 = f"nearest={payload['nearest_surface_xy_m']}"
                line3 = f"num_points={payload['num_points']}"
                x, y0, dy = 20, 40, 30
                font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                cv2.putText(debug, line1, (x, y0 + 0 * dy), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(debug, line2, (x, y0 + 1 * dy), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)
                self.pub_debug_image.publish(self.cv2_to_ros_image(debug, image_msg.header))
            # self.pub_mask_image.publish(self.cv2_to_ros_image(mask_bgr, image_msg.header))

            # if self.save_debug_images:
            #     stamp_ns = str(image_msg.header.stamp.to_nsec())
            #     cv2.imwrite(os.path.join(self.output_dir, f"{stamp_ns}_debug.jpg"), debug)
            #     cv2.imwrite(os.path.join(self.output_dir, f"{stamp_ns}_mask.png"), mask_u8)

            rospy.loginfo(
                "fusion done: caption='%s' inside=%.3f obj=%d center=%s",
                self.caption,
                inside_ratio,
                object_xyz.shape[0],
                str(payload["centroid_xy_m"]),
            )
            if payload["centroid_xy_m"] is not None and payload["nearest_surface_xy_m"] is not None:
                self._send_follow_goal(
                    center_xy=payload["centroid_xy_m"],
                    surface_xy=payload["nearest_surface_xy_m"],
                )
        except Exception as exc:
            rospy.logerr("fusion callback failed: %s", str(exc))


def main() -> None:
    """ROS node entrypoint."""
    rospy.init_node("fusion_lidar_camera_node")
    FusionLidarCameraNode()
    rospy.spin()


if __name__ == "__main__":
    main()
