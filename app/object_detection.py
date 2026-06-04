#!/usr/bin/env python3
import configparser
import json
import math
import os
import queue
import threading
from enum import IntEnum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

from GroundingDINO.gdino import GroundingDINO
from MobileSAM.sam import Sam
from yoloe.yoloe import Yoloe

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import tf
from tf.transformations import quaternion_from_euler

from camdepthfusion.project_cloudpoints import points_project
from camdepthfusion.project_cloudpoints import cloudpoints_handle
from camdepthfusion.camera_op import camera_handle
from app.recovery import RecoveryAction, RecoveryController
from app.params_load import _load_runtime_config, _cfg_get

class TaskState(IntEnum):
    Notask = 0
    Follow = 1
    Recognize = 2
    Follow_once = 3
    Recognize_once = 4


class FusionLidarCameraNode:
    def __init__(self) -> None:
        """Initialize models, runtime config, pubs/subs, and time synchronizer."""
        cfg = _load_runtime_config()

        default_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output_fusion",
        )

        params = camera_handle.load_camera_params_from_yaml(camera_model="fisheye")  # For logging camera params and compatibility check
        self.K = np.asarray(params["K"],dtype=np.float64)
        self.D = np.asarray(params["D"],dtype=np.float64)

        self.topic_image = _cfg_get(cfg, "topic_image", "/camera/go2/front/image_raw", str)
        self.topic_points = _cfg_get(cfg, "topic_points", "/lidar_points", str)

        if self.topic_points.endswith("loc_scan_undistort"):
            self.R = points_project.R_base_cam
            self.T = points_project.T_base_cam
            self.points_undisort = True
        else:
            self.R = points_project.R
            self.T = points_project.T
            self.points_undisort = False

        caption = _cfg_get(cfg, "caption", "black box", str)
        box_threshold = _cfg_get(cfg, "box_threshold", 0.55, float)
        text_threshold = _cfg_get(cfg, "text_threshold", 0.55, float)

        self.min_points = _cfg_get(cfg, "min_points", 5, int)
        self.mask_dilate_px = _cfg_get(cfg, "mask_dilate_px", 2, int)

        self.cluster_grid_size = _cfg_get(cfg, "cluster_grid_size", 0.2, float)
        self.cluster_dense_threshold = _cfg_get(cfg, "cluster_dense_threshold", 100, int)
        self.cluster_min_cell_sparse = _cfg_get(cfg, "cluster_min_cell_sparse", 2, int)
        self.cluster_min_points_sparse = _cfg_get(cfg, "cluster_min_points_sparse", 5, int)
        self.cluster_min_cell_dense = _cfg_get(cfg, "cluster_min_cell_dense", 5, int)
        self.cluster_min_points_dense = _cfg_get(cfg, "cluster_min_points_dense", 10, int)
        self.cluster_z_grid_size = _cfg_get(cfg, "cluster_z_grid_size", 0.6, float)
        self.min_goal_dist_m = _cfg_get(cfg, "min_goal_dist_m", 0.5, float)
        self.goal_frame = _cfg_get(cfg, "goal_frame", "map", str)
        self.base_frame = _cfg_get(cfg, "base_frame", "base_link", str)
        self.tf_timeout_s = _cfg_get(cfg, "tf_timeout_s", 0.03, float)
        self.max_lidarimage_delay = _cfg_get(cfg, "max_lidarimage_delay", 0.6, float)
        self.max_tolerate_delay = _cfg_get(cfg, "max_tolerate_delay", 0.0, float)
        self.max_infer_fps = _cfg_get(cfg, "max_infer_fps", 1.0, float)
        self.enable_debug_overlay = _cfg_get(cfg, "enable_debug_overlay", True, bool)
        self.use_bbox_mask_only = _cfg_get(cfg, "use_bbox_mask_only", False, bool)
        self.debug_max_points = _cfg_get(cfg, "debug_max_points", 500, int)
        self.save_debug_images = _cfg_get(cfg, "save_debug_images", False, bool)
        self.output_dir = _cfg_get(cfg, "output_dir", default_output_dir, str)
        self.recovery_move_after_s = _cfg_get(cfg, "recovery_move_after_s", 5.0, float)
        self.recovery_rotate_after_s = _cfg_get(cfg, "recovery_rotate_after_s", 10.0, float)
        self.recovery_cancel_after_s = _cfg_get(cfg, "recovery_cancel_after_s", 30.0, float)
        self.recovery_rotate_interval_s = _cfg_get(cfg, "recovery_rotate_interval_s", 2.0, float)
        self.recovery_forward_dist_m = _cfg_get(cfg, "recovery_forward_dist_m", 2.0, float)
        self.recovery_rotate_deg = _cfg_get(cfg, "recovery_rotate_deg", 90.0, float)
        self.recovery_tick_hz = _cfg_get(cfg, "recovery_tick_hz", 10.0, float)
        
        self.run = TaskState.Notask
        self.cmd_stamp = rospy.Time(0)
        self.cmd_seq = 0
        self.last_infer_stamp_sec = 0.0
        self.last_time = rospy.Time.now().to_sec()
        self.low_identity_streak = 0
        self.reacquire_counter = 0
        self.state_lock = threading.Lock()
        self.recovery = RecoveryController(
            move_after_sec=self.recovery_move_after_s,
            rotate_after_sec=self.recovery_rotate_after_s,
            cancel_after_sec=self.recovery_cancel_after_s,
            rotate_interval_sec=self.recovery_rotate_interval_s,
        )


        select = _cfg_get(cfg, "model", "gdino", str).strip().lower()
        select = rospy.get_param("~model", select)
        if select == "gdino":
            self.detecte_model = GroundingDINO()
            self.detecte_model.setparameters(caption=caption, box_threshold=box_threshold, 
                                        text_threshold=text_threshold,
                                        return_labels=self.enable_debug_overlay)
        else:
            model = select[-3:]
            self.detecte_model = Yoloe(model)
            self.detecte_model.setparameters(caption=caption, threshold=box_threshold)
        
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_target = cv2.imread(os.path.join(repo_root, "bag", "test0_crop.png"))
        if self.image_target is None:
            rospy.logwarn(
                "template image not found or unreadable: %s",
                os.path.join(repo_root, "bag", "test0_crop.png"),
            )
        self.template_hs_hist = self._compute_hs_hist(self.image_target)
        self.sam_model = Sam()

        self.tf_listener = tf.TransformListener()
        self.job_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self.recovery_thread = threading.Thread(
            target=self._recovery_loop,
            name="fusion_recovery",
            daemon=True,
        )
        self.recovery_thread.start()

        self.pub_debug_image = rospy.Publisher("/fusion_lidar_camera/image", Image, queue_size=1, latch=True)
        self.pub_object_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)
        self.pub_depth_json = rospy.Publisher("/fusion_lidar_camera/object_depth_json", String, latch=True, queue_size=2)

        #  注册雷达和图像同步
        sync_slop = _cfg_get(cfg, "sync_slop", 0.05, float)
        sync_queue_size = _cfg_get(cfg, "sync_queue_size", 1, int)
        sub_image = Subscriber(self.topic_image, Image)
        sub_points = Subscriber(self.topic_points, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            fs=[sub_image, sub_points],
            queue_size=max(1, sync_queue_size),
            slop=sync_slop,
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
        self.command_topic = "/task/stream"
        self.sub_cmd = rospy.Subscriber(self.command_topic, String, self.cmd_callback, queue_size=10)
        rospy.on_shutdown(self._on_shutdown)

        if self.save_debug_images:
            os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo(
            "fusionLidarCamera ready: image=%s, points=%s, sync_queue=%d, sync_slop=%.3f, max_infer_fps=%.2f, "
            "debug=%s, bbox_mask_only=%s, max_lidarimage_delay=%.3f, max_tolerate_delay=%.3f, model_id=%s",
            self.topic_image,
            self.topic_points,
            max(1, sync_queue_size),
            sync_slop,
            self.max_infer_fps,
            str(self.enable_debug_overlay),
            str(self.use_bbox_mask_only),
            self.max_lidarimage_delay,
            self.max_tolerate_delay,
            getattr(self.detecte_model, "name", type(self.detecte_model).__name__),
        )

    def cmd_callback(self, msg: String) -> None:
        """Parse task command JSON and switch node running mode."""
        try:
            tmsg = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("Invalid %s JSON: %s, raw=%s", self.command_topic, str(exc), msg.data)
            return

        parsed = self._parse_task_stream_command(tmsg)
        if parsed is None:
            rospy.loginfo_throttle(
                1.0,
                "skip unsupported %s command: %s",
                self.command_topic,
                msg.data,
            )
            return

        task, caption, params = parsed
        now = rospy.Time.now()
        now_sec = now.to_sec()

        # Only accepted commands start a fresh recovery context.
        self.recovery.clear()
        with self.state_lock:
            self.cmd_seq += 1

            if task == "follow":
                self.detecte_model.setparameters(caption=caption)
                self.run = TaskState.Follow
                self.cmd_stamp = now
            elif task == "recognition":
                self.detecte_model.setparameters(caption=caption)
                self.run = TaskState.Recognize
                self.cmd_stamp = now
            elif task == "recognition_once":
                self.detecte_model.setparameters(caption=caption)
                self.run = TaskState.Recognize_once
                self.cmd_stamp = now
            elif task == "follow_once":
                self.detecte_model.setparameters(caption=caption)
                self.run = TaskState.Follow_once
                self.cmd_stamp = now
            elif task == "cancel":
                self.run = TaskState.Notask
                self.cmd_stamp = rospy.Time(0)
            else:
                return

            cmd_stamp = self.cmd_stamp
            cmd_seq = self.cmd_seq

        self.recovery.on_task(task=task, now_sec=now_sec)

        if task in ("follow", "recognition", "recognition_once", "follow_once"):
            self._ensure_worker_started()

        if task == "cancel":
            self.client.cancel_goal()
            self._clear_pending_jobs()

        self.last_time = now_sec
        rospy.loginfo(
            "Received cmd: params=%s task=%s  cmd_stamp=%.6f cmd_seq=%d",
            params,
            task,
            self._as_float_seconds(cmd_stamp),
            cmd_seq,
        )

    def _parse_task_stream_command(self, tmsg: dict) -> Optional[Tuple[str, str, object]]:
        """Accept only the supported /task/stream command shapes."""
        raw_task = str(tmsg.get("task", "")).strip()
        params = tmsg.get("params", None)
        task_lower = raw_task.lower()

        # Compatible with the follow command protocol used by object_detection_gdino.py.
        if task_lower in ("follow", "recognition", "recognition_once", "follow_once", "cancel"):
            caption = self._params_to_caption(params, self.detecte_model.caption)
            return task_lower, caption, params

        if raw_task == "搜索" and isinstance(params, (list, tuple)) and len(params) >= 2:
            action = str(params[0]).strip()
            caption = "person"

            # {"task": "搜索", "params": ["开始", "伤员"], ...}
            if action == "开始" and caption:
                return "recognition", caption, params

            # {"task": "搜索", "params": ["暂停", ""], ...}
            if action == "暂停" and caption == "":
                return "cancel", "", params

            # {"task": "搜索", "params": ["继续", "伤员"], ...}
            if action == "继续" and caption:
                return "recognition", caption, params

            return None

        # {"task": "基础动作", "params": "取消任务", ...}
        if raw_task == "基础动作" and str(params).strip() == "取消任务":
            return "cancel", "", params

        return None

    @staticmethod
    def _params_to_caption(params, default: str = "") -> str:
        """Convert task params into the detection text prompt."""
        if params is None:
            return str(default)
        if isinstance(params, str):
            return params
        if isinstance(params, (list, tuple)):
            return ", ".join(str(item) for item in params)
        return str(params)

    def _ensure_worker_started(self) -> None:
        """Start worker lazily after receiving a valid task command."""
        with self.worker_lock:
            if self.stop_event.is_set():
                return
            if self.worker_thread is not None and self.worker_thread.is_alive():
                return
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                name="fusion_worker",
                daemon=True,
            )
            self.worker_thread.start()

    def _recovery_loop(self) -> None:
        tick_hz = max(1.0, float(self.recovery_tick_hz))
        sleep_dt = 1.0 / tick_hz  
              
        if self.run == TaskState.Notask:
            self.stop_event.wait(sleep_dt)  

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            event = self.recovery.poll(now_sec=rospy.Time.now().to_sec())
            if event is not None:
                self._handle_recovery_event(event)
            self.stop_event.wait(sleep_dt)

    def _lookup_base_pose(self) -> Optional[Tuple[Tuple[float, float, float], float]]:
        try:
            self.tf_listener.waitForTransform(
                self.goal_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_s),
            )
            trans, rot = self.tf_listener.lookupTransform(
                self.goal_frame,
                self.base_frame,
                rospy.Time(0),
            )
            base_yaw = float(tf.transformations.euler_from_quaternion(rot)[2])
            return (float(trans[0]), float(trans[1]), float(trans[2])), base_yaw
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "recovery TF lookup failed: %s", str(exc))
            return None

    def _send_recovery_forward_goal(self, heading_rad: float) -> None:
        pose = self._lookup_base_pose()
        if pose is None:
            return
        base_trans, base_yaw = pose
        yaw_map = base_yaw + float(heading_rad)
        goal_x = float(base_trans[0] + self.recovery_forward_dist_m * math.cos(yaw_map))
        goal_y = float(base_trans[1] + self.recovery_forward_dist_m * math.sin(yaw_map))
        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = goal_x
        goal.target_pose.pose.position.y = goal_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)
        self.client.send_goal(goal)

    def _send_recovery_rotate_goal(self, heading_rad: float) -> None:
        pose = self._lookup_base_pose()
        if pose is None:
            return
        base_trans, base_yaw = pose
        turn_sign = 1.0 if float(heading_rad) >= 0.0 else -1.0
        yaw_map = base_yaw + turn_sign * math.radians(float(self.recovery_rotate_deg))
        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = float(base_trans[0])
        goal.target_pose.pose.position.y = float(base_trans[1])
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)
        self.client.send_goal(goal)

    def _handle_recovery_event(self, event) -> None:
        if event.action == RecoveryAction.MOVE_LAST_DIRECTION:
            self._send_recovery_forward_goal(event.heading_rad)
            rospy.logwarn("recovery move triggered: lost=%.2fs", event.lost_sec)
            return

        if event.action == RecoveryAction.ROTATE_IN_PLACE:
            self._send_recovery_rotate_goal(event.heading_rad)
            rospy.logwarn("recovery rotate triggered: lost=%.2fs", event.lost_sec)
            return

        if event.action == RecoveryAction.CANCEL_TASK:
            with self.state_lock:
                self.run = TaskState.Notask
                self.cmd_stamp = rospy.Time(0)
                self.cmd_seq += 1
            self.client.cancel_goal()
            self._clear_pending_jobs()
            rospy.logwarn("recovery cancel triggered: lost=%.2fs", event.lost_sec)

    def _on_shutdown(self) -> None:
        """Stop worker thread quickly during node shutdown."""
        self.stop_event.set()
        self._clear_pending_jobs()
        if self.recovery_thread.is_alive():
            self.recovery_thread.join(timeout=0.5)
        worker = self.worker_thread
        if worker is not None and worker.is_alive():
            try:
                self.job_queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=0.5)

    def _clear_pending_jobs(self) -> None:
        """Drop all queued jobs so outdated frames are not processed."""
        while True:
            try:
                self.job_queue.get_nowait()
            except queue.Empty:
                break

    def _enqueue_latest_job(self, job: dict) -> None:
        """Keep only the latest synced frame job in queue."""
        try:
            self.job_queue.put_nowait(job)
            return
        except queue.Full:
            pass

        try:
            self.job_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            # Worker may have just queued another task; skip silently.
            pass

    def _cluster_params(self, num_points: int) -> Tuple[int, int]:
        """Select clustering thresholds by point count."""
        if num_points >= self.cluster_dense_threshold:
            return self.cluster_min_cell_dense, self.cluster_min_points_dense
        return self.cluster_min_cell_sparse, self.cluster_min_points_sparse

    def _send_follow_goal(
        self,
        center_xy: List[float],
        surface_xy: List[float],
        base_trans: Tuple[float, float, float],
        base_yaw: float,
    ) -> None:
        """Send goal: nearest surface point minus follow distance, facing object center."""
        center = np.asarray(center_xy, dtype=np.float32)[:2]
        surface = np.asarray(surface_xy, dtype=np.float32)[:2]
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(surface))):
            rospy.logwarn("skip follow goal: invalid center/surface")
            return

        surface_dist = float(np.linalg.norm(surface))
        center_dist = float(np.linalg.norm(center))
        if surface_dist <= self.min_goal_dist_m or center_dist <= self.min_goal_dist_m:
            rospy.loginfo("object distance too close")
            return       

        if surface_dist > center_dist:
            rospy.logwarn("surface_dist > center_dist")
            return 
           
        yaw_center = float(np.arctan2(center[1] , center[0]))
        ratio = min(1, max(0, (surface_dist - self.min_goal_dist_m) / center_dist))
        goal_x = ratio * center[0]
        goal_y = ratio * center[1]

        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)

        goal_map_x = float(base_trans[0] + cos_yaw * goal_x - sin_yaw * goal_y)
        goal_map_y = float(base_trans[1] + sin_yaw * goal_x + cos_yaw * goal_y)
        yaw_map = base_yaw + yaw_center

        self.recovery.on_detection(goal_map_x, goal_map_y, rospy.Time.now().to_sec())
        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = goal_map_x
        goal.target_pose.pose.position.y = goal_map_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)

        self.client.send_goal(goal)
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
            center_dist * (1- ratio),
            yaw_center,
            yaw_map,
        )

    @staticmethod
    def _as_float_seconds(stamp) -> float:
        """Normalize ROS time-like objects to float seconds."""
        if hasattr(stamp, "to_sec"):
            return float(stamp.to_sec())
        return float(stamp)

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

    @staticmethod
    def _compute_hs_hist(image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Compute normalized HS histogram for a BGR crop/image."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        # Prefer center region to reduce background impact.
        y1 = int(0.2 * h)
        y2 = int(0.85 * h)
        x1 = int(0.2 * w)
        x2 = int(0.8 * w)
        if y2 <= y1 or x2 <= x1:
            roi = image_bgr
        else:
            roi = image_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist

    @staticmethod
    def _transform_base_xyz_to_global(
        xyz_base: np.ndarray,
        base_trans: Tuple[float, float, float],
        base_rot,
    ) -> np.ndarray:
        xyz = np.asarray(xyz_base, dtype=np.float32)
        if xyz.shape[0] < 3 or not np.all(np.isfinite(xyz[:3])):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        rot_matrix = tf.transformations.quaternion_matrix(base_rot)[:3, :3]
        trans = np.asarray(base_trans, dtype=np.float32)
        return (rot_matrix @ xyz[:3].astype(np.float64) + trans.astype(np.float64)).astype(np.float32)

    def _get_payload(
        self,
        stamp_sec: float,
        frame_id: str,
        caption: str,
        box_xyxy: List[float],
        gdino_score: float,
        center: np.ndarray,
        center_global_xyz: np.ndarray,
        nearest_surface_xyz: np.ndarray,
        nearest_surface_global_xyz: np.ndarray,
        base_global_xyz: np.ndarray,
        num_points: int,
    ) -> Optional[dict]:
        """Build JSON payload for object center and nearest surface point."""
        center_arr = np.asarray(center, dtype=np.float32) if center is not None else None
        center_global_arr = (
            np.asarray(center_global_xyz, dtype=np.float32)
            if center_global_xyz is not None
            else None
        )
        nearest_arr = (
            np.asarray(nearest_surface_xyz, dtype=np.float32)
            if nearest_surface_xyz is not None
            else None
        )
        nearest_global_arr = (
            np.asarray(nearest_surface_global_xyz, dtype=np.float32)
            if nearest_surface_global_xyz is not None
            else None
        )
        base_global_arr = (
            np.asarray(base_global_xyz, dtype=np.float32)
            if base_global_xyz is not None
            else None
        )
        valid_center = (
            center_arr is not None
            and center_arr.shape[0] >= 3
            and np.all(np.isfinite(center_arr[:3]))
        )
        valid_center_global = (
            center_global_arr is not None
            and center_global_arr.shape[0] >= 3
            and np.all(np.isfinite(center_global_arr[:3]))
        )
        valid_nearest = (
            nearest_arr is not None
            and nearest_arr.shape[0] >= 3
            and np.all(np.isfinite(nearest_arr[:3]))
        )
        valid_nearest_global = (
            nearest_global_arr is not None
            and nearest_global_arr.shape[0] >= 3
            and np.all(np.isfinite(nearest_global_arr[:3]))
        )
        valid_base_global = (
            base_global_arr is not None
            and base_global_arr.shape[0] >= 3
            and np.all(np.isfinite(base_global_arr[:3]))
        )

        if (
            num_points < self.min_points
            or not valid_center
            or not valid_nearest
            or not valid_center_global
            or not valid_nearest_global
            or not valid_base_global
        ):
            return None

        nearest_dist = float(np.linalg.norm(nearest_global_arr[:3] - base_global_arr[:3]))
        return {
            "stamp": stamp_sec,
            "frame_id": frame_id,
            "base_frame": self.base_frame,
            "global_frame": self.goal_frame,
            "caption": caption,
            "bbox_xyxy": box_xyxy,
            "gdino_score": gdino_score,
            "num_points": int(num_points),
            "centroid_xyz_m": [
                float(center_global_arr[0]),
                float(center_global_arr[1]),
                float(center_global_arr[2]),
            ],
            "nearest_surface_xyz_m": [
                float(nearest_global_arr[0]),
                float(nearest_global_arr[1]),
                float(nearest_global_arr[2]),
            ],
            "nearest_surface_dist_m": nearest_dist,
        }
    
    def _job_too_old(self, frame_stamp_sec: float, stage: str) -> bool:
        """Drop outputs older than max_tolerate_delay when enabled (>0)."""
        if self.max_tolerate_delay <= 0.0:
            return False
        age = rospy.Time.now().to_sec() - frame_stamp_sec
        if age <= self.max_tolerate_delay:
            return False
        rospy.logwarn_throttle(
            1.0,
            "drop %s: age=%.3fs > max_tolerate_delay=%.3fs",
            stage,
            age,
            self.max_tolerate_delay,
        )
        return True

    def _worker_loop(self) -> None:
        """Consume latest queued jobs and run heavy GDINO+SAM+pointcloud processing."""
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                continue

            try:
                self._process_job(job)
            except Exception as exc:
                rospy.logerr("fusion worker failed: %s", str(exc))

    def _process_job(self, job: dict) -> None:
        """Process one frozen image+cloud job in worker thread."""
        image_msg: Image = job["image_msg"]
        cloud_msg: PointCloud2 = job["cloud_msg"]
        frame_stamp = job["frame_stamp"]
        frame_stamp_sec = float(job["frame_stamp_sec"])
        caption = job["caption"]
        cmd_seq = int(job["cmd_seq"])
        run_mode = TaskState(int(job["run_mode"]))
        queue_wait = rospy.Time.now().to_sec() - float(job["enqueue_wall_sec"])

        with self.state_lock:
            latest_cmd_seq = self.cmd_seq
        if cmd_seq != latest_cmd_seq:
            return

        if self._job_too_old(frame_stamp_sec, "queued job"):
            return

        try:
            self.tf_listener.waitForTransform(
                self.goal_frame,
                self.base_frame,
                frame_stamp,
                rospy.Duration(self.tf_timeout_s),
            )
            trans, rot = self.tf_listener.lookupTransform(
                self.goal_frame,
                self.base_frame,
                frame_stamp,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "drop frame by TF mismatch: %s <- %s at stamp=%.6f | %s",
                self.goal_frame,
                self.base_frame,
                self._as_float_seconds(frame_stamp),
                str(exc),
            )
            return

        base_yaw = float(tf.transformations.euler_from_quaternion(rot)[2])
        base_trans = (float(trans[0]), float(trans[1]), float(trans[2]))
        base_global_xyz = np.array([base_trans[0], base_trans[1], base_trans[2]], dtype=np.float32)
        image = camera_handle._ros_image_to_cv2_fallback(image_msg)

        detections, labels = self.detecte_model.predict(
            image=image,
            caption=caption
        )
        if len(detections.xyxy) == 0:
            rospy.logwarn("No detections for caption='%s'", caption)
            return

        conf = (
            np.asarray(detections.confidence, dtype=np.float32)
            if getattr(detections, "confidence", None) is not None
            else np.zeros((len(detections.xyxy),), dtype=np.float32)
        )
        best_idx = np.argmax(conf)
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
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        xyz = cloudpoints_handle._read_xyz(cloud_msg)
        if xyz.shape[0] == 0:
            rospy.logwarn("No points from %s", self.topic_points)
            return
        
        h, w = image.shape[:2]
        xyz_proj, uv, _ = points_project.project_lidar_to_image_with_fisheye_distortion(
            xyz_lidar=xyz,
            R_optical_lidar=self.R,
            t_optical_lidar=self.T,
            K_camera=self.K,
            width=w,
            height=h,
            dist_coeffs=self.D,
            undisort_points=self.points_undisort,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "no projected points inside image")
            return
      
        # uv can be very close to image border; clip after rounding to avoid OOB.
        u_inside = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, w - 1)
        v_inside = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, h - 1)

        on_object = mask[v_inside, u_inside]
        object_xyz = xyz_proj[on_object]

        if object_xyz.shape[0] < self.min_points:
            rospy.logwarn(
                "Object points too few: %d < min_points=%d",
                object_xyz.shape[0],
                self.min_points,
            )


        if object_xyz.shape[0] > 0:
            min_points_per_cell, min_cluster_points = self._cluster_params(object_xyz.shape[0])
            center, nearest_surface_xyz = cloudpoints_handle.cluster_3d_center_nearest_surface(
                object_xyz,
                grid_size=self.cluster_grid_size,
                z_grid_size=self.cluster_z_grid_size,
                min_points_per_cell=min_points_per_cell,
                min_cluster_points=min_cluster_points,
            )
        else:
            center = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            nearest_surface_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        center_base_xyz = np.array([-center[1], center[0], center[2]], dtype=np.float32)
        nearest_surface_base_xyz = np.array(
            [-nearest_surface_xyz[1], nearest_surface_xyz[0], nearest_surface_xyz[2]],
            dtype=np.float32,
        )
        center_global_xyz = self._transform_base_xyz_to_global(center_base_xyz, base_trans, rot)
        nearest_surface_global_xyz = self._transform_base_xyz_to_global(
            nearest_surface_base_xyz,
            base_trans,
            rot,
        )
        if self._job_too_old(frame_stamp_sec, "inference result"):
            return

        payload = self._get_payload(
            stamp_sec=self._as_float_seconds(image_msg.header.stamp),
            frame_id=cloud_msg.header.frame_id,
            caption=caption,
            box_xyxy=[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])],
            gdino_score=gdino_score,
            center=center_base_xyz,
            center_global_xyz=center_global_xyz,
            nearest_surface_xyz=nearest_surface_base_xyz,
            nearest_surface_global_xyz=nearest_surface_global_xyz,
            base_global_xyz=base_global_xyz,
            num_points=object_xyz.shape[0],
        )
        if payload is None:
            rospy.logwarn_throttle(1.0, "skip object publish: invalid object geometry")
            return

        with self.state_lock:
            current_run = self.run
            current_cmd_seq = self.cmd_seq

        if current_cmd_seq != cmd_seq:
            return

        if run_mode in (TaskState.Follow, TaskState.Recognize) and current_run != run_mode:
            return

        self.pub_depth_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        
        if self.enable_debug_overlay:
            annotated = self.detecte_model.annotate(image, detections, labels)
            debug = annotated
            debug[mask] = (debug[mask] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)

            line1 = f"center={payload['centroid_xyz_m']}"
            line2 = f"nearest={payload['nearest_surface_xyz_m']}"
            line3 = f"dist={payload['nearest_surface_dist_m']}"
            line4 = f"num_points={payload['num_points']} gdino_score={payload['gdino_score']}"
            x, y0, dy = 20, 40, 30
            font, scale, color, thick  = cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            cv2.putText(debug, line1, (x, y0 + 0 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line2, (x, y0 + 1 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line4, (x, y0 + 3 * dy), font, scale, color, thick, cv2.LINE_AA)
            self.pub_debug_image.publish(camera_handle._cv2_to_ros_image_fallback(debug, image_msg.header))
            self.pub_object_points.publish(cloudpoints_handle._build_cloud_xyz(cloud_msg.header, object_xyz))


        if (
            run_mode in (TaskState.Follow, TaskState.Follow_once)
            and np.all(np.isfinite(center_base_xyz[:2]))
            and np.all(np.isfinite(nearest_surface_base_xyz[:2]))
        ):
            self._send_follow_goal(
                center_xy=center_base_xyz[:2],
                surface_xy=nearest_surface_base_xyz[:2],
                base_trans=base_trans,
                base_yaw=base_yaw,
            )

    
    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        """Gate and enqueue synced frames; heavy compute runs only in worker thread."""
        lidar_sec = self._as_float_seconds(cloud_msg.header.stamp)
        image_sec = self._as_float_seconds(image_msg.header.stamp)
        frame_stamp = image_msg.header.stamp if lidar_sec < image_sec else cloud_msg.header.stamp
        frame_stamp_sec = image_sec if lidar_sec < image_sec else lidar_sec

        now_sec = rospy.Time.now().to_sec()
        if frame_stamp != rospy.Time(0):
            data_delay = now_sec - frame_stamp_sec
            if data_delay > self.max_lidarimage_delay:
                rospy.logwarn_throttle(
                    1.0,
                    "drop stale frame: age=%.3fs > max_lidarimage_delay=%.3fs",
                    data_delay,
                    self.max_lidarimage_delay,
                )
                return

        with self.state_lock:
            run_mode = self.run
            cmd_stamp = self.cmd_stamp
            cmd_seq = self.cmd_seq
            caption = self.detecte_model.caption

            if run_mode not in (
                TaskState.Follow_once,
                TaskState.Follow,
                TaskState.Recognize,
                TaskState.Recognize_once,
            ):
                return

            if cmd_stamp != rospy.Time(0) and frame_stamp != rospy.Time(0) and frame_stamp < cmd_stamp:
                return

            if run_mode in (TaskState.Follow, TaskState.Recognize) and self.max_infer_fps > 0.0:
                min_dt = 1.0 / self.max_infer_fps
                if self.last_infer_stamp_sec > 0.0 and (frame_stamp_sec - self.last_infer_stamp_sec) < min_dt:
                    return
                self.last_infer_stamp_sec = frame_stamp_sec

            if run_mode in (TaskState.Follow_once, TaskState.Recognize_once):
                self.run = TaskState.Notask
                self.recovery.clear()

        job = {
            "image_msg": image_msg,
            "cloud_msg": cloud_msg,
            "frame_stamp": frame_stamp,
            "frame_stamp_sec": frame_stamp_sec,
            "caption": caption,
            "cmd_seq": cmd_seq,
            "run_mode": int(run_mode),
            "enqueue_wall_sec": now_sec,
        }
        self._enqueue_latest_job(job)


def main() -> None:
    """ROS node entrypoint."""
    rospy.init_node("fusion_lidar_camera_node")
    FusionLidarCameraNode()
    rospy.spin()


if __name__ == "__main__":
    main()
