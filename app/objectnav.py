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

from camdepthfusion import points_project
from camdepthfusion import cloudpoints_handle
from camdepthfusion import camera_handle
from recovery import RecoveryAction, RecoveryController


class TaskState(IntEnum):
    Notask = 0
    Follow = 1
    Recognize = 2
    Follow_once = 3


class FusionLidarCameraNode:
    @staticmethod
    def _as_bool(value: str) -> bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @classmethod
    def _cfg_get(cls, cfg: configparser.SectionProxy, key: str, default, cast):
        raw = cfg.get(key, fallback=None)
        if raw is None or raw == "":
            return default
        try:
            if cast is bool:
                return cls._as_bool(raw)
            return cast(raw)
        except Exception as exc:
            raise ValueError(f"Invalid config value for '{key}': {raw}") from exc

    @staticmethod
    def _load_runtime_config() -> configparser.SectionProxy:
        default_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg")
        config_path = os.environ.get("OBJECTNAV_CONFIG", default_cfg_path)

        parser = configparser.ConfigParser()
        loaded = parser.read(config_path, encoding="utf-8")
        if not loaded:
            raise FileNotFoundError(
                f"Cannot read config file: {config_path}. "
                "Set OBJECTNAV_CONFIG or create app/config.cfg."
            )
        if "objectnav" not in parser:
            raise KeyError(
                f"Missing [objectnav] section in config file: {config_path}"
            )
        rospy.loginfo("Loaded runtime config: %s", config_path)
        return parser["objectnav"]

    def __init__(self) -> None:
        """Initialize models, runtime config, pubs/subs, and time synchronizer."""
        cfg = self._load_runtime_config()

        default_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output_fusion",
        )

        self.topic_image = self._cfg_get(cfg, "topic_image", "/camera/go2/front/image_raw", str)
        self.topic_visual_points = self._cfg_get(cfg, "topic_visual_points", "/visual_points/cam_front_lidar", str)
        self.topic_points = self._cfg_get(cfg, "topic_points", "/lidar_points", str)

        caption = self._cfg_get(cfg, "caption", "black box", str)
        box_threshold = self._cfg_get(cfg, "box_threshold", 0.55, float)
        text_threshold = self._cfg_get(cfg, "text_threshold", 0.85, float)

        self.sync_slop = self._cfg_get(cfg, "sync_slop", 0.05, float)
        self.sync_queue_size = self._cfg_get(cfg, "sync_queue_size", 1, int)
        self.min_points = self._cfg_get(cfg, "min_points", 5, int)
        self.mask_dilate_px = self._cfg_get(cfg, "mask_dilate_px", 2, int)
        self.u_field = self._cfg_get(cfg, "u_field", "u", str)
        self.v_field = self._cfg_get(cfg, "v_field", "v", str)
        self.cluster_grid_size = self._cfg_get(cfg, "cluster_grid_size", 0.2, float)

        self.cluster_dense_threshold = self._cfg_get(cfg, "cluster_dense_threshold", 100, int)
        self.cluster_min_cell_sparse = self._cfg_get(cfg, "cluster_min_cell_sparse", 2, int)
        self.cluster_min_points_sparse = self._cfg_get(cfg, "cluster_min_points_sparse", 5, int)
        self.cluster_min_cell_dense = self._cfg_get(cfg, "cluster_min_cell_dense", 5, int)
        self.cluster_min_points_dense = self._cfg_get(cfg, "cluster_min_points_dense", 10, int)
        self.min_goal_dist_m = self._cfg_get(cfg, "min_goal_dist_m", 0.5, float)
        self.goal_frame = self._cfg_get(cfg, "goal_frame", "map", str)
        self.base_frame = self._cfg_get(cfg, "base_frame", "base_link", str)
        self.tf_timeout_s = self._cfg_get(cfg, "tf_timeout_s", 0.03, float)
        self.max_lidarimage_delay = self._cfg_get(cfg, "max_lidarimage_delay", 0.6, float)
        self.max_tolerate_delay = self._cfg_get(cfg, "max_tolerate_delay", 0.0, float)
        self.max_infer_fps = self._cfg_get(cfg, "max_infer_fps", 1.0, float)
        self.enable_debug_overlay = self._cfg_get(cfg, "enable_debug_overlay", True, bool)
        self.use_bbox_mask_only = self._cfg_get(cfg, "use_bbox_mask_only", False, bool)
        self.debug_max_points = self._cfg_get(cfg, "debug_max_points", 500, int)
        self.save_debug_images = self._cfg_get(cfg, "save_debug_images", False, bool)
        self.output_dir = self._cfg_get(cfg, "output_dir", default_output_dir, str)
        self.recovery_move_after_s = self._cfg_get(cfg, "recovery_move_after_s", 3.0, float)
        self.recovery_rotate_after_s = self._cfg_get(cfg, "recovery_rotate_after_s", 5.0, float)
        self.recovery_cancel_after_s = self._cfg_get(cfg, "recovery_cancel_after_s", 10.0, float)
        self.recovery_forward_dist_m = self._cfg_get(cfg, "recovery_forward_dist_m", 2.0, float)
        self.recovery_rotate_deg = self._cfg_get(cfg, "recovery_rotate_deg", 90.0, float)
        self.recovery_tick_hz = self._cfg_get(cfg, "recovery_tick_hz", 2.0, float)
        
        self.run = TaskState.Notask
        self.cmd_stamp = rospy.Time(0)
        self.cmd_seq = 0
        self.last_infer_stamp_sec = 0.0
        self.last_time = rospy.Time.now().to_sec()
        self.state_lock = threading.Lock()
        self.recovery = RecoveryController(
            move_after_sec=self.recovery_move_after_s,
            rotate_after_sec=self.recovery_rotate_after_s,
            cancel_after_sec=self.recovery_cancel_after_s,
        )


        select = self._cfg_get(cfg, "model", "gdino", str).strip().lower()
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

        self.pub_debug_image = rospy.Publisher("/fusion_lidar_camera/debug_image", Image, queue_size=1, latch=True)
        self.pub_object_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)
        self.pub_depth_json = rospy.Publisher("/fusion_lidar_camera/object_depth_json", String, latch=True, queue_size=2)

        #  注册雷达和图像同步
        sub_image = Subscriber(self.topic_image, Image)
        sub_visual_points = Subscriber(self.topic_visual_points, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            fs=[sub_image, sub_visual_points],
            queue_size=max(1, self.sync_queue_size),
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
        rospy.on_shutdown(self._on_shutdown)

        if self.save_debug_images:
            os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo(
            "fusionLidarCamera ready: image=%s, points=%s, sync_queue=%d, sync_slop=%.3f, max_infer_fps=%.2f, "
            "debug=%s, bbox_mask_only=%s, max_lidarimage_delay=%.3f, max_tolerate_delay=%.3f, model_id=%s",
            self.topic_image,
            self.topic_visual_points,
            max(1, self.sync_queue_size),
            self.sync_slop,
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
            rospy.logwarn("Invalid object_cmd JSON: %s, raw=%s", str(exc), msg.data)
            return

        now = rospy.Time.now()
        now_sec = now.to_sec()
        task = str(tmsg.get("task", "none")).lower()
        with self.state_lock:
            caption = tmsg.get("caption", self.detecte_model.caption)
            self.cmd_seq += 1

            self.detecte_model.setparameters(caption=caption)
            if task == "follow":
                self.run = TaskState.Follow
                self.cmd_stamp = now
            elif task == "recognition":
                self.run = TaskState.Recognize
                self.cmd_stamp = now
            elif task == "follow_once":
                self.run = TaskState.Follow_once
                self.cmd_stamp = now
            else:
                self.run = TaskState.Notask

            cmd_stamp = self.cmd_stamp
            cmd_seq = self.cmd_seq

        self.recovery.on_task(task=task, now_sec=now_sec)

        if task in ("follow", "recognition", "follow_once"):
            self._ensure_worker_started()

        if task == "cancel":
            self.client.cancel_goal()
            self._clear_pending_jobs()

        self.last_time = now_sec
        rospy.loginfo(
            "Received cmd: caption=%s task=%s  cmd_stamp=%.6f cmd_seq=%d",
            caption,
            task,
            self._as_float_seconds(cmd_stamp),
            cmd_seq,
        )

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

        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = goal_map_x
        goal.target_pose.pose.position.y = goal_map_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)

        self.client.send_goal(goal)
        # rospy.loginfo(
        #     "sent follow goal (%s): goal=[%.3f, %.3f] center_xy=[%.3f, %.3f] surface_xy=[%.3f, %.3f] "
        #     "surface_dist=%.3f follow_distance=%.3f yaw_rel=%.3f yaw_map=%.3f",
        #     self.goal_frame,
        #     goal_map_x,
        #     goal_map_y,
        #     center[0],
        #     center[1],
        #     surface[0],
        #     surface[1],
        #     surface_dist,
        #     center_dist * (1- ratio),
        #     yaw_center,
        #     yaw_map,
        # )

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
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        xyzuv_inside = cloudpoints_handle._read_xyzuv(cloud_msg)
        if xyzuv_inside.shape[0] == 0:
            rospy.logwarn("No points from %s", self.topic_visual_points)
            return

        u_inside = np.rint(xyzuv_inside[:, 3]).astype(np.int16)
        v_inside = np.rint(xyzuv_inside[:, 4]).astype(np.int16)

        on_object = mask[u_inside, v_inside]
        object_xyzuv = xyzuv_inside[on_object]
        object_xyz = object_xyzuv[:, :3] if object_xyzuv.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

        if object_xyz.shape[0] < self.min_points:
            rospy.logwarn(
                "Object points too few: %d < min_points=%d",
                object_xyz.shape[0],
                self.min_points,
            )


        if object_xyz.shape[0] > 0:
            min_points_per_cell, min_cluster_points = self._cluster_params(object_xyz.shape[0])
            center, nearest_surface_xy = cloudpoints_handle.cluster_2d_center_nearest_surface(
                object_xyz[:, :2],
                grid_size=self.cluster_grid_size,
                min_points_per_cell=min_points_per_cell,
                min_cluster_points=min_cluster_points,
            )
        else:
            center = np.array([np.nan, np.nan], dtype=np.float32)
            nearest_surface_xy = np.array([np.nan, np.nan], dtype=np.float32)

        if self._job_too_old(frame_stamp_sec, "inference result"):
            return

        payload = self._get_payload(
            stamp_sec=self._as_float_seconds(image_msg.header.stamp),
            frame_id=cloud_msg.header.frame_id,
            caption=caption,
            box_xyxy=[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])],
            gdino_score=gdino_score,
            center=center,
            nearest_surface_xy=nearest_surface_xy,
            num_points=object_xyz.shape[0],
        )
        self.pub_depth_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if payload["centroid_xy_m"] is not None:
            center_xy = payload["centroid_xy_m"]
            self.recovery.on_detection(
                center_xy=(float(center_xy[0]), float(center_xy[1])),
                now_sec=rospy.Time.now().to_sec(),
            )

        if self.enable_debug_overlay:
            annotated = self.detecte_model.annotate(image, detections, labels)
            debug = annotated
            debug[mask] = (debug[mask] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)

            line1 = f"center={payload['centroid_xy_m']}"
            line2 = f"nearest={payload['nearest_surface_xy_m']}"
            line3 = f"num_points={payload['num_points']}"
            line4 = f"gdino_score={payload['gdino_score']}"
            x, y0, dy = 20, 40, 30
            font, scale, color, thick  = cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            cv2.putText(debug, line1, (x, y0 + 0 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line2, (x, y0 + 1 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line4, (x, y0 + 3 * dy), font, scale, color, thick, cv2.LINE_AA)
            self.pub_debug_image.publish(camera_handle._cv2_to_ros_image_fallback(debug, image_msg.header))
            self.pub_object_points.publish(cloudpoints_handle._build_cloud_xyz(cloud_msg.header, object_xyz))


        if payload["centroid_xy_m"] is not None and payload["nearest_surface_xy_m"] is not None:
            self._send_follow_goal(
                center_xy=payload["centroid_xy_m"],
                surface_xy=payload["nearest_surface_xy_m"],
                base_trans=(float(trans[0]), float(trans[1]), float(trans[2])),
                base_yaw=base_yaw,
            )
        rospy.loginfo("labels: %s  conf: %f", labels[0], gdino_score)

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

            if run_mode not in (TaskState.Follow_once, TaskState.Follow):
                return

            if cmd_stamp != rospy.Time(0) and frame_stamp != rospy.Time(0) and frame_stamp < cmd_stamp:
                return

            if run_mode == TaskState.Follow and self.max_infer_fps > 0.0:
                min_dt = 1.0 / self.max_infer_fps
                if self.last_infer_stamp_sec > 0.0 and (frame_stamp_sec - self.last_infer_stamp_sec) < min_dt:
                    return
                self.last_infer_stamp_sec = frame_stamp_sec

            if run_mode == TaskState.Recognize or run_mode == TaskState.Follow_once:
                self.run = TaskState.Notask

        job = {
            "image_msg": image_msg,
            "cloud_msg": cloud_msg,
            "frame_stamp": frame_stamp,
            "frame_stamp_sec": frame_stamp_sec,
            "caption": caption,
            "cmd_seq": cmd_seq,
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
