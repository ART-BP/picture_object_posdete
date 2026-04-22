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

from camdepthfusion import points_project
from camdepthfusion import cloudpoints_handle
from camdepthfusion import camera_handle

NOTASK = 0
FOLLOW = 1
RECON = 2


class FusionLidarCameraNode:
    def __init__(self) -> None:
        """Initialize model instances, ROS params, pubs/subs, and time synchronizer."""
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_visual_points = rospy.get_param("~topic_visual_points", "/visual_points/cam_front_lidar")

        # Detection params
        caption = rospy.get_param("~caption", "black box")
        box_threshold = float(rospy.get_param("~box_threshold", 0.45))
        text_threshold = float(rospy.get_param("~text_threshold", 0.35))


        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 2))
        self.min_points = int(rospy.get_param("~min_points", 5))
        self.mask_dilate_px = int(rospy.get_param("~mask_dilate_px", 0))
        self.u_field = rospy.get_param("~u_field", "u")
        self.v_field = rospy.get_param("~v_field", "v")
        self.swap_uv = bool(rospy.get_param("~swap_uv", True))
        self.cluster_grid_size = float(rospy.get_param("~cluster_grid_size", 0.2))

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
        self.max_lidarimage_delay = float(rospy.get_param("~max_lidarimage_delay", 0.6))
        self.max_infer_fps = float(rospy.get_param("~max_infer_fps", 0.0))
        self.enable_debug_overlay = bool(rospy.get_param("~enable_debug_overlay", True))
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
        self.gdino_model.setparameters(caption=caption, box_threshold=box_threshold, 
                                       text_threshold=text_threshold,
                                       return_labels=self.enable_debug_overlay)
        self.sam_model = Sam()
        self.tf_listener = tf.TransformListener()

        self.pub_debug_image = rospy.Publisher("/fusion_lidar_camera/debug_image", Image, queue_size=1, latch=True)
        self.pub_object_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)
        self.pub_depth_json = rospy.Publisher("/fusion_lidar_camera/object_depth_json", String, latch=True, queue_size=2)

        #  注册雷达和图像同步
        self.sub_image = Subscriber(self.topic_image, Image)
        self.sub_visual_points = Subscriber(self.topic_visual_points, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            fs=[self.sub_image, self.sub_visual_points],
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

        if self.save_debug_images:
            os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo(
            "fusionLidarCamera ready: image=%s, points=%s, sync_queue=%d, sync_slop=%.3f, max_infer_fps=%.2f, debug=%s, bbox_mask_only=%s, max_lidarimage_delay=%.3f",
            self.topic_image,
            self.topic_visual_points,
            max(1, self.sync_queue_size),
            self.sync_slop,
            self.max_infer_fps,
            str(self.enable_debug_overlay),
            str(self.use_bbox_mask_only),
            self.max_lidarimage_delay,
        )

    def cmd_callback(self, msg: String) -> None:
        """Parse task command JSON and switch node running mode."""
        try:
            tmsg = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("Invalid object_cmd JSON: %s, raw=%s", str(exc), msg.data)
            return

        caption = tmsg.get("caption", self.gdino_model.caption)
        self.gdino_model.setparameters(caption=caption)
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
        self.last_time = rospy.Time.now().to_sec()
        rospy.loginfo(
            "Received cmd: caption=%s task=%s run=%d cmd_stamp=%.6f",
            caption,
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

        cos_yaw = math.cos(self.base_yaw)
        sin_yaw = math.sin(self.base_yaw)

        goal_map_x = float(self.trans[0] + cos_yaw * goal_x - sin_yaw * goal_y)
        goal_map_y = float(self.trans[1] + sin_yaw * goal_x + cos_yaw * goal_y)
        yaw_map = self.base_yaw + yaw_center

        q = quaternion_from_euler(0.0, 0.0, yaw_map)

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = goal_map_x
        goal.target_pose.pose.position.y = goal_map_y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = Quaternion(*q)

        self.client.send_goal(goal)
        now = rospy.Time.now().to_sec()
        rospy.loginfo("----------delta_time: %f", now - self.last_time)
        self.last_time = now
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
        
        # image_time = self._as_float_seconds(image_msg.header.stamp)
        # cloud_time = self._as_float_seconds(cloud_msg.header.stamp)
        # rospy.loginfo("image_time: %f,  cloud_time: %f, delta_time: %f", image_time, cloud_time, image_time - cloud_time)

        lidar_sec = self._as_float_seconds(cloud_msg.header.stamp)
        image_sec = self._as_float_seconds(image_msg.header.stamp)
        frame_stamp = image_msg.header.stamp if lidar_sec < image_sec else cloud_msg.header.stamp
        frame_stamp_sec = image_sec if lidar_sec < image_sec else lidar_sec

        # Strict temporal alignment: stale frames are dropped directly.
        if frame_stamp != rospy.Time(0):
            data_delay = rospy.Time.now().to_sec() - frame_stamp_sec
            if data_delay > self.max_lidarimage_delay:
                rospy.logwarn_throttle(
                    1.0,
                    "drop stale frame: age=%.3fs > max_lidarimage_delay=%.3fs",
                    data_delay,
                    self.max_lidarimage_delay,
                )
                return

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

            self.base_yaw = float(tf.transformations.euler_from_quaternion(rot)[2])


            image = camera_handle._ros_image_to_cv2_fallback(image_msg)
            
            caption = self.gdino_model.caption
            detections, labels = self.gdino_model.predict(
                image=image,
                caption=caption,
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
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            # # 1080, 1920
            # h, w = mask.shape[:2]
            # rospy.loginfo("h: %d, w: %d", h, w)

            xyzuv_inside = cloudpoints_handle._read_xyzuv(cloud_msg)
            if xyzuv_inside.shape[0] == 0:
                rospy.logwarn("No points from %s", self.topic_visual_points)
                return

            u_inside = np.rint(xyzuv_inside[:, 3]).astype(np.int16)
            v_inside = np.rint(xyzuv_inside[:, 4]).astype(np.int16)

            if xyzuv_inside.shape[0] == 0:
                rospy.logwarn("No projected points inside image bounds")
                return


            on_object = mask[u_inside, v_inside]
            object_xyzuv = xyzuv_inside[on_object]
            object_xyz = object_xyzuv[:, :3] if object_xyzuv.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

            if object_xyz.shape[0] < self.min_points:
                rospy.logwarn(
                    "Object points too few: %d < min_points=%d",
                    object_xyz.shape[0],
                    self.min_points,
                )
            inside_ratio = float(object_xyzuv.shape[0]) / float(xyzuv_inside.shape[0])
            self.pub_object_points.publish(cloudpoints_handle._build_cloud_xyz(cloud_msg.header, object_xyz))

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


            # debug
            if self.enable_debug_overlay:
                annotated = self.gdino_model.annotate(image, detections, labels)
                debug = annotated
                debug[mask] = (debug[mask] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)

                # if object_xyzuv.shape[0] > 0:
                #     sample_step = max(1, object_xyzuv.shape[0] // self.debug_max_points)
                #     obj_u = u_inside[on_object]
                #     obj_v = v_inside[on_object]
                #     sample_uv = np.stack([obj_u, obj_v], axis=1)[::sample_step].astype(np.int32)
                #     for uu, vv in sample_uv:
                #         cv2.circle(debug, (int(uu), int(vv)), 1, (0, 0, 255), -1)

                line1 = f"center={payload['centroid_xy_m']}"
                line2 = f"nearest={payload['nearest_surface_xy_m']}"
                line3 = f"num_points={payload['num_points']}"
                line4 = f"gdino_score={payload['gdino_score']}"
                x, y0, dy = 20, 40, 30
                font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                cv2.putText(debug, line1, (x, y0 + 0 * dy), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(debug, line2, (x, y0 + 1 * dy), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)
                self.pub_debug_image.publish(camera_handle._cv2_to_ros_image_fallback(debug, image_msg.header))

                if self.save_debug_images:
                    stamp_ns = str(image_msg.header.stamp.to_nsec())
                    cv2.imwrite(os.path.join(self.output_dir, f"{stamp_ns}_debug.jpg"), debug)

            rospy.loginfo(
                "fusion done: caption='%s' inside=%.3f obj=%d gdino_score=%.2f",
                caption,
                inside_ratio,
                object_xyz.shape[0],
                gdino_score
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
