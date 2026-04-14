#!/usr/bin/env python3
import json
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


class FusionLidarCameraNode:
    def __init__(self) -> None:
        """Initialize model instances, ROS params, pubs/subs, and time synchronizer."""
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_visual_points = rospy.get_param("~topic_visual_points", "/visual_points/cam_front_lidar")
        self.caption = rospy.get_param("~caption", "black box")
        self.box_threshold = float(rospy.get_param("~box_threshold", 0.45))
        self.text_threshold = float(rospy.get_param("~text_threshold", 0.35))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.process_every_frame = bool(rospy.get_param("~process_every_frame", False))
        self.min_points = int(rospy.get_param("~min_points", 3))
        self.mask_dilate_px = int(rospy.get_param("~mask_dilate_px", 0))
        self.u_field = rospy.get_param("~u_field", "u")
        self.v_field = rospy.get_param("~v_field", "v")
        self.swap_uv = bool(rospy.get_param("~swap_uv", True))
        self.debug_max_points = int(rospy.get_param("~debug_max_points", 500))
        self.save_debug_images = bool(rospy.get_param("~save_debug_images", False))
        self.output_dir = rospy.get_param(
            "~output_dir",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_fusion"),
        )

        self.gdino_model = GroundingDINO()
        self.sam_model = Sam()

        self.pub_debug_image = rospy.Publisher("/fusion_lidar_camera/debug_image", Image, queue_size=1)
        self.pub_mask_image = rospy.Publisher("/fusion_lidar_camera/mask_image", Image, queue_size=1)
        self.pub_image_fov_points = rospy.Publisher(
            "/fusion_lidar_camera/image_fov_points",
            PointCloud2,
            queue_size=1,
        )
        self.pub_object_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)
        self.pub_depth_json = rospy.Publisher("/fusion_lidar_camera/object_depth_json", String, latch=True, queue_size=2)

        self.sub_image = Subscriber(self.topic_image, Image)
        self.sub_visual_points = Subscriber(self.topic_visual_points, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            fs=[self.sub_image, self.sub_visual_points],
            queue_size=5,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        if self.save_debug_images:
            os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo(
            "fusionLidarCamera ready: image=%s, points=%s, process_every_frame=%s",
            self.topic_image,
            self.topic_visual_points,
            str(self.process_every_frame),
        )

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
        if self.process_every_frame:
            return True

        run_once = bool(rospy.get_param("~run_once", False))
        if run_once:
            rospy.set_param("~run_once", False)
            return True
        return False

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

    def _read_xyzuv(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Read (x,y,z,u,v) points from PointCloud2 into an Nx5 float32 array."""
        u_name, v_name = self._resolve_uv_fields(cloud_msg)
        rows: List[Tuple[float, float, float, float, float]] = []
        for x, y, z, u, v in pc2.read_points(
            cloud_msg,
            field_names=("x", "y", "z", u_name, v_name),
            skip_nans=True,
        ):
            rows.append((float(x), float(y), float(z), float(u), float(v)))

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

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
    def _distance_stats_with_points(xyz: np.ndarray) -> dict:
        """Return min/max/median/mean origin-distance stats and corresponding xyz points."""
        if xyz.size == 0:
            return {
                "distance_to_origin_m": {
                    "min": None,
                    "max": None,
                    "median": None,
                    "mean": None,
                },
                "distance_point_xyz_m": {
                    "min": None,
                    "max": None,
                    "median_nearest": None,
                    "mean_nearest": None,
                },
            }

        ranges = np.linalg.norm(xyz, axis=1)
        idx_min = int(np.argmin(ranges))
        idx_max = int(np.argmax(ranges))
        median_val = float(np.median(ranges))
        mean_val = float(np.mean(ranges))
        idx_median_nearest = int(np.argmin(np.abs(ranges - median_val)))
        idx_mean_nearest = int(np.argmin(np.abs(ranges - mean_val)))

        def point_at(index: int) -> List[float]:
            return [float(xyz[index, 0]), float(xyz[index, 1]), float(xyz[index, 2])]

        return {
            "distance_to_origin_m": {
                "min": float(ranges[idx_min]),
                "max": float(ranges[idx_max]),
                "median": median_val,
                "mean": mean_val,
            },
            "distance_point_xyz_m": {
                "min": point_at(idx_min),
                "max": point_at(idx_max),
                "median_nearest": point_at(idx_median_nearest),
                "mean_nearest": point_at(idx_mean_nearest),
            },
        }

    def _compute_depth_payload(
        self,
        stamp_sec: float,
        frame_id: str,
        caption: str,
        box_xyxy: List[float],
        gdino_score: float,
        mask_pixels: int,
        xyz: np.ndarray,
    ) -> dict:
        """Compute robust depth statistics and pack them into a JSON-serializable payload."""
        dist_stats = self._distance_stats_with_points(xyz)
        if xyz.size == 0:
            return {
                "stamp": stamp_sec,
                "frame_id": frame_id,
                "caption": caption,
                "bbox_xyxy": box_xyxy,
                "gdino_score": gdino_score,
                "mask_pixels": mask_pixels,
                "num_points": 0,
                "depth_z_median_m": None,
                "depth_z_mean_m": None,
                "range_median_m": None,
                "range_min_m": None,
                "range_max_m": None,
                "range_mean_m": None,
                "centroid_xyz_m": None,
                "distance_to_origin_m": dist_stats["distance_to_origin_m"],
                "distance_point_xyz_m": dist_stats["distance_point_xyz_m"],
            }

        z = xyz[:, 2]
        ranges = np.linalg.norm(xyz, axis=1)
        centroid = np.median(xyz, axis=0)
        return {
            "stamp": stamp_sec,
            "frame_id": frame_id,
            "caption": caption,
            "bbox_xyxy": box_xyxy,
            "gdino_score": gdino_score,
            "mask_pixels": mask_pixels,
            "num_points": int(xyz.shape[0]),
            "depth_z_median_m": float(np.median(z)),
            "depth_z_mean_m": float(np.mean(z)),
            "range_median_m": float(np.median(ranges)),
            "range_min_m": float(np.min(ranges)),
            "range_max_m": float(np.max(ranges)),
            "range_mean_m": float(np.mean(ranges)),
            "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            "distance_to_origin_m": dist_stats["distance_to_origin_m"],
            "distance_point_xyz_m": dist_stats["distance_point_xyz_m"],
        }

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        """Run full fusion pipeline on synced image+visual-points messages."""
        if not self._should_process():
            return

        try:
            self.caption = rospy.get_param("~caption", self.caption)
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

            xyzuv = self._read_xyzuv(cloud_msg)
            if xyzuv.shape[0] == 0:
                rospy.logwarn("No points from %s", self.topic_visual_points)
                return

            h, w = mask.shape[:2]
            u, v = self._to_pixel_uv(xyzuv[:, 3:5])
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            inside_ratio = float(np.mean(inside)) if inside.size > 0 else 0.0
            xyzuv_inside = xyzuv[inside]
            u_inside = u[inside]
            v_inside = v[inside]
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

            self.pub_object_points.publish(self._build_cloud_xyz(cloud_msg.header, object_xyz))

            payload = self._compute_depth_payload(
                stamp_sec=self._as_float_seconds(image_msg.header.stamp),
                frame_id=cloud_msg.header.frame_id,
                caption=self.caption,
                box_xyxy=[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])],
                gdino_score=gdino_score,
                mask_pixels=int(mask.sum()),
                xyz=object_xyz,
            )
            self.pub_depth_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))

            annotated = self.gdino_model.annotate(image, detections, labels)
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

            debug = annotated.copy()
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

            line1 = f"depth_med={payload['depth_z_median_m']} range_min={payload['range_min_m']}"
            line2 = f"centroid_xyz={payload['centroid_xyz_m']}"
            line3 = f"range_max={payload['range_max_m']}"

            x, y0, dy = 20, 40, 30
            font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2

            cv2.putText(debug, line1, (x, y0 + 0 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line2, (x, y0 + 1 * dy), font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(debug, line3, (x, y0 + 2 * dy), font, scale, color, thick, cv2.LINE_AA)

            self.pub_debug_image.publish(self.cv2_to_ros_image(debug, image_msg.header))
            self.pub_mask_image.publish(self.cv2_to_ros_image(mask_bgr, image_msg.header))

            if self.save_debug_images:
                stamp_ns = str(image_msg.header.stamp.to_nsec())
                cv2.imwrite(os.path.join(self.output_dir, f"{stamp_ns}_debug.jpg"), debug)
                cv2.imwrite(os.path.join(self.output_dir, f"{stamp_ns}_mask.png"), mask_u8)

            rospy.loginfo(
                "fusion done: caption='%s' swap_uv=%s inside=%.3f obj=%d depth_med=%s frame=%s",
                self.caption,
                str(self.swap_uv),
                inside_ratio,
                object_xyz.shape[0],
                str(payload["depth_z_median_m"]),
                cloud_msg.header.frame_id,
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
