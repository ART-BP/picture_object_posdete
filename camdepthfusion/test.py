#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from typing import List, Tuple

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from GroundingDINO.gdino import GroundingDINO
from MobileSAM.sam import Sam


# ---------------------- Hard-coded configuration ----------------------
TOPIC_IMAGE = "/camera/go2/front/image_raw"
TOPIC_LIDAR = "/lidar_points"
TOPIC_PROJECTED_CLOUD = "/camdepthfusion/test/projected_cloud"
TOPIC_OBJECT_CLOUD = "/camdepthfusion/test/object_cloud"
TOPIC_DEBUG_IMAGE = "/camdepthfusion/test/debug_image"

SYNC_QUEUE_SIZE = 2
SYNC_SLOP = 0.05
MAX_INFER_FPS = 0.0

CAPTION = "white column"
BOX_THRESHOLD = 0.55
TEXT_THRESHOLD = 0.85
MAX_DETECTIONS = 1

MIN_DEPTH = 0.05
MIN_OBJECT_POINTS = 10
MASK_DILATE_PX = 2

POINT_RADIUS = 2
MAX_OVERLAY_POINTS = 8000

K = np.array(
    [
        1166.271789008082,
        0.0,
        959.0530108632893,
        0.0,
        1167.7222552768242,
        544.1763261328632,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float64,
).reshape(3, 3)

D = np.array(
    [
        -0.1269125424187082,
        0.08472977776858055,
        0.36834095572628567,
        -0.27857771753696225,
        0.0,
    ],
    dtype=np.float64,
)

AXIS_REMAP = np.array(
    [
        0.0,
        -1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float64,
).reshape(3, 3)

R = np.array(
    [
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
).reshape(3, 3)

T = np.array([0.0, 0.0, 0.2], dtype=np.float64)
# ---------------------------------------------------------------------


def _ros_image_to_cv2_fallback(ros_image: Image) -> np.ndarray:
    """Decode ROS Image to BGR without relying on cv_bridge runtime libs."""
    h = int(ros_image.height)
    w = int(ros_image.width)
    step = int(ros_image.step)
    enc = (ros_image.encoding or "").lower()
    data = np.frombuffer(ros_image.data, dtype=np.uint8)

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size: h=%d w=%d" % (h, w))
    if step <= 0:
        raise ValueError("Invalid image step: %d" % step)
    if data.size < h * step:
        raise ValueError("Image data too short: bytes=%d expected>=%d" % (data.size, h * step))

    row_view = data[: h * step].reshape((h, step))
    if enc in ("bgr8", "rgb8"):
        row_bytes = w * 3
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        frame = row_view[:, :row_bytes].reshape((h, w, 3))
        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    if enc in ("bgra8", "rgba8"):
        row_bytes = w * 4
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        frame = row_view[:, :row_bytes].reshape((h, w, 4))
        if enc == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if enc in ("mono8", "8uc1"):
        row_bytes = w
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        gray = row_view[:, :row_bytes].reshape((h, w))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unsupported image encoding: %s" % ros_image.encoding)
    return decoded


def _cv2_to_ros_image_fallback(image_bgr: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height = int(image_bgr.shape[0])
    msg.width = int(image_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(image_bgr.shape[1] * 3)
    msg.data = np.ascontiguousarray(image_bgr, dtype=np.uint8).tobytes()
    return msg


def _read_xyz(cloud_msg: PointCloud2) -> np.ndarray:
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
    for name in ("x", "y", "z"):
        if name not in field_map:
            raise ValueError("PointCloud2 missing required field '%s'" % name)
        field = field_map[name]
        if int(field.count) != 1:
            raise ValueError("Field '%s' has count=%s, only scalar fields are supported" % (name, field.count))
        base_fmt = dtype_map.get(field.datatype)
        if base_fmt is None:
            raise ValueError("Unsupported PointField datatype=%s for field '%s'" % (field.datatype, name))
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
        return np.zeros((0, 3), dtype=np.float32)

    expected_bytes = int(cloud_msg.row_step) * height
    if len(cloud_msg.data) < expected_bytes:
        raise ValueError("PointCloud2 data too short: len(data)=%d expected>=%d" % (len(cloud_msg.data), expected_bytes))

    points = np.ndarray(
        shape=(height, width),
        dtype=point_dtype,
        buffer=cloud_msg.data,
        strides=(int(cloud_msg.row_step), int(cloud_msg.point_step)),
    )

    x = np.asarray(points["x"], dtype=np.float32).reshape(-1)
    y = np.asarray(points["y"], dtype=np.float32).reshape(-1)
    z = np.asarray(points["z"], dtype=np.float32).reshape(-1)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        return np.zeros((0, 3), dtype=np.float32)

    return np.stack((x[finite], y[finite], z[finite]), axis=1).astype(np.float32, copy=False)


def project_lidar_to_image(
    xyz_lidar: np.ndarray,
    r_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    k_camera: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_cam = (r_optical_lidar @ xyz_lidar.T).T + t_optical_lidar.reshape(1, 3)
    valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > float(min_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar[valid]
    xyz_cam = xyz_cam[valid]

    uvw = (k_camera @ xyz_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    if not np.any(inside):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_inside = xyz_lidar_valid[inside].astype(np.float32, copy=False)
    uv_inside = uv[inside].astype(np.float32, copy=False)
    depth_inside = xyz_cam[inside, 2].astype(np.float32, copy=False)
    return xyz_inside, uv_inside, depth_inside


def _build_cloud_xyzuv(header, xyz: np.ndarray, uv: np.ndarray) -> PointCloud2:
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("u", 12, PointField.FLOAT32, 1),
        PointField("v", 16, PointField.FLOAT32, 1),
    ]
    points = np.concatenate([xyz, uv], axis=1).astype(np.float32)
    return pc2.create_cloud(header, fields, points)


def _build_cloud_xyz(header, xyz: np.ndarray) -> PointCloud2:
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(header, fields, xyz.astype(np.float32).tolist())


def _draw_depth_overlay(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    count = int(uv.shape[0])
    if count <= 0:
        return overlay

    if count > MAX_OVERLAY_POINTS:
        idx = np.linspace(0, count - 1, MAX_OVERLAY_POINTS).astype(np.int32)
        uv = uv[idx]
        depth = depth[idx]

    min_d = float(np.min(depth))
    max_d = float(np.max(depth))
    denom = max(max_d - min_d, 1e-6)
    depth_norm = ((depth - min_d) / denom).astype(np.float32)
    color_idx = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    colors = cv2.applyColorMap(color_idx.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    uv_int = np.round(uv).astype(np.int32)
    for i in range(uv_int.shape[0]):
        u_i = int(uv_int[i, 0])
        v_i = int(uv_int[i, 1])
        c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(overlay, (u_i, v_i), int(POINT_RADIUS), c, -1, lineType=cv2.LINE_AA)

    return overlay


class LidarImageTester:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.last_infer_sec = 0.0

        self.gdino = GroundingDINO()
        self.gdino.setparameters(
            caption=CAPTION,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            return_labels=True,
            max_detections=MAX_DETECTIONS,
        )
        self.sam = Sam()

        self.pub_projected_cloud = rospy.Publisher(TOPIC_PROJECTED_CLOUD, PointCloud2, queue_size=1)
        self.pub_object_cloud = rospy.Publisher(TOPIC_OBJECT_CLOUD, PointCloud2, queue_size=1)
        self.pub_debug_image = rospy.Publisher(TOPIC_DEBUG_IMAGE, Image, queue_size=1)

        self.sub_image = Subscriber(TOPIC_IMAGE, Image)
        self.sub_lidar = Subscriber(TOPIC_LIDAR, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_lidar],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo(
            "test node ready: image=%s lidar=%s projected=%s object=%s debug=%s caption=%s",
            TOPIC_IMAGE,
            TOPIC_LIDAR,
            TOPIC_PROJECTED_CLOUD,
            TOPIC_OBJECT_CLOUD,
            TOPIC_DEBUG_IMAGE,
            CAPTION,
        )
        rospy.loginfo("AXIS_REMAP(row-major)=%s det=%.3f", AXIS_REMAP.reshape(-1).tolist(), float(np.linalg.det(AXIS_REMAP)))

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        try:
            now_sec = rospy.Time.now().to_sec()
            if MAX_INFER_FPS > 0.0:
                min_dt = 1.0 / MAX_INFER_FPS
                if self.last_infer_sec > 0.0 and (now_sec - self.last_infer_sec) < min_dt:
                    return
                self.last_infer_sec = now_sec

            try:
                image_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            except Exception:
                image_bgr = _ros_image_to_cv2_fallback(image_msg)

            xyz_raw = _read_xyz(cloud_msg)
            if xyz_raw.shape[0] == 0:
                return

            # Publish and project with axis-remapped lidar coordinates.
            xyz_lidar = (AXIS_REMAP @ xyz_raw.T).T
            h, w = image_bgr.shape[:2]
            xyz_inside, uv_inside, depth_inside = project_lidar_to_image(
                xyz_lidar=xyz_lidar,
                r_optical_lidar=R,
                t_optical_lidar=T,
                k_camera=K,
                width=w,
                height=h,
                min_depth=MIN_DEPTH,
            )

            projected_cloud = _build_cloud_xyzuv(cloud_msg.header, xyz_inside, uv_inside)
            self.pub_projected_cloud.publish(projected_cloud)

            if xyz_inside.shape[0] == 0:
                empty_object = np.zeros((0, 3), dtype=np.float32)
                self.pub_object_cloud.publish(_build_cloud_xyz(cloud_msg.header, empty_object))
                return

            detections, labels = self.gdino.predict(image=image_bgr, caption=CAPTION)
            if detections is None or len(detections.xyxy) == 0:
                empty_object = np.zeros((0, 3), dtype=np.float32)
                self.pub_object_cloud.publish(_build_cloud_xyz(cloud_msg.header, empty_object))
                return

            conf = (
                np.asarray(detections.confidence, dtype=np.float32)
                if getattr(detections, "confidence", None) is not None
                else np.zeros((len(detections.xyxy),), dtype=np.float32)
            )
            best_idx = int(np.argmax(conf))
            box_xyxy = detections.xyxy[best_idx]
            label = labels[best_idx] if labels is not None and best_idx < len(labels) else CAPTION
            score = float(conf[best_idx]) if best_idx < len(conf) else 0.0

            mask, _, _ = self.sam.get_mask_by_box(
                box_xyxy=box_xyxy,
                image=image_bgr,
                image_format="BGR",
                multimask_output=False,
            )

            if MASK_DILATE_PX > 0:
                k = MASK_DILATE_PX * 2 + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            u_inside = np.rint(uv_inside[:, 0]).astype(np.int32)
            v_inside = np.rint(uv_inside[:, 1]).astype(np.int32)
            valid_pix = (u_inside >= 0) & (u_inside < w) & (v_inside >= 0) & (v_inside < h)
            on_object = np.zeros((uv_inside.shape[0],), dtype=bool)
            on_object[valid_pix] = mask[v_inside[valid_pix], u_inside[valid_pix]]

            object_xyz = xyz_inside[on_object]
            if object_xyz.shape[0] < MIN_OBJECT_POINTS:
                object_xyz = np.zeros((0, 3), dtype=np.float32)

            self.pub_object_cloud.publish(_build_cloud_xyz(cloud_msg.header, object_xyz))

            debug = _draw_depth_overlay(image_bgr, uv_inside, depth_inside)
            if mask is not None:
                debug[mask] = (debug[mask] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)

            x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

            text = "%s %.2f obj_pts=%d in_img=%d" % (label, score, int(object_xyz.shape[0]), int(xyz_inside.shape[0]))
            cv2.putText(debug, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            sample_uv = np.stack((u_inside[on_object], v_inside[on_object]), axis=1) if np.any(on_object) else np.zeros((0, 2), dtype=np.int32)
            if sample_uv.shape[0] > 0:
                step = max(1, sample_uv.shape[0] // 300)
                for uu, vv in sample_uv[::step]:
                    cv2.circle(debug, (int(uu), int(vv)), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            except Exception:
                debug_msg = _cv2_to_ros_image_fallback(debug, image_msg.header)
            debug_msg.header = image_msg.header
            self.pub_debug_image.publish(debug_msg)

            rospy.loginfo_throttle(
                1.0,
                "test: projected=%d object=%d caption='%s'",
                int(xyz_inside.shape[0]),
                int(object_xyz.shape[0]),
                CAPTION,
            )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "test callback failed: %s", str(exc))
            rospy.logerr_throttle(2.0, "traceback:\n%s", traceback.format_exc())


def main() -> None:
    rospy.init_node("lidar_image_test_node")
    LidarImageTester()
    rospy.spin()


if __name__ == "__main__":
    main()
