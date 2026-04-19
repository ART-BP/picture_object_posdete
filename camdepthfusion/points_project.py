#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
from typing import List

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField


# ---------------------- Hard-coded configuration ----------------------
TOPIC_IMAGE = "/camera/go2/front/image_raw"
TOPIC_LIDAR = "/lidar_points"
TOPIC_PROJECTED_CLOUD = "/camdepthfusion/projected_cloud"
TOPIC_PROJECTED_IMAGE = "/camdepthfusion/projected_image"

SYNC_QUEUE_SIZE = 2
SYNC_SLOP = 0.05
MIN_DEPTH = 0.05
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

# Kept for compatibility with your calibration set (not used in undistorted projection).
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
        -1.0,
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
    """Decode sensor_msgs/Image to BGR without relying on cv_bridge runtime libs."""
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


def _read_xyz(cloud_msg: PointCloud2) -> np.ndarray:
    """Vectorized PointCloud2 decode: return finite xyz in lidar frame."""
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
        raise ValueError(
            "PointCloud2 data too short: len(data)=%d expected>=%d" % (len(cloud_msg.data), expected_bytes)
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

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        return np.zeros((0, 3), dtype=np.float32)

    # Keep raw lidar coordinates for published xyz.
    xyz = np.stack((x[finite], y[finite], z[finite]), axis=1).astype(np.float32, copy=False)
    return xyz


def project_lidar_to_image(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    width: int,
    height: int,
    dist_coeffs: np.ndarray,
    min_depth: float,
):
    """Map lidar points to image pixels on undistorted image."""
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_cam = (R_optical_lidar @ xyz_lidar.T).T + t_optical_lidar.reshape(1, 3)
    valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > float(min_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar[valid]
    xyz_cam = xyz_cam[valid]

    # Keep D for compatibility; projection assumes undistorted image.
    _ = dist_coeffs
    uvw = (K_camera @ xyz_cam.T).T
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


def _build_projected_cloud_msg(header, xyz_lidar: np.ndarray, uv: np.ndarray) -> PointCloud2:
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("u", 12, PointField.FLOAT32, 1),
        PointField("v", 16, PointField.FLOAT32, 1),
    ]
    points = np.concatenate([xyz_lidar, uv], axis=1).astype(np.float32)
    return pc2.create_cloud(header, fields, points)


def _draw_overlay(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
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


class ProjectLidarNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_projected_cloud = rospy.Publisher(TOPIC_PROJECTED_CLOUD, PointCloud2, queue_size=1)
        self.pub_projected_image = rospy.Publisher(TOPIC_PROJECTED_IMAGE, Image, queue_size=1)

        self.sub_image = Subscriber(TOPIC_IMAGE, Image)
        self.sub_lidar = Subscriber(TOPIC_LIDAR, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_lidar],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo(
            "[project_lidar] ready: image=%s lidar=%s out_cloud=%s out_image=%s",
            TOPIC_IMAGE,
            TOPIC_LIDAR,
            TOPIC_PROJECTED_CLOUD,
            TOPIC_PROJECTED_IMAGE,
        )

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        try:
            try:
                image_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            except Exception:
                image_bgr = _ros_image_to_cv2_fallback(image_msg)

            xyz_lidar = _read_xyz(cloud_msg)
            if xyz_lidar.shape[0] <= 0:
                return

            h, w = image_bgr.shape[:2]
            # Apply axis remap only in projection chain, not in published xyz.
            R_for_projection = R @ AXIS_REMAP
            xyz_inside, uv_inside, depth_inside = project_lidar_to_image(
                xyz_lidar=xyz_lidar,
                R_optical_lidar=R_for_projection,
                t_optical_lidar=T,
                K_camera=K,
                width=w,
                height=h,
                dist_coeffs=D,
                min_depth=MIN_DEPTH,
            )
            if xyz_inside.shape[0] <= 0:
                return

            projected_cloud_msg = _build_projected_cloud_msg(cloud_msg.header, xyz_inside, uv_inside)
            self.pub_projected_cloud.publish(projected_cloud_msg)

            overlay = _draw_overlay(image_bgr=image_bgr, uv=uv_inside, depth=depth_inside)
            try:
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            except Exception:
                overlay_msg = Image()
                overlay_msg.height = int(overlay.shape[0])
                overlay_msg.width = int(overlay.shape[1])
                overlay_msg.encoding = "bgr8"
                overlay_msg.is_bigendian = False
                overlay_msg.step = int(overlay.shape[1] * 3)
                overlay_msg.data = np.ascontiguousarray(overlay, dtype=np.uint8).tobytes()

            overlay_msg.header = image_msg.header
            self.pub_projected_image.publish(overlay_msg)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "[project_lidar] callback failed: %s", str(exc))
            rospy.logerr_throttle(2.0, "[project_lidar] traceback:\n%s", traceback.format_exc())


def main():
    rospy.init_node("project_lidar")
    ProjectLidarNode()
    rospy.spin()


if __name__ == "__main__":
    main()
