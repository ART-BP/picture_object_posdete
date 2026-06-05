#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField


MIN_DEPTH = 0.1
MAX_OVERLAY_POINTS = 20000
POINT_RADIUS = 1

# Fill these values with your calibrated pinhole camera intrinsics/extrinsics.
# K_PINHOLE maps camera optical coordinates to pixels.
K_PINHOLE = np.array(
    [
        [598.0689565033549, 0.0, 329.6593672355304],
        [0.0, 597.6070441110179, 251.03062290746962],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# R_OPTICAL_LIDAR and T_OPTICAL_LIDAR transform lidar points into the camera
# optical frame: xyz_cam = R_OPTICAL_LIDAR @ xyz_lidar + T_OPTICAL_LIDAR.
R_OPTICAL_LIDAR = np.array(
    [
        [-0.014907, -0.999863, -0.007248],
        [-0.009365, 0.007388, -0.999929],
        [0.999845, -0.014838, -0.009474],
    ],
    dtype=np.float64,
)

T_OPTICAL_LIDAR = np.array([0.025103, -0.192104, -0.042620], dtype=np.float64)


def ros_image_to_cv2_bgr(ros_image: Image) -> np.ndarray:
    """Decode ROS Image to BGR without cv_bridge."""
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
        raise ValueError(
            "Image data too short: bytes=%d expected>=%d" % (data.size, h * step)
        )

    row_view = data[: h * step].reshape((h, step))
    if enc in ("bgr8", "rgb8"):
        row_bytes = w * 3
        if step < row_bytes:
            raise ValueError(
                "Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes)
            )
        frame = row_view[:, :row_bytes].reshape((h, w, 3))
        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    if enc in ("bgra8", "rgba8"):
        row_bytes = w * 4
        if step < row_bytes:
            raise ValueError(
                "Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes)
            )
        frame = row_view[:, :row_bytes].reshape((h, w, 4))
        if enc == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if enc in ("mono8", "8uc1"):
        row_bytes = w
        if step < row_bytes:
            raise ValueError(
                "Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes)
            )
        gray = row_view[:, :row_bytes].reshape((h, w))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unsupported image encoding: %s" % ros_image.encoding)
    return decoded


def cv2_bgr_to_ros_image(image_bgr: np.ndarray, header) -> Image:
    """Encode BGR image as sensor_msgs/Image."""
    msg = Image()
    msg.header = header
    msg.height = int(image_bgr.shape[0])
    msg.width = int(image_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(image_bgr.shape[1] * 3)
    msg.data = np.ascontiguousarray(image_bgr, dtype=np.uint8).tobytes()
    return msg


def read_xyz(cloud_msg: PointCloud2) -> np.ndarray:
    """Read finite x/y/z fields from PointCloud2."""
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
            raise ValueError(
                "Field '%s' has count=%s, only scalar fields are supported"
                % (name, field.count)
            )
        base_fmt = dtype_map.get(field.datatype)
        if base_fmt is None:
            raise ValueError(
                "Unsupported PointField datatype=%s for field '%s'"
                % (field.datatype, name)
            )
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
            "PointCloud2 data too short: len(data)=%d expected>=%d"
            % (len(cloud_msg.data), expected_bytes)
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

    return np.stack((x[finite], y[finite], z[finite]), axis=1).astype(
        np.float32, copy=False
    )


def build_cloud_xyzuv(header, xyz: np.ndarray, uv: np.ndarray) -> PointCloud2:
    """Build PointCloud2 with x/y/z/u/v float32 fields."""
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("u", 12, PointField.FLOAT32, 1),
        PointField("v", 16, PointField.FLOAT32, 1),
    ]
    points = np.concatenate([xyz, uv], axis=1).astype(np.float32)
    return pc2.create_cloud(header, fields, points)


def draw_overlay(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Draw projected points on image, colored by camera depth."""
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
    color_idx = np.maximum(255.0 * 5.0 * (1.0 - depth_norm), 0).astype(np.uint8)
    colors = cv2.applyColorMap(color_idx.reshape(-1, 1), cv2.COLORMAP_JET).reshape(
        -1, 3
    )

    uv_int = np.round(uv).astype(np.int32)
    for i in range(uv_int.shape[0]):
        u_i = int(uv_int[i, 0])
        v_i = int(uv_int[i, 1])
        c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(overlay, (u_i, v_i), int(POINT_RADIUS), c, -1, lineType=cv2.LINE_AA)

    return overlay


def project_lidar_to_pinhole_image(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
):
    """Project lidar points to raw pinhole image pixels with zero distortion."""
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    R_use = np.asarray(R_optical_lidar, dtype=np.float64).reshape(3, 3)
    t_use = np.asarray(t_optical_lidar, dtype=np.float64).reshape(1, 3)
    K_use = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)

    xyz = np.asarray(xyz_lidar, dtype=np.float64).reshape(-1, 3)
    xyz_cam = (R_use @ xyz.T).T + t_use
    in_front = xyz_cam[:, 2] > float(min_depth)
    if not np.any(in_front):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_valid = xyz[in_front]
    xyz_cam_valid = xyz_cam[in_front]
    uvw = (K_use @ xyz_cam_valid.T).T
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

    xyz_inside = xyz_valid[inside].astype(np.float32, copy=False)
    uv_inside = uv[inside].astype(np.float32, copy=False)
    depth_inside = xyz_cam_valid[inside, 2].astype(np.float32, copy=False)
    return xyz_inside, uv_inside, depth_inside


class PinholeLidarImageTester:
    """Test node for pinhole lidar-image correspondence."""

    def __init__(self) -> None:
        self.topic_image = rospy.get_param("~topic_image", "/camera/color/image_raw")
        self.topic_lidar = rospy.get_param("~topic_lidar", "/rslidar_points")
        self.topic_projected_cloud = rospy.get_param(
            "~topic_projected_cloud", "/fusionout/projected_cloud"
        )
        self.topic_debug_image = rospy.get_param("~topic_debug_image", "/fusionout/image")

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", MIN_DEPTH))

        self.K = np.asarray(K_PINHOLE, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(R_OPTICAL_LIDAR, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T_OPTICAL_LIDAR, dtype=np.float64).reshape(3)

        self.pub_projected_cloud = rospy.Publisher(
            self.topic_projected_cloud, PointCloud2, queue_size=1
        )
        self.pub_debug_image = rospy.Publisher(
            self.topic_debug_image, Image, queue_size=1
        )

        self.sub_image = Subscriber(self.topic_image, Image)
        self.sub_lidar = Subscriber(self.topic_lidar, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_lidar],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo(
            "pinhole test node ready: image=%s lidar=%s projected=%s debug=%s",
            self.topic_image,
            self.topic_lidar,
            self.topic_projected_cloud,
            self.topic_debug_image,
        )
        rospy.loginfo("K(row-major)=%s", self.K.reshape(-1).tolist())
        rospy.loginfo("distortion coefficients are forced to zero")
        rospy.loginfo("R(row-major)=%s T=%s", self.R.reshape(-1).tolist(), self.T.tolist())

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        try:
            image_bgr = ros_image_to_cv2_bgr(image_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(exc))
            return

        try:
            xyz_lidar = read_xyz(cloud_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cloud decode failed: %s", str(exc))
            return

        if xyz_lidar.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "empty cloud after finite filter")
            return

        h, w = image_bgr.shape[:2]
        xyz_proj, uv, depth = project_lidar_to_pinhole_image(
            xyz_lidar=xyz_lidar,
            R_optical_lidar=self.R,
            t_optical_lidar=self.T,
            K_camera=self.K,
            width=w,
            height=h,
            min_depth=self.min_depth,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "no projected points inside image")
            return

        try:
            projected_cloud = build_cloud_xyzuv(cloud_msg.header, xyz_proj, uv)
            self.pub_projected_cloud.publish(projected_cloud)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "build/publish projected cloud failed: %s", str(exc))
            return

        overlay = draw_overlay(image_bgr, uv, depth)
        overlay_msg = cv2_bgr_to_ros_image(overlay, image_msg.header)
        self.pub_debug_image.publish(overlay_msg)

        rospy.loginfo_throttle(
            1.0,
            "pinhole projection ok: input=%d projected=%d",
            int(xyz_lidar.shape[0]),
            int(xyz_proj.shape[0]),
        )


def main() -> None:
    rospy.init_node("pinhole_lidar_image_test_node")
    PinholeLidarImageTester()
    rospy.spin()


if __name__ == "__main__":
    main()
