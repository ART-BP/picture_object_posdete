#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import rospy
import sys
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from camdepthfusion.camera_op import camera_handle
from camdepthfusion.project_cloudpoints import cloudpoints_handle
from camdepthfusion.project_cloudpoints import points_project


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

# Pinhole distortion coefficients. Common ROS plumb_bob order:
# [k1, k2, p1, p2, k3]. Use zeros for an ideal pinhole camera.
# D_PINHOLE = np.array([0.10084586419551093, -0.1017755753474589, 0.0014449907043059024, -0.0009732172673344239, -0.3645681115342965], dtype=np.float64)
D_PINHOLE = np.zeros((5,), dtype=np.float64)
# R_OPTICAL_LIDAR and T_OPTICAL_LIDAR transform lidar points into the camera
# optical frame: xyz_cam = R_OPTICAL_LIDAR @ xyz_lidar + T_OPTICAL_LIDAR.
R_OPTICAL_LIDAR = np.array(
    [
        [-0.014907, -0.999863, -0.007248],
        [-0.009365,  0.007388, -0.999929],
        [0.999845, -0.014838, -0.009474],
    ],
    dtype=np.float64,
)
T_OPTICAL_LIDAR = np.array([0.025103, -0.192104, -0.042620], dtype=np.float64)

def project_lidar_to_pinhole_image(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
):
    """Project lidar points to raw pinhole image pixels."""
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    R_use = np.asarray(R_optical_lidar, dtype=np.float64).reshape(3, 3)
    t_use = np.asarray(t_optical_lidar, dtype=np.float64).reshape(1, 3)
    K_use = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)
    D_use = camera_handle._normalize_dist_coeffs_for_pinhole(dist_coeffs)

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
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(
        objectPoints=xyz_cam_valid,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K_use,
        distCoeffs=D_use,
    )
    uv = uv.reshape(-1, 2)

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
            "~topic_projected_cloud", "/test/projected_cloud"
        )
        self.topic_debug_image = rospy.get_param("~topic_debug_image", "/test/debug_image")

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", points_project.MIN_DEPTH))
        self.remap_published_xyz = bool(rospy.get_param("~remap_published_xyz", False))

        self.K = np.asarray(K_PINHOLE, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D_PINHOLE, dtype=np.float64).reshape(-1)
        self.R = np.asarray(R_OPTICAL_LIDAR, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T_OPTICAL_LIDAR, dtype=np.float64).reshape(3)
        self.axis_remap = points_project.AXIS_REMAP

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
        rospy.loginfo("D=%s", self.D.tolist())
        rospy.loginfo("R(row-major)=%s T=%s", self.R.reshape(-1).tolist(), self.T.tolist())

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        try:
            image_bgr = camera_handle._ros_image_to_cv2_fallback(image_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(exc))
            return

        try:
            xyz_lidar = cloudpoints_handle._read_xyz(cloud_msg)
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
            dist_coeffs=self.D,
            width=w,
            height=h,
            min_depth=self.min_depth,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "no projected points inside image")
            return

        try:
            xyz_publish = xyz_proj
            if self.remap_published_xyz:
                xyz_publish = (self.axis_remap @ xyz_publish.T).T
            projected_cloud = cloudpoints_handle._build_cloud_xyzuv(
                cloud_msg.header, xyz_publish, uv
            )
            self.pub_projected_cloud.publish(projected_cloud)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "build/publish projected cloud failed: %s", str(exc))
            return

        overlay = points_project.draw_overlay(image_bgr, uv, depth)
        overlay_msg = camera_handle._cv2_to_ros_image_fallback(overlay, image_msg.header)
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
