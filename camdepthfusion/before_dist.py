#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

import camera_handle
from cloudpoints_handle import CloudPointsHandle
import points_project


class LidarImageBeforeDistTester:
    """Undistort image first, then project lidar points on undistorted image."""

    def __init__(self) -> None:
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_lidar = rospy.get_param("~topic_lidar", "/lidar_points")
        self.topic_projected_cloud = rospy.get_param("~topic_projected_cloud", "/before_dist/projected_cloud")
        self.topic_debug_image = rospy.get_param("~topic_debug_image", "/before_dist/debug_image")

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", points_project.MIN_DEPTH))
        self.undistort_alpha = float(rospy.get_param("~undistort_alpha", 0.0))
        self.camera_model = str(rospy.get_param("~camera_model", "rational_polynomial"))

        params = camera_handle.load_camera_params_from_yaml(camera_model=self.camera_model)
        self.K = np.asarray(params["K"], dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(params["D"], dtype=np.float64).reshape(-1)

        self.axis_remap = points_project.AXIS_REMAP
        self.R = points_project.R
        self.T = points_project.T

        self._map1 = None
        self._map2 = None
        self._K_undist = None
        self._map_size = None

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
            "before_dist ready: image=%s lidar=%s projected=%s debug=%s model=%s",
            self.topic_image,
            self.topic_lidar,
            self.topic_projected_cloud,
            self.topic_debug_image,
            self.camera_model,
        )
        rospy.loginfo("K_raw(row-major)=%s", self.K.reshape(-1).tolist())

    def _prepare_undistort_maps(self, width: int, height: int) -> None:
        size = (int(width), int(height))
        if self._map_size == size and self._map1 is not None and self._map2 is not None:
            return

        if self.camera_model == "fisheye":
            D_use = self.D[:4].astype(np.float64, copy=False).reshape(4, 1)
            balance = float(np.clip(self.undistort_alpha, 0.0, 1.0))
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K=self.K,
                D=D_use,
                image_size=size,
                R=np.eye(3, dtype=np.float64),
                balance=balance,
                new_size=size,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K=self.K,
                D=D_use,
                R=np.eye(3, dtype=np.float64),
                P=K_new,
                size=size,
                m1type=cv2.CV_32FC1,
            )
        else:
            K_new, _ = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=self.K,
                distCoeffs=self.D,
                imageSize=size,
                alpha=self.undistort_alpha,
                newImgSize=size,
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=self.K,
                distCoeffs=self.D,
                R=np.eye(3, dtype=np.float64),
                newCameraMatrix=K_new,
                size=size,
                m1type=cv2.CV_32FC1,
            )

        self._map1 = map1
        self._map2 = map2
        self._K_undist = K_new.astype(np.float64, copy=False)
        self._map_size = size
        rospy.loginfo("undistort map initialized: size=%s", str(size))
        rospy.loginfo("K_undist(row-major)=%s", self._K_undist.reshape(-1).tolist())

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        try:
            image_bgr = camera_handle._ros_image_to_cv2_fallback(image_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(exc))
            return

        try:
            xyz_lidar = CloudPointsHandle._read_xyz(cloud_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cloud decode failed: %s", str(exc))
            return

        if xyz_lidar.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "empty cloud after finite filter")
            return

        h_raw, w_raw = image_bgr.shape[:2]
        self._prepare_undistort_maps(width=w_raw, height=h_raw)
        undist_bgr = cv2.remap(
            image_bgr,
            self._map1,
            self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        xyz_lidar = (self.axis_remap.reshape(3, 3) @ xyz_lidar.T).T
        h, w = undist_bgr.shape[:2]
        xyz_proj, uv, depth = points_project.project_lidar_to_image(
            xyz_lidar=xyz_lidar,
            R_optical_lidar=self.R,
            t_optical_lidar=self.T,
            K_camera=self._K_undist,
            width=w,
            height=h,
            dist_coeffs=np.zeros((4,), dtype=np.float64),
            min_depth=self.min_depth,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "no projected points inside undistorted image")
            return

        try:
            projected_cloud = CloudPointsHandle._build_cloud_xyzuv(cloud_msg.header, xyz_proj, uv)
            self.pub_projected_cloud.publish(projected_cloud)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "build/publish projected cloud failed: %s", str(exc))
            return

        overlay = points_project.draw_overlay(undist_bgr, uv, depth)
        overlay_msg = camera_handle._cv2_to_ros_image_fallback(overlay, image_msg.header)
        self.pub_debug_image.publish(overlay_msg)

        rospy.loginfo_throttle(
            1.0,
            "before_dist ok: input=%d projected=%d",
            int(xyz_lidar.shape[0]),
            int(xyz_proj.shape[0]),
        )


def main() -> None:
    rospy.init_node("lidar_image_before_dist_node")
    LidarImageBeforeDistTester()
    rospy.spin()


if __name__ == "__main__":
    main()
