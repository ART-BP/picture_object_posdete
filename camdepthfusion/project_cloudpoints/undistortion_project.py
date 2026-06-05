#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

from camdepthfusion.camera_op import camera_handle
from camdepthfusion.project_cloudpoints import cloudpoints_handle
from camdepthfusion.project_cloudpoints import points_project


class LidarImageTester:
    """Test node for lidar-image correspondence.

    Subscribes image + lidar cloud, projects lidar points to image pixels,
    publishes projected cloud (x,y,z,u,v) and debug overlay image.
    """

    def __init__(self) -> None:
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_lidar = rospy.get_param("~topic_lidar", "/lidar_points")
        self.topic_projected_cloud = rospy.get_param("~topic_projected_cloud", "/test/projected_cloud")
        self.topic_debug_image = rospy.get_param("~topic_debug_image", "/test/debug_image")
        self.camera_model = "fenduifisheye"

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", points_project.MIN_DEPTH))
        self.fisheye_balance = float(
            rospy.get_param("~fisheye_balance", 0.0)
        )
        self.pinhole_alpha = float(rospy.get_param("~pinhole_alpha", 0.0))

        params = camera_handle.load_camera_params_from_yaml(camera_model=self.camera_model) 
        # For logging camera params and compatibility check
        self.K = np.asarray(params["K"],dtype=np.float64)
        self.D = np.asarray(params["D"],dtype=np.float64)
        self.distortion_model = str(params.get("distortion_model", self.camera_model))
        self.R_rect = np.asarray(params.get("R_rect", np.eye(3, dtype=np.float64)), dtype=np.float64)
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_K = None
        self._undistort_cache_key = None
        if self.topic_lidar.endswith("loc_scan_undistort"):
            self.R = points_project.R_base_cam
            self.T = points_project.T_base_cam
            self.points_undisort_points = True
        else:
            self.R = points_project.R_fendui
            self.T = points_project.T_fendui
            self.points_undisort_points = False

        self.pub_projected_cloud = rospy.Publisher(
            self.topic_projected_cloud, PointCloud2, queue_size=1
        )
        self.pub_debug_image = rospy.Publisher(self.topic_debug_image, Image, queue_size=1)

        self.sub_image = Subscriber(self.topic_image, Image)
        self.sub_lidar = Subscriber(self.topic_lidar, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_lidar],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo(
            "test node ready: image=%s lidar=%s projected=%s debug=%s",
            self.topic_image,
            self.topic_lidar,
            self.topic_projected_cloud,
            self.topic_debug_image,
        )
        rospy.loginfo(
            "camera_model=%s distortion_model=%s K(row-major)=%s",
            self.camera_model,
            self.distortion_model,
            self.K.reshape(-1).tolist(),
        )

    @staticmethod
    def _cache_array_bytes(arr: np.ndarray) -> bytes:
        return np.ascontiguousarray(arr, dtype=np.float64).tobytes()

    def _make_undistort_cache_key(self, width: int, height: int):
        return (
            int(width),
            int(height),
            str(self.distortion_model).strip().lower(),
            float(self.fisheye_balance),
            float(self.pinhole_alpha),
            self._cache_array_bytes(self.K),
            self._cache_array_bytes(self.D),
            self._cache_array_bytes(self.R_rect),
        )

    def _ensure_undistort_maps(self, width: int, height: int) -> np.ndarray:
        cache_key = self._make_undistort_cache_key(width, height)
        if (
            cache_key == self._undistort_cache_key
            and self._undistort_map1 is not None
            and self._undistort_map2 is not None
            and self._undistort_K is not None
        ):
            return self._undistort_K

        K_use = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        D_all = np.asarray(self.D, dtype=np.float64).reshape(-1)
        R_use = np.asarray(self.R_rect, dtype=np.float64).reshape(3, 3)
        model = str(self.distortion_model or "").strip().lower()

        if "fisheye" in model:
            if D_all.size < 4:
                raise ValueError(
                    "fisheye model requires at least 4 coefficients, got %d"
                    % int(D_all.size)
                )
            D_use = D_all[:4].reshape(4, 1)
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K=K_use,
                D=D_use,
                image_size=(int(width), int(height)),
                R=R_use,
                balance=float(np.clip(self.fisheye_balance, 0.0, 1.0)),
                new_size=(int(width), int(height)),
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K=K_use,
                D=D_use,
                R=R_use,
                P=K_new,
                size=(int(width), int(height)),
                m1type=cv2.CV_16SC2,
            )
        else:
            D_use = camera_handle._normalize_dist_coeffs_for_pinhole(D_all)
            K_new, _ = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=K_use,
                distCoeffs=D_use,
                imageSize=(int(width), int(height)),
                alpha=float(np.clip(self.pinhole_alpha, 0.0, 1.0)),
                newImgSize=(int(width), int(height)),
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K_use,
                distCoeffs=D_use,
                R=R_use,
                newCameraMatrix=K_new,
                size=(int(width), int(height)),
                m1type=cv2.CV_16SC2,
            )

        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_K = K_new.astype(np.float64, copy=False)
        self._undistort_cache_key = cache_key
        rospy.loginfo(
            "undistort map cached: size=%dx%d model=%s",
            int(width),
            int(height),
            self.distortion_model,
        )
        return self._undistort_K

    def _undistort_image_cached(self, image: np.ndarray):
        if image is None:
            raise ValueError("image is None")
        if image.ndim not in (2, 3):
            raise ValueError("image must be HxW or HxWxC, got ndim=%d" % int(image.ndim))

        height, width = int(image.shape[0]), int(image.shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("invalid image shape: %s" % (str(image.shape),))

        K_undist = self._ensure_undistort_maps(width, height)
        image_undist = cv2.remap(
            image,
            self._undistort_map1,
            self._undistort_map2,
            cv2.INTER_LINEAR,
        )
        return image_undist, K_undist

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:

        try:
            image_bgr = camera_handle._ros_image_to_cv2_fallback(image_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(exc))
            return

        try:
            image_undist, K_undist = self._undistort_image_cached(image_bgr)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image undistort failed: %s", str(exc))
            return

        try:
            xyz_lidar = cloudpoints_handle._read_xyz(cloud_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cloud decode failed: %s", str(exc))
            return

        if xyz_lidar.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "empty cloud after finite filter")
            return

        h, w = image_undist.shape[:2]
        xyz_proj, uv, depth = points_project.project_lidar_to_image(
            xyz_lidar=xyz_lidar,
            R_optical_lidar=self.R,
            t_optical_lidar=self.T,
            K_camera=K_undist,
            width=w,
            height=h,
            dist_coeffs=self.D,
            undisort_points=self.points_undisort_points,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(
                2.0,
                "no projected points inside image (camera_model=%s)",
                self.camera_model,
            )
            return

        try:
            projected_cloud = cloudpoints_handle._build_cloud_xyzuv(cloud_msg.header, xyz_proj, uv)
            self.pub_projected_cloud.publish(projected_cloud)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "build/publish projected cloud failed: %s", str(exc))
            return

        overlay = points_project.draw_overlay(image_undist, uv, depth)
        overlay_msg = camera_handle._cv2_to_ros_image_fallback(overlay, image_msg.header)
        self.pub_debug_image.publish(overlay_msg)

        rospy.loginfo_throttle(
                1.0,
                "projection ok: input=%d projected=%d",
                int(xyz_lidar.shape[0]),
                int(xyz_proj.shape[0]),
            )


def main() -> None:
    rospy.init_node("lidar_image_test_node")
    LidarImageTester()
    rospy.spin()


if __name__ == "__main__":
    main()
