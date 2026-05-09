#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    publishes debug overlay image.
    """

    def __init__(self) -> None:
        self.topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
        self.topic_lidar = rospy.get_param("~topic_lidar", "/lidar_points")
        self.topic_debug_image = rospy.get_param("~topic_debug_image", "/test/debug_image")
        self.camera_model = "fisheye"

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", points_project.MIN_DEPTH))
        self.fisheye_balance = float(
            rospy.get_param("~fisheye_balance", 0.0)
        )

        params = camera_handle.load_camera_params_from_yaml(camera_model=self.camera_model) 
        # For logging camera params and compatibility check
        self.K = np.asarray(params["K"],dtype=np.float64)
        self.D = np.asarray(params["D"],dtype=np.float64)
        self.distortion_model = str(params.get("distortion_model", self.camera_model))
        self.R_rect = np.asarray(params.get("R_rect", np.eye(3, dtype=np.float64)), dtype=np.float64)

        self.R = points_project.R
        self.T = points_project.T

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
            "test node ready: image=%s lidar=%s debug=%s",
            self.topic_image,
            self.topic_lidar,
            self.topic_debug_image,
        )
        rospy.loginfo(
            "camera_model=%s distortion_model=%s K(row-major)=%s",
            self.camera_model,
            self.distortion_model,
            self.K.reshape(-1).tolist(),
        )


    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:

        try:
            image_bgr = camera_handle._ros_image_to_cv2_fallback(image_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", str(exc))
            return

        try:
            image_undist, K_undist = camera_handle.undistort_image(
                image=image_bgr,
                K_camera=self.K,
                dist_coeffs=self.D,
                distortion_model=self.distortion_model,
                R_rect=self.R_rect,
                P=None,
                fisheye_balance=self.fisheye_balance,
            )
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
            min_depth=self.min_depth,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(
                2.0,
                "no projected points inside image (camera_model=%s)",
                self.camera_model,
            )
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
