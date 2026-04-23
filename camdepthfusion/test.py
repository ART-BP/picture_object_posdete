#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

from camdepthfusion import camera_handle
from camdepthfusion import cloudpoints_handle
from camdepthfusion import points_project


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

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 3))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", points_project.MIN_DEPTH))
        self.fisheye_theta_margin_deg = float(
            rospy.get_param("~fisheye_theta_margin_deg", 1.0)
        )
        self.enable_online_extrinsic_update = bool(
            rospy.get_param("~enable_online_extrinsic_update", True)
        )

        params = camera_handle.load_camera_params_from_yaml(camera_model="fisheye")  # For logging camera params and compatibility check
        self.K = np.asarray(params["K"],dtype=np.float64)
        self.D = np.asarray(params["D"],dtype=np.float64)

        self.axis_remap = points_project.AXIS_REMAP

        self.R = points_project.R
        self.T = points_project.T

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
            "test node ready: image=%s lidar=%s projected=%s debug=%s",
            self.topic_image,
            self.topic_lidar,
            self.topic_projected_cloud,
            self.topic_debug_image,
        )
        rospy.loginfo("K(row-major)=%s", self.K.reshape(-1).tolist())


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

        # xyz_lidar = (self.axis_remap.reshape(3, 3) @ xyz_lidar.T).T

        # h, w = image_bgr.shape[:2]
        # xyz_proj, uv, depth = points_project.project_lidar_to_image_with_distortion(
        #     xyz_lidar=xyz_lidar,
        #     R_optical_lidar=self.R,
        #     t_optical_lidar=self.T,
        #     K_camera=self.K,
        #     width=w,
        #     height=h,
        #     dist_coeffs=self.D,
        #     min_depth=self.min_depth,
        # )
        # if xyz_proj.shape[0] == 0:
        #     rospy.logwarn_throttle(2.0, "no projected points inside image")
        #     return

        h, w = image_bgr.shape[:2]
        xyz_proj, uv, depth = points_project.project_lidar_to_image_with_fisheye_distortion(
            xyz_lidar=xyz_lidar,
            R_optical_lidar=self.R,
            t_optical_lidar=self.T,
            K_camera=self.K,
            width=w,
            height=h,
            dist_coeffs=self.D,
            min_depth=self.min_depth,
        )
        if xyz_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "no projected points inside image")
            return

        # h, w = image_bgr.shape[:2]
        # xyz_proj, uv, depth = points_project.project_lidar_to_image_with_rational_polynomial(
        #     xyz_lidar=xyz_lidar,
        #     R_optical_lidar=self.R,
        #     t_optical_lidar=self.T,
        #     K_camera=self.K,
        #     width=w,
        #     height=h,
        #     dist_coeffs=self.D,
        #     min_depth=self.min_depth,
        # )
        # if xyz_proj.shape[0] == 0:
        #     rospy.logwarn_throttle(2.0, "no projected points inside image")
        #     return
        
        try:
            xyz_proj = (self.axis_remap.reshape(3, 3) @ xyz_proj.T).T
            projected_cloud = cloudpoints_handle._build_cloud_xyzuv(cloud_msg.header, xyz_proj, uv)
            self.pub_projected_cloud.publish(projected_cloud)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "build/publish projected cloud failed: %s", str(exc))
            return

        overlay = points_project.draw_overlay(image_bgr, uv, depth)
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
