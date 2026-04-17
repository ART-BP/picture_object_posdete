#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import rospy
import sensor_msgs.point_cloud2 as pc2
import math
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from typing import List

class CamLidarProjectorNode:
    """Subscribe image + lidar cloud, project lidar points to image, republish projected cloud."""

    def __init__(self):
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.min_depth = float(rospy.get_param("~min_depth", 0.05))
        self.point_radius = int(rospy.get_param("~point_radius", 2))
        self.max_overlay_points = int(rospy.get_param("~max_overlay_points", 8000))
        self.camera_width = int(rospy.get_param("~camera_width", 1920))
        self.camera_height = int(rospy.get_param("~camera_height", 1080))
        self.distortion_model = rospy.get_param("~distortion_model", "rational_polynomial")
        self.enable_online_extrinsic_update = bool(
            rospy.get_param("~enable_online_extrinsic_update", True)
        )
        self.extrinsic_check_period = float(rospy.get_param("~extrinsic_check_period", 0.2))
        self._last_extrinsic_signature = None
        self._last_extrinsic_check_sec = 0.0

        self.K = np.asarray([
            1166.271789008082,
            0.0,
            959.0530108632893,
            0.0,
            1167.7222552768242,
            544.1763261328632,
            0.0,
            0.0,
            1.0,
        ]).reshape(3, 3)
        self.dist_coeffs = np.asarray([
            -0.1269125424187082,
            0.08472977776858055,
            0.36834095572628567,
            -0.27857771753696225,
            0.0,
        ]).reshape(-1)
        self.R_optical_from_camera = self._build_R_optical_from_camera_physical("left")
        self.T_optical_from_camera = np.eye(4, dtype=np.float64)
        self.T_optical_from_camera[:3, :3] = self.R_optical_from_camera
        self.R_cam_lidar = np.eye(3, dtype=np.float64)
        self.t_cam_lidar = np.zeros((3,), dtype=np.float64)
        self.T_cam_lidar = np.eye(4, dtype=np.float64)
        self.T_optical_lidar = np.eye(4, dtype=np.float64)
        self.R_optical_lidar = np.eye(3, dtype=np.float64)
        self.t_optical_lidar = np.zeros((3,), dtype=np.float64)
        self._refresh_extrinsic_if_needed(force=True)

        self.bridge = CvBridge()
        self.pub_projected_cloud = rospy.Publisher(
            "/camdepthfusion/projected_cloud", PointCloud2, queue_size=1
        )
        self.pub_overlay_image = rospy.Publisher(
            "/camdepthfusion/projected_image", Image, queue_size=1
        )

        self.sub_image = Subscriber("/camera/go2/front/image_raw", Image)
        self.sub_lidar = Subscriber("/lidar_points", PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_lidar],
            queue_size=2,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)


    @staticmethod
    def _require_length(name, values, expected_len):
        if values is None or len(values) != expected_len:
            raise ValueError("%s must be a list of length %d" % (name, expected_len))

    @staticmethod
    def _rot_x(a):
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]],
            dtype=np.float64,
        )

    @staticmethod
    def _rot_y(a):
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array(
            [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]],
            dtype=np.float64,
        )

    @staticmethod
    def _rot_z(a):
        ca = math.cos(a)
        sa = math.sin(a)
        return np.array(
            [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _build_R_optical_from_camera_physical(camera_y_axis):
        # Convert camera physical frame (x forward, y ?, z up)
        # to optical frame used by OpenCV projection (x right, y down, z forward).
        if camera_y_axis == "left":
            return np.array(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        if camera_y_axis == "right":
            return np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        raise ValueError("~camera_physical_y_axis must be 'left' or 'right'")

    def _read_extrinsic_params(self):
        x = float(rospy.get_param("~x", 0.0))
        y = float(rospy.get_param("~y", 0.0))
        z = float(rospy.get_param("~z", 0.0))
        roll = float(rospy.get_param("~roll", 0.0))
        pitch = float(rospy.get_param("~pitch", 0.0))
        yaw = float(rospy.get_param("~yaw", 0.0))
        angle_unit = str(rospy.get_param("~angle_unit", rospy.get_param("~euler_unit", "deg"))).lower()
        camera_y_axis = str(rospy.get_param("~camera_physical_y_axis", "left")).lower()

        if angle_unit not in ("deg", "rad"):
            raise ValueError("~angle_unit must be 'deg' or 'rad'")
        if camera_y_axis not in ("left", "right"):
            raise ValueError("~camera_physical_y_axis must be 'left' or 'right'")
        return x, y, z, roll, pitch, yaw, angle_unit, camera_y_axis

    def _build_T_cam_lidar_from_xyzrpy(self, x, y, z, roll, pitch, yaw, angle_unit):
        roll_rad = roll
        pitch_rad = pitch
        yaw_rad = yaw

        if angle_unit == "deg":
            roll_rad = math.radians(roll)
            pitch_rad = math.radians(pitch)
            yaw_rad = math.radians(yaw)

        # ZYX euler: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R = self._rot_z(yaw_rad) @ self._rot_y(pitch_rad) @ self._rot_x(roll_rad)
        t = np.array([x, y, z], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return R, t, T

    def _refresh_extrinsic_if_needed(self, force=False):
        now_sec = rospy.Time.now().to_sec()
        if not force:
            if not self.enable_online_extrinsic_update:
                return
            if self.extrinsic_check_period > 0.0 and (now_sec - self._last_extrinsic_check_sec) < self.extrinsic_check_period:
                return
        self._last_extrinsic_check_sec = now_sec

        x, y, z, roll, pitch, yaw, angle_unit, camera_y_axis = self._read_extrinsic_params()
        signature = (x, y, z, roll, pitch, yaw, angle_unit, camera_y_axis)
        if (not force) and (signature == self._last_extrinsic_signature):
            return

        self.R_optical_from_camera = self._build_R_optical_from_camera_physical(camera_y_axis)
        self.T_optical_from_camera[:3, :3] = self.R_optical_from_camera
        self.R_cam_lidar, self.t_cam_lidar, self.T_cam_lidar = self._build_T_cam_lidar_from_xyzrpy(
            x, y, z, roll, pitch, yaw, angle_unit
        )
        self.T_optical_lidar = self.T_optical_from_camera @ self.T_cam_lidar
        self.R_optical_lidar = self.T_optical_lidar[:3, :3]
        self.t_optical_lidar = self.T_optical_lidar[:3, 3]
        self._last_extrinsic_signature = signature

        rospy.loginfo(
            "[camdepthfusion] extrinsic updated: x=%.4f y=%.4f z=%.4f roll=%.4f pitch=%.4f yaw=%.4f unit=%s camera_y_axis=%s",
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            angle_unit,
            camera_y_axis,
        )
        rospy.loginfo("[camdepthfusion] T_cam_lidar(row-major)=%s", self.T_cam_lidar.reshape(-1).tolist())
        rospy.loginfo(
            "[camdepthfusion] T_optical_lidar(row-major, used for projection)=%s",
            self.T_optical_lidar.reshape(-1).tolist(),
        )

    @staticmethod
    def _read_xyz_from_cloud(cloud_msg):
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
            field = field_map[name]
            if int(field.count) != 1:
                raise ValueError(f"Field '{name}' has count={field.count}, only scalar fields are supported")
            base_fmt = dtype_map.get(field.datatype)
            if base_fmt is None:
                raise ValueError(f"Unsupported PointField datatype={field.datatype} for field '{name}'")
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

        # height 1
        width = int(cloud_msg.width)
        height = int(cloud_msg.height)
        n_points = width * height

        if n_points <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        expected_bytes = int(cloud_msg.row_step) * height
        if len(cloud_msg.data) < expected_bytes:
            raise ValueError(
                f"PointCloud2 data too short: len(data)={len(cloud_msg.data)} expected>={expected_bytes}"
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
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            return np.zeros((0, 3), dtype=np.float32)

        x = x[finite]
        y = y[finite]
        z = z[finite]
        
        xyz = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)
        return xyz

    def _project_lidar_to_image(self, xyz_lidar):
        # Use lidar -> optical extrinsic for projection.
        xyz_cam_optical = (self.R_optical_lidar @ xyz_lidar.T).T + self.t_optical_lidar

        valid_depth = xyz_cam_optical[:, 2] > self.min_depth
        xyz_lidar_valid = xyz_lidar[valid_depth]
        xyz_cam_optical = xyz_cam_optical[valid_depth]
        if xyz_cam_optical.shape[0] == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        if self.dist_coeffs.size > 0:
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            uv, _ = cv2.projectPoints(
                xyz_cam_optical.astype(np.float64),
                rvec,
                tvec,
                self.K,
                self.dist_coeffs,
            )
            uv = uv.reshape(-1, 2)
        else:
            uvw = (self.K @ xyz_cam_optical.T).T
            uv = uvw[:, :2] / uvw[:, 2:3]

        return (
            xyz_lidar_valid.astype(np.float32),
            uv.astype(np.float32),
            xyz_cam_optical[:, 2].astype(np.float32),
        )

    @staticmethod
    def _build_projected_cloud_msg(header, xyz_cam, uv):
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("u", 12, PointField.FLOAT32, 1),
            PointField("v", 16, PointField.FLOAT32, 1),
        ]
        points = np.concatenate([xyz_cam, uv], axis=1).astype(np.float32)
        return pc2.create_cloud(header, fields, points)

    def _draw_overlay(self, image_bgr, uv, depth):
        overlay = image_bgr.copy()
        count = uv.shape[0]
        if count == 0:
            return overlay

        if count > self.max_overlay_points:
            idx = np.linspace(0, count - 1, self.max_overlay_points).astype(np.int32)
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
            u_i, v_i = int(uv_int[i, 0]), int(uv_int[i, 1])
            c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            cv2.circle(overlay, (u_i, v_i), self.point_radius, c, -1, lineType=cv2.LINE_AA)

        return overlay

    def synced_callback(self, image_msg, cloud_msg):
        self._refresh_extrinsic_if_needed(force=False)
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "[camdepthfusion] image convert failed: %s", str(exc))
            return

        h, w = image_bgr.shape[:2]
        if w != self.camera_width or h != self.camera_height:
            rospy.logwarn_throttle(
                2.0,
                "[camdepthfusion] image size mismatch: got=%dx%d expected=%dx%d",
                w,
                h,
                self.camera_width,
                self.camera_height,
            )
        xyz_lidar = self._read_xyz_from_cloud(cloud_msg)
        if xyz_lidar.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "[camdepthfusion] input cloud has no valid xyz points")
            return

        xyz_lidar_proj, uv, depth_optical = self._project_lidar_to_image(xyz_lidar)
        if xyz_lidar_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "[camdepthfusion] all points removed by depth filter")
            return
        projected_count = int(xyz_lidar_proj.shape[0])

        inside = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < float(w))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < float(h))
        )
        xyz_lidar_proj = xyz_lidar_proj[inside]
        uv = uv[inside]
        depth_optical = depth_optical[inside]
        inside_count = int(xyz_lidar_proj.shape[0])

        if xyz_lidar_proj.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "[camdepthfusion] no projected points inside image")
            return

        header = Header()
        header = cloud_msg.header

        # Published xyz are in lidar frame.
        projected_cloud_msg = self._build_projected_cloud_msg(header, xyz_lidar_proj, uv)
        self.pub_projected_cloud.publish(projected_cloud_msg)

        overlay = self._draw_overlay(image_bgr, uv, depth_optical)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = image_msg.header
        self.pub_overlay_image.publish(overlay_msg)

        rospy.logdebug(
            "[camdepthfusion] in=%d projected=%d inside=%d",
            xyz_lidar.shape[0],
            projected_count,
            inside_count,
        )


def main():
    rospy.init_node("cam_lidar_projector_node")
    try:
        CamLidarProjectorNode()
    except Exception as exc:
        rospy.logerr("[camdepthfusion] node init failed: %s", str(exc))
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
