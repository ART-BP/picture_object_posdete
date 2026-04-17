#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from typing import List

class PointCloudRangeFilterNode:
    """Subscribe PointCloud2, filter by xyz ranges, republish filtered cloud."""

    def __init__(self):
        self.input_cloud_topic = str(rospy.get_param("~input_cloud_topic", "/lidar_points"))
        self.output_cloud_topic = str(
            rospy.get_param("~output_cloud_topic", "/camdepthfusion/range_filtered_cloud")
        )
        self.enable_online_range_update = bool(rospy.get_param("~enable_online_range_update", True))
        self.range_check_period = float(rospy.get_param("~range_check_period", 0.2))
        self._last_range_signature = None
        self._last_range_check_sec = 0.0

        self.x_min = -math.inf
        self.x_max = math.inf
        self.y_min = -math.inf
        self.y_max = math.inf
        self.z_min = -math.inf
        self.z_max = math.inf
        self._refresh_ranges_if_needed(force=True)

        self.pub_filtered_cloud = rospy.Publisher(
            self.output_cloud_topic, PointCloud2, queue_size=1
        )
        self.sub_lidar = rospy.Subscriber(
            self.input_cloud_topic, PointCloud2, self.cloud_callback, queue_size=1
        )
        rospy.loginfo(
            "[camdepthfusion] point cloud range filter ready: input=%s output=%s",
            self.input_cloud_topic,
            self.output_cloud_topic,
        )

    @staticmethod
    def _read_bound_param(name, default_value):
        value = rospy.get_param(name, default_value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("inf", "+inf", "infinity", "+infinity"):
                return math.inf
            if text in ("-inf", "-infinity"):
                return -math.inf
        return float(value)

    def _read_range_params(self):
        x_min = self._read_bound_param("~x_min", -math.inf)
        x_max = self._read_bound_param("~x_max", math.inf)
        y_min = self._read_bound_param("~y_min", -math.inf)
        y_max = self._read_bound_param("~y_max", math.inf)
        z_min = self._read_bound_param("~z_min", -math.inf)
        z_max = self._read_bound_param("~z_max", math.inf)

        if x_min > x_max:
            raise ValueError("~x_min must be <= ~x_max")
        if y_min > y_max:
            raise ValueError("~y_min must be <= ~y_max")
        if z_min > z_max:
            raise ValueError("~z_min must be <= ~z_max")
        return x_min, x_max, y_min, y_max, z_min, z_max

    def _refresh_ranges_if_needed(self, force=False):
        now_sec = rospy.Time.now().to_sec()
        if not force:
            if not self.enable_online_range_update:
                return
            if self.range_check_period > 0.0 and (now_sec - self._last_range_check_sec) < self.range_check_period:
                return
        self._last_range_check_sec = now_sec

        x_min, x_max, y_min, y_max, z_min, z_max = self._read_range_params()
        signature = (x_min, x_max, y_min, y_max, z_min, z_max)
        if (not force) and (signature == self._last_range_signature):
            return

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self._last_range_signature = signature

        rospy.loginfo(
            "[camdepthfusion] range updated: x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]",
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
        )

    @staticmethod
    def _build_xyz_dtype(cloud_msg):
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
                raise ValueError(f"PointCloud2 missing required field '{name}'")
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
        return point_dtype

    @staticmethod
    def _build_point_view(cloud_msg, point_dtype):
        width = int(cloud_msg.width)
        height = int(cloud_msg.height)
        n_points = width * height
        if n_points <= 0:
            return np.zeros((0,), dtype=point_dtype)

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
        return points.reshape(-1)

    @staticmethod
    def _build_filtered_cloud_msg(cloud_msg, filtered_points, is_dense):
        out = PointCloud2()
        out.header = cloud_msg.header
        out.height = 1
        out.width = int(filtered_points.shape[0])
        out.fields = cloud_msg.fields
        out.is_bigendian = cloud_msg.is_bigendian
        out.point_step = int(cloud_msg.point_step)
        out.row_step = out.point_step * out.width
        out.is_dense = bool(is_dense)
        out.data = filtered_points.tobytes(order="C")
        return out

    def cloud_callback(self, cloud_msg):
        try:
            self._refresh_ranges_if_needed(force=False)
            point_dtype = self._build_xyz_dtype(cloud_msg)
            points = self._build_point_view(cloud_msg, point_dtype)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[camdepthfusion] cloud parse failed: %s", str(exc))
            return

        if points.shape[0] == 0:
            out_msg = self._build_filtered_cloud_msg(cloud_msg, points, is_dense=cloud_msg.is_dense)
            self.pub_filtered_cloud.publish(out_msg)
            return

        x = np.asarray(points["x"], dtype=np.float32)
        y = np.asarray(points["y"], dtype=np.float32)
        z = np.asarray(points["z"], dtype=np.float32)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        in_range = (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )
        keep = finite & in_range
        filtered_points = points[keep]

        out_msg = self._build_filtered_cloud_msg(cloud_msg, filtered_points, is_dense=cloud_msg.is_dense)
        self.pub_filtered_cloud.publish(out_msg)

        rospy.logdebug(
            "[camdepthfusion] in=%d kept=%d ratio=%.3f range[x=%.3f..%.3f y=%.3f..%.3f z=%.3f..%.3f]",
            points.shape[0],
            filtered_points.shape[0],
            (float(filtered_points.shape[0]) / float(max(points.shape[0], 1))),
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
        )


def main():
    rospy.init_node("cam_lidar_projector_node")
    try:
        PointCloudRangeFilterNode()
    except Exception as exc:
        rospy.logerr("[camdepthfusion] node init failed: %s", str(exc))
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
