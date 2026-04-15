import numpy as np
from sensor_msgs.msg import PointField, PointCloud2
from typing import List, Tuple
import rospy
import sensor_msgs.point_cloud2 as pc2

class test:
    def __init__(self):
        self.sub = rospy.Subscriber("/visual_points/cam_front_lidar", PointCloud2, callback=self.callback)
        self.pub_inside_points = rospy.Publisher("/fusion_lidar_camera/object_points", PointCloud2, queue_size=1)

    def _read_xyzuv(self,cloud_msg: PointCloud2, h: int, w: int) -> Tuple[np.ndarray, float]:
        """Vectorized PointCloud2 decode: return in-bound finite (x,y,z,u,v) and inside ratio."""
        u_name, v_name = ("u", "v")
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
        for name in ("x", "y", "z", u_name, v_name):
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

        width = int(cloud_msg.width)
        height = int(cloud_msg.height)
        n_points = width * height
        if n_points <= 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0

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
        u = np.asarray(points[u_name], dtype=np.float32).reshape(-1)
        v = np.asarray(points[v_name], dtype=np.float32).reshape(-1)

        rospy.loginfo("max u: %d, min u: %d, max v: %d, min v: %d", u.max(), u.min(), v.max(), v.min())
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0

        x = x[finite]
        y = y[finite]
        z = z[finite]
        u = u[finite]
        v = v[finite]

        inside = (u >= 0.0) & (u < float(h)) & (v >= 0.0) & (v < float(w))
        inside_count = int(np.count_nonzero(inside))
        if inside_count == 0:
            return np.zeros((0, 5), dtype=np.float32), 0.0
        rospy.loginfo("----------------inside-------------")
        rospy.loginfo("max u: %d, min u: %d, max v: %d, min v: %d", u[inside].max(), u[inside].min(), v[inside].max(), v[inside].min())
        xyzuv_inside = np.stack((x[inside], y[inside], z[inside], u[inside], v[inside]), axis=1).astype(
            np.float32, copy=False
        )
        ratio = float(inside_count) / float(finite_count)
        inside_ratio = ratio if 0.0 < ratio < 1.0 else 0.0
        return xyzuv_inside, inside_ratio

    def _build_cloud_xyz(self, header, xyz: np.ndarray) -> PointCloud2:
        """Build a PointCloud2 containing only x/y/z fields from an Nx3 array."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="u", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="v", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, xyz.astype(np.float32).tolist())

    def callback(self, cloud_msg: PointCloud2):
        h = 1080
        w = 1920
        inside, ratio = self._read_xyzuv(cloud_msg=cloud_msg, h=h, w=w)
        self.pub_inside_points.publish(self._build_cloud_xyz(cloud_msg.header, inside))


def main():
    rospy.init_node("test_pointcloud")
    test()
    rospy.spin()

if __name__ == "__main__":
    main()