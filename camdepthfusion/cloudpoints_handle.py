import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from typing import List, Tuple
import sensor_msgs.point_cloud2 as pc2

def _read_xyz(cloud_msg: PointCloud2) -> np.ndarray:
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
        raise ValueError("PointCloud2 data too short: len(data)=%d expected>=%d" % (len(cloud_msg.data), expected_bytes))

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

    return np.stack((x[finite], y[finite], z[finite]), axis=1).astype(np.float32, copy=False)


def _read_xyzuv(cloud_msg: PointCloud2) -> np.ndarray:
    """Vectorized PointCloud2 decode: return finite (x,y,z,u,v)."""
    u_name = 'u'
    v_name = 'v'
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

    # height 1
    width = int(cloud_msg.width)
    height = int(cloud_msg.height)
    n_points = width * height

    if n_points <= 0:
        return np.zeros((0, 5), dtype=np.float32)

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

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return np.zeros((0, 5), dtype=np.float32)

    x = x[finite]
    y = y[finite]
    z = z[finite]
    u = u[finite]
    v = v[finite]
    
    xyzuv = np.stack((x, y, z, u, v), axis=1).astype(np.float32, copy=False)
    return xyzuv


def _build_cloud_xyzuv(header, xyz: np.ndarray, uv: np.ndarray) -> PointCloud2:
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("u", 12, PointField.FLOAT32, 1),
        PointField("v", 16, PointField.FLOAT32, 1),
    ]
    points = np.concatenate([xyz, uv], axis=1).astype(np.float32)
    return pc2.create_cloud(header, fields, points)


def _build_cloud_xyz(header, xyz: np.ndarray) -> PointCloud2:
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(header, fields, xyz.astype(np.float32).tolist())


def cluster_2d_center_nearest_surface(
    points_xy: np.ndarray,
    grid_size: float = 0.1,
    min_points_per_cell: int = 2,
    min_cluster_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster 2D points and return (median center, nearest surface point)."""
    if points_xy is None:
        nan_xy = np.array([np.nan, np.nan], dtype=np.float32)
        return nan_xy, nan_xy.copy()

    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Expected Nx2 points, got shape={pts.shape}")
    if pts.shape[1] > 2:
        pts = pts[:, :2]
    if pts.shape[0] == 0:
        nan_xy = np.array([np.nan, np.nan], dtype=np.float32)
        return nan_xy, nan_xy.copy()

    if grid_size <= 0:
        raise ValueError(f"grid_size must be > 0, got {grid_size}")

    cell_points = np.floor(pts / float(grid_size)).astype(np.int32)
    unique_cells, counts = np.unique(cell_points, axis=0, return_counts=True)

    dense_mask = counts >= max(1, int(min_points_per_cell))
    if not np.any(dense_mask):
        center = np.median(pts, axis=0).astype(np.float32)
        # Same ordering as norm(), but avoids sqrt for better speed.
        nearest_idx = int(np.argmin(np.sum(pts * pts, axis=1)))
        nearest_xy = pts[nearest_idx].astype(np.float32)
        return center, nearest_xy

    dense_cells = unique_cells[dense_mask]
    dense_set = {tuple(c.tolist()) for c in dense_cells}
    cell_count = {tuple(c.tolist()): int(n) for c, n in zip(unique_cells, counts)}

    visited = set()
    best_component_cells = set()
    best_component_points = 0

    for cell_arr in dense_cells:
        start = tuple(cell_arr.tolist())
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component_cells = set()
        component_points = 0

        while stack:
            cx, cy = stack.pop()
            cell = (cx, cy)
            component_cells.add(cell)
            component_points += cell_count.get(cell, 0)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (cx + dx, cy + dy)
                    if nxt in dense_set and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

        if component_points > best_component_points:
            best_component_points = component_points
            best_component_cells = component_cells

    if not best_component_cells:
        cluster_pts = pts
    else:
        keep = np.fromiter(
            (tuple(c.tolist()) in best_component_cells for c in cell_points),
            dtype=np.bool_,
            count=cell_points.shape[0],
        )
        cluster_pts = pts[keep]
        if cluster_pts.shape[0] < max(1, int(min_cluster_points)):
            cluster_pts = pts

    center = np.median(cluster_pts, axis=0).astype(np.float32)
    # Same ordering as norm(), but avoids sqrt for better speed.
    nearest_idx = int(np.argmin(np.sum(cluster_pts * cluster_pts, axis=1)))
    nearest_xy = cluster_pts[nearest_idx].astype(np.float32)
    return center, nearest_xy