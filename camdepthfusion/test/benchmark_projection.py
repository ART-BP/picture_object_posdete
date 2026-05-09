#!/usr/bin/env python3
"""Benchmark point-cloud projection functions in camdepthfusion/project_cloudpoints/points_project.py."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camdepthfusion.project_cloudpoints import points_project


def _load_camera_model_params(yaml_path: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if model_name not in cfg:
        raise KeyError(f"camera model '{model_name}' not found in {yaml_path}")

    model_cfg = cfg[model_name]
    k_raw = np.asarray(model_cfg["camera_matrix"]["data"], dtype=np.float64).reshape(-1)
    d_raw = np.asarray(model_cfg["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)
    if k_raw.size != 9:
        raise ValueError(f"camera_matrix must have 9 values, got {k_raw.size}")
    if d_raw.size == 0:
        raise ValueError("distortion_coefficients is empty")
    return k_raw.reshape(3, 3), d_raw


def _gen_xyz(num_points: int, max_range_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, max_range_m, size=num_points)
    y = rng.uniform(-max_range_m, max_range_m, size=num_points)
    z = rng.uniform(-0.3 * max_range_m, 0.3 * max_range_m, size=num_points)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _as_float_seconds(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp)


def _load_first_matched_xyz_from_bag(
    bag_path: Path,
    cloud_topic: str,
    image_topic: str,
    sync_slop: float,
) -> Tuple[np.ndarray, int, int, float, float, float]:
    try:
        import rosbag
    except Exception as exc:
        raise ImportError(
            "import rosbag failed. Please source ROS env first (e.g. `source /opt/ros/noetic/setup.bash`) "
            f"and ensure rosbag Python deps are installed. Original error: {exc}"
        ) from exc

    from camdepthfusion.project_cloudpoints import cloudpoints_handle

    latest_cloud_msg = None
    latest_cloud_sec = None
    latest_image_msg = None
    latest_image_sec = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[cloud_topic, image_topic]):
            stamp_sec = _as_float_seconds(msg.header.stamp)
            if topic == cloud_topic:
                latest_cloud_msg = msg
                latest_cloud_sec = stamp_sec
            elif topic == image_topic:
                latest_image_msg = msg
                latest_image_sec = stamp_sec
            else:
                continue

            if latest_cloud_msg is None or latest_image_msg is None:
                continue

            dt = abs(float(latest_cloud_sec) - float(latest_image_sec))
            if dt > float(sync_slop):
                continue

            xyz = cloudpoints_handle._read_xyz(latest_cloud_msg)
            if xyz.shape[0] == 0:
                continue

            width = int(getattr(latest_image_msg, "width", 0))
            height = int(getattr(latest_image_msg, "height", 0))
            if width <= 0 or height <= 0:
                raise ValueError("matched image frame has invalid width/height")

            return xyz, width, height, float(latest_cloud_sec), float(latest_image_sec), float(dt)

    raise RuntimeError(
        f"no matched frame found in bag={bag_path} for topics {cloud_topic} / {image_topic} "
        f"within sync_slop={sync_slop}s"
    )


def _prepare_xyz(xyz: np.ndarray, sample_step: int, max_points: int) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float32)
    step = max(1, int(sample_step))
    if step > 1 and pts.shape[0] > 0:
        pts = pts[::step]

    cap = int(max_points)
    if cap > 0 and pts.shape[0] > cap:
        idx = np.linspace(0, pts.shape[0] - 1, cap).astype(np.int64)
        pts = pts[idx]
    return pts


def _bench_once(
    fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    xyz: np.ndarray,
    loops: int,
    warmup: int,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn(xyz)

    times_ms = []
    out_counts = []
    for _ in range(loops):
        t0 = time.perf_counter()
        xyz_out, _, _ = fn(xyz)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        out_counts.append(float(xyz_out.shape[0]))

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "out_mean": float(np.mean(out_counts)),
    }


def main() -> None:
    default_yaml = REPO_ROOT / "camdepthfusion" / "camera_op" / "config" / "param_camera.yaml"
    default_bag = REPO_ROOT / "bag" / "two_person.bag"

    parser = argparse.ArgumentParser(description="Benchmark projection methods in points_project.py")
    parser.add_argument("--source", choices=["bag", "random"], default="bag")
    parser.add_argument("--bag", type=Path, default=default_bag, help="bag path when --source bag")
    parser.add_argument("--cloud-topic", type=str, default="/lidar_points")
    parser.add_argument("--image-topic", type=str, default="/camera/go2/front/image_raw")
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--yaml", type=Path, default=default_yaml, help="camera yaml path")
    parser.add_argument("--pinhole-model", type=str, default="rational_polynomial")
    parser.add_argument("--fisheye-model", type=str, default="fisheye")
    parser.add_argument("--num-points", type=int, default=120000, help="used only when --source random")
    parser.add_argument("--width", type=int, default=1920, help="used only when --source random")
    parser.add_argument("--height", type=int, default=1080, help="used only when --source random")
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-range-m", type=float, default=20.0)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--sample-step", type=int, default=1, help="take every Nth point before benchmarking")
    parser.add_argument("--max-points", type=int, default=0, help="cap input points after sampling; 0 means unlimited")
    args = parser.parse_args()

    K_pin, D_pin = _load_camera_model_params(args.yaml, args.pinhole_model)
    K_fish, D_fish = _load_camera_model_params(args.yaml, args.fisheye_model)
    R = points_project.R
    T = points_project.T

    if args.source == "bag":
        xyz_raw, width, height, cloud_sec, image_sec, dt = _load_first_matched_xyz_from_bag(
            bag_path=args.bag,
            cloud_topic=args.cloud_topic,
            image_topic=args.image_topic,
            sync_slop=args.sync_slop,
        )
        source_info = (
            f"source=bag path={args.bag} cloud_topic={args.cloud_topic} image_topic={args.image_topic} "
            f"cloud_t={cloud_sec:.6f} image_t={image_sec:.6f} dt={dt:.6f}s"
        )
    else:
        xyz_raw = _gen_xyz(args.num_points, args.max_range_m, args.seed)
        width = int(args.width)
        height = int(args.height)
        source_info = (
            f"source=random num_points={args.num_points} max_range_m={args.max_range_m} seed={args.seed}"
        )

    xyz = _prepare_xyz(xyz_raw, args.sample_step, args.max_points)
    if xyz.shape[0] == 0:
        raise RuntimeError("input point cloud is empty after sampling/capping")

    methods: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "undistorted_pinhole": lambda pts: points_project.project_lidar_to_image(
            xyz_lidar=pts,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_pin,
            width=width,
            height=height,
            dist_coeffs=D_pin,
            min_depth=args.min_depth_m,
        ),
        "opencv_distortion": lambda pts: points_project.project_lidar_to_image_with_distortion(
            xyz_lidar=pts,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_pin,
            dist_coeffs=D_pin,
            width=width,
            height=height,
            min_depth=args.min_depth_m,
        ),
        "rational_polynomial": lambda pts: points_project.project_lidar_to_image_with_rational_polynomial(
            xyz_lidar=pts,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_pin,
            dist_coeffs=D_pin,
            width=width,
            height=height,
            min_depth=args.min_depth_m,
        ),
        "fisheye": lambda pts: points_project.project_lidar_to_image_with_fisheye_distortion(
            xyz_lidar=pts,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_fish,
            dist_coeffs=D_fish,
            width=width,
            height=height,
            min_depth=args.min_depth_m,
        ),
    }

    print(
        f"BENCH image={width}x{height}, input_points_raw={xyz_raw.shape[0]}, input_points_used={xyz.shape[0]}, "
        f"sample_step={max(1, args.sample_step)}, max_points={args.max_points}, loops={args.loops}, warmup={args.warmup}"
    )
    print(source_info)
    print(f"pinhole-model={args.pinhole_model}, fisheye-model={args.fisheye_model}")
    print("-" * 96)
    print(f"{'method':22s} {'mean(ms)':>10s} {'std':>8s} {'min':>8s} {'max':>8s} {'out_points':>12s}")
    print("-" * 96)
    for name, fn in methods.items():
        stats = _bench_once(fn, xyz, args.loops, args.warmup)
        print(
            f"{name:22s} {stats['mean_ms']:10.3f} {stats['std_ms']:8.3f} "
            f"{stats['min_ms']:8.3f} {stats['max_ms']:8.3f} {stats['out_mean']:12.1f}"
        )


if __name__ == "__main__":
    main()
