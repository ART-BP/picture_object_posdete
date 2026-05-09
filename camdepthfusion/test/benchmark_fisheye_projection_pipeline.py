#!/usr/bin/env python3
"""Compare timing: direct fisheye projection vs undistort-then-project."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camdepthfusion.camera_op import camera_handle
from camdepthfusion.project_cloudpoints import cloudpoints_handle
from camdepthfusion.project_cloudpoints import points_project


def _as_float_seconds(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp)


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
    if d_raw.size < 4:
        raise ValueError("fisheye distortion_coefficients must have at least 4 values")
    return k_raw.reshape(3, 3), d_raw


def _load_first_matched_bag_frame(
    bag_path: Path,
    cloud_topic: str,
    image_topic: str,
    sync_slop: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    try:
        import rosbag
    except Exception as exc:
        raise ImportError(
            "import rosbag failed. Please source ROS env first "
            "(e.g. `source /opt/ros/noetic/setup.bash`) and ensure rosbag deps are installed. "
            f"Original error: {exc}"
        ) from exc

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
            image = camera_handle._ros_image_to_cv2_fallback(latest_image_msg)
            if xyz.shape[0] == 0:
                continue
            if image is None or image.size == 0:
                continue
            return xyz, image, float(latest_cloud_sec), float(latest_image_sec), float(dt)

    raise RuntimeError(
        f"no matched frame found in bag={bag_path} for topics {cloud_topic} / {image_topic} "
        f"within sync_slop={sync_slop}s"
    )


def _gen_random_input(num_points: int, width: int, height: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 20.0, size=num_points)
    y = rng.uniform(-20.0, 20.0, size=num_points)
    z = rng.uniform(-6.0, 6.0, size=num_points)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    image = (rng.random((height, width, 3)) * 255.0).astype(np.uint8)
    return xyz, image


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


def _bench_projection(
    fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    loops: int,
    warmup: int,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    times_ms = []
    out_counts = []
    for _ in range(loops):
        t0 = time.perf_counter()
        xyz_out, _, _ = fn()
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


def _build_fisheye_undistort_map(
    K: np.ndarray,
    D: np.ndarray,
    width: int,
    height: int,
    fisheye_balance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_use = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D_use = np.asarray(D, dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
    R_rect = np.eye(3, dtype=np.float64)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K=K_use,
        D=D_use,
        image_size=(width, height),
        R=R_rect,
        balance=float(np.clip(fisheye_balance, 0.0, 1.0)),
        new_size=(width, height),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K=K_use,
        D=D_use,
        R=R_rect,
        P=K_new,
        size=(width, height),
        m1type=cv2.CV_16SC2,
    )
    return K_new, map1, map2


def main() -> None:
    default_yaml = REPO_ROOT / "camdepthfusion" / "camera_op" / "config" / "param_camera.yaml"
    default_bag = REPO_ROOT / "bag" / "two_person.bag"

    parser = argparse.ArgumentParser(
        description="Compare fisheye direct projection vs undistort-then-project timing."
    )
    parser.add_argument("--source", choices=["bag", "random"], default="bag")
    parser.add_argument("--bag", type=Path, default=default_bag)
    parser.add_argument("--cloud-topic", type=str, default="/lidar_points")
    parser.add_argument("--image-topic", type=str, default="/camera/go2/front/image_raw")
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--yaml", type=Path, default=default_yaml)
    parser.add_argument("--camera-model", type=str, default="fisheye")
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--theta-margin-deg", type=float, default=1.0)
    parser.add_argument("--fisheye-balance", type=float, default=0.0)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=120000)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    K, D = _load_camera_model_params(args.yaml, args.camera_model)
    R = points_project.R
    T = points_project.T

    if args.source == "bag":
        xyz_raw, image_bgr, cloud_sec, image_sec, dt = _load_first_matched_bag_frame(
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
        xyz_raw, image_bgr = _gen_random_input(args.num_points, args.width, args.height, args.seed)
        source_info = (
            f"source=random num_points={args.num_points} image={args.width}x{args.height} seed={args.seed}"
        )

    xyz = _prepare_xyz(xyz_raw, args.sample_step, args.max_points)
    if xyz.shape[0] == 0:
        raise RuntimeError("input point cloud is empty after sampling/capping")
    h, w = image_bgr.shape[:2]

    K_new_pre, map1, map2 = _build_fisheye_undistort_map(
        K=K,
        D=D,
        width=w,
        height=h,
        fisheye_balance=args.fisheye_balance,
    )

    def _direct_fisheye_projection():
        return points_project.project_lidar_to_image_with_fisheye_distortion(
            xyz_lidar=xyz,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K,
            dist_coeffs=D,
            width=w,
            height=h,
            min_depth=args.min_depth_m,
            theta_margin_deg=args.theta_margin_deg,
        )

    def _undistort_each_then_project():
        _, K_new_each = camera_handle.undistort_image(
            image=image_bgr,
            K_camera=K,
            dist_coeffs=D,
            distortion_model="fisheye",
            fisheye_balance=args.fisheye_balance,
        )
        return points_project.project_lidar_to_image(
            xyz_lidar=xyz,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_new_each,
            width=w,
            height=h,
            dist_coeffs=np.zeros((4,), dtype=np.float64),
            min_depth=args.min_depth_m,
        )

    def _undistort_precomputed_then_project():
        _ = cv2.remap(image_bgr, map1, map2, cv2.INTER_LINEAR)
        return points_project.project_lidar_to_image(
            xyz_lidar=xyz,
            R_optical_lidar=R,
            t_optical_lidar=T,
            K_camera=K_new_pre,
            width=w,
            height=h,
            dist_coeffs=np.zeros((4,), dtype=np.float64),
            min_depth=args.min_depth_m,
        )

    direct_stats = _bench_projection(_direct_fisheye_projection, args.loops, args.warmup)
    undist_each_stats = _bench_projection(_undistort_each_then_project, args.loops, args.warmup)
    undist_pre_stats = _bench_projection(_undistort_precomputed_then_project, args.loops, args.warmup)

    print(
        f"BENCH image={w}x{h}, raw_points={xyz_raw.shape[0]}, used_points={xyz.shape[0]}, "
        f"sample_step={max(1, args.sample_step)}, max_points={args.max_points}, "
        f"loops={args.loops}, warmup={args.warmup}"
    )
    print(source_info)
    print(f"camera_model={args.camera_model}, fisheye_balance={args.fisheye_balance}")
    print("-" * 112)
    print(f"{'pipeline':40s} {'mean(ms)':>10s} {'std':>8s} {'min':>8s} {'max':>8s} {'out_points':>12s}")
    print("-" * 112)

    def _fmt(name: str, st: Dict[str, float]) -> str:
        return (
            f"{name:40s} {st['mean_ms']:10.3f} {st['std_ms']:8.3f} "
            f"{st['min_ms']:8.3f} {st['max_ms']:8.3f} {st['out_mean']:12.1f}"
        )

    print(_fmt("A) direct fisheye projection", direct_stats))
    print(_fmt("B1) undistort(each frame)+pinhole proj", undist_each_stats))
    print(_fmt("B2) undistort(precomputed map)+proj", undist_pre_stats))
    print("-" * 112)
    print(
        f"time_ratio(B1/A)={undist_each_stats['mean_ms'] / max(direct_stats['mean_ms'], 1e-9):.2f}x, "
        f"time_ratio(B2/A)={undist_pre_stats['mean_ms'] / max(direct_stats['mean_ms'], 1e-9):.2f}x "
        "(<1 means faster than direct; >1 means slower)"
    )


if __name__ == "__main__":
    main()
