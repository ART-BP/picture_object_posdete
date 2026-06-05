#!/usr/bin/env python3

import argparse

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


# Fill these with your calibration values.
K = np.array(
    [
        [1130.1035455064891, 0.0, 951.0193473268226],
        [0.0, 1130.1541762377883, 581.0902322193278],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D = np.array(
    [-0.10949451690742605, 0.01563345714193377, -0.07072000499253979, 0.04326396138302371],
    dtype=np.float64,
).reshape(4, 1)
R_LIDAR_TO_CAMERA = np.array(
    [
        [-0.023465, -0.999725, 0.000279],
        [-0.009149, -0.000065, -0.999958],
        [0.999683, -0.023467, -0.009145],
    ],
    dtype=np.float64,
)
T_LIDAR_TO_CAMERA = np.array([0.101330, -0.108852, -0.082845], dtype=np.float64)

def first_cloud_xyz(bag_path, topic):
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            points = np.asarray(
                list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                dtype=np.float64,
            )
            if points.size:
                return points.reshape(-1, 3)
    raise RuntimeError("No non-empty PointCloud2 frame found on topic: %s" % topic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="mid_f.jpg")
    parser.add_argument("bag", nargs="?", default="mid_f.bag")
    parser.add_argument("output", nargs="?", default="mid.jpg")
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument("--balance", type=float, default=0.0)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError("Cannot read image: %s" % args.image)

    height, width = image.shape[:2]
    size = (width, height)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=float(np.clip(args.balance, 0.0, 1.0))
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, size, cv2.CV_16SC2
    )
    output = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    xyz_lidar = first_cloud_xyz(args.bag, args.topic)
    xyz_camera = (R_LIDAR_TO_CAMERA @ xyz_lidar.T).T + T_LIDAR_TO_CAMERA
    valid = xyz_camera[:, 2] > 0.1
    xyz_camera = xyz_camera[valid]

    uvw = (K_new @ xyz_camera.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv = np.rint(uv[inside]).astype(np.int32)
    depth = xyz_camera[inside, 2]

    if depth.size:
        depth_norm = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-6)
        colors = cv2.applyColorMap(
            np.asarray(255 * (1.0 - depth_norm), dtype=np.uint8).reshape(-1, 1),
            cv2.COLORMAP_JET,
        ).reshape(-1, 3)
        for (u, v), color in zip(uv, colors):
            cv2.circle(output, (int(u), int(v)), 1, tuple(int(c) for c in color), -1)

    if not cv2.imwrite(args.output, output):
        raise RuntimeError("Cannot write output image: %s" % args.output)
    print("saved=%s input_points=%d projected_points=%d" % (args.output, len(xyz_lidar), len(uv)))


if __name__ == "__main__":
    main()
