#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict

import cv2
import numpy as np
from camera_handle import _estimate_fisheye_theta_limit

MAX_OVERLAY_POINTS = 20000
POINT_RADIUS = 1
MIN_DEPTH = 0.1
MIN_OBJECT_POINTS = 5

AXIS_REMAP = np.array(
    [
        0.0,
        -1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float64,
).reshape(3, 3)

R = np.array(
    [
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
).reshape(3, 3)

T = np.array([0.05, 0.08, 0.182], dtype=np.float64)


def project_lidar_to_image(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    width: int,
    height: int,
    dist_coeffs: np.ndarray,
    min_depth: float,
):
    """Map lidar points to image pixels on undistorted image."""
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_cam = (R_optical_lidar @ xyz_lidar.T).T + t_optical_lidar.reshape(1, 3)
    valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > float(min_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar[valid]
    xyz_cam = xyz_cam[valid]

    # Keep D for compatibility; projection assumes undistorted image.
    _ = dist_coeffs
    uvw = (K_camera @ xyz_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    if not np.any(inside):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_inside = xyz_lidar_valid[inside].astype(np.float32, copy=False)
    uv_inside = uv[inside].astype(np.float32, copy=False)
    depth_inside = xyz_cam[inside, 2].astype(np.float32, copy=False)
    return xyz_inside, uv_inside, depth_inside


def project_lidar_to_image_with_distortion(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
):
    """Map lidar points to image pixels using distortion coefficients.

    This function uses cv2.projectPoints, so it is suitable for raw (distorted)
    camera images when K and D are from the same calibration model.
    """
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_cam = (R_optical_lidar @ xyz_lidar.T).T + t_optical_lidar.reshape(1, 3)
    valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > float(min_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar[valid]
    xyz_cam = xyz_cam[valid].astype(np.float64, copy=False)
    K_use = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)
    D_use = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(
        objectPoints=xyz_cam,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K_use,
        distCoeffs=D_use,
    )
    uv = uv.reshape(-1, 2)

    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    if not np.any(inside):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_inside = xyz_lidar_valid[inside].astype(np.float32, copy=False)
    uv_inside = uv[inside].astype(np.float32, copy=False)
    depth_inside = xyz_cam[inside, 2].astype(np.float32, copy=False)
    return xyz_inside, uv_inside, depth_inside


def project_lidar_to_image_with_rational_polynomial(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
):
    """Map lidar points to raw image pixels using rational_polynomial distortion.

    Typical ROS rational_polynomial coefficients are:
    [k1, k2, p1, p2, k3, k4, k5, k6] (8) with optional extension to
    thin-prism/tilt terms (12 or 14 total terms).
    """
    D_all = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if D_all.size < 8:
        raise ValueError(
            "rational_polynomial requires at least 8 coefficients "
            "[k1, k2, p1, p2, k3, k4, k5, k6]"
        )

    # OpenCV projectPoints commonly accepts distortion lengths 4/5/8/12/14.
    # For rational_polynomial, normalize to 8/12/14 to keep behavior stable.
    if D_all.size <= 8:
        D_use = np.zeros((8,), dtype=np.float64)
        D_use[: D_all.size] = D_all
    elif D_all.size <= 12:
        D_use = np.zeros((12,), dtype=np.float64)
        D_use[: D_all.size] = D_all
    elif D_all.size <= 14:
        D_use = np.zeros((14,), dtype=np.float64)
        D_use[: D_all.size] = D_all
    else:
        D_use = D_all[:14]

    return project_lidar_to_image_with_distortion(
        xyz_lidar=xyz_lidar,
        R_optical_lidar=R_optical_lidar,
        t_optical_lidar=t_optical_lidar,
        K_camera=K_camera,
        dist_coeffs=D_use,
        width=width,
        height=height,
        min_depth=min_depth,
    )


def project_lidar_to_image_with_fisheye_distortion(
    xyz_lidar: np.ndarray,
    R_optical_lidar: np.ndarray,
    t_optical_lidar: np.ndarray,
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    theta_margin_deg: float = 1.0,
):
    """Map lidar points to image pixels using the OpenCV fisheye model.

    Use this for raw fisheye images when `K_camera` and `dist_coeffs` are from
    `cv2.fisheye.calibrate` (D = [k1, k2, k3, k4]).
    """
    if xyz_lidar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_cam = (R_optical_lidar @ xyz_lidar.T).T + t_optical_lidar.reshape(1, 3)
    valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > float(min_depth))
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar[valid]
    xyz_cam_valid = xyz_cam[valid].astype(np.float64, copy=False)

    theta_limit = _estimate_fisheye_theta_limit(
        K_camera=K_camera,
        dist_coeffs=dist_coeffs,
        width=width,
        height=height,
    ) + np.deg2rad(float(theta_margin_deg))
    theta_points = np.arctan2(
        np.linalg.norm(xyz_cam_valid[:, :2], axis=1),
        xyz_cam_valid[:, 2],
    )
    in_fov = theta_points <= theta_limit
    if not np.any(in_fov):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_lidar_valid = xyz_lidar_valid[in_fov]
    xyz_cam_valid = xyz_cam_valid[in_fov]
    xyz_cam = xyz_cam_valid.reshape(-1, 1, 3)
    K_use = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)
    D_all = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if D_all.size < 4:
        raise ValueError(
            "Fisheye distortion requires at least 4 coefficients: [k1, k2, k3, k4]"
        )
    D_use = D_all[:4].reshape(4, 1)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.fisheye.projectPoints(
        objectPoints=xyz_cam,
        rvec=rvec,
        tvec=tvec,
        K=K_use,
        D=D_use,
    )
    uv = uv.reshape(-1, 2)

    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    if not np.any(inside):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_inside = xyz_lidar_valid[inside].astype(np.float32, copy=False)
    uv_inside = uv[inside].astype(np.float32, copy=False)
    depth_inside = xyz_cam[inside, 0, 2].astype(np.float32, copy=False)
    return xyz_inside, uv_inside, depth_inside


def draw_overlay(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    count = int(uv.shape[0])
    if count <= 0:
        return overlay

    if count > MAX_OVERLAY_POINTS:
        idx = np.linspace(0, count - 1, MAX_OVERLAY_POINTS).astype(np.int32)
        uv = uv[idx]
        depth = depth[idx]

    min_d = float(np.min(depth))
    max_d = float(np.max(depth))
    denom = max(max_d - min_d, 1e-6)
    depth_norm = ((depth - min_d) / denom).astype(np.float32)
    color_idx = np.maximum(255.0 * 5.0 * (1.0 - depth_norm), 0).astype(np.uint8)
    
    colors = cv2.applyColorMap(color_idx.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    uv_int = np.round(uv).astype(np.int32)
    for i in range(uv_int.shape[0]):
        u_i = int(uv_int[i, 0])
        v_i = int(uv_int[i, 1])
        c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(overlay, (u_i, v_i), int(POINT_RADIUS), c, -1, lineType=cv2.LINE_AA)

    return overlay
