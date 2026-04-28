import cv2
import numpy as np
from sensor_msgs.msg import Image
import os
from typing import Dict, Any
DEFAULT_CAMERA_PARAM_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config/param_camera.yaml",
)

def _ros_image_to_cv2_fallback(ros_image: Image) -> np.ndarray:
    """Decode ROS Image to BGR without relying on cv_bridge runtime libs."""
    h = int(ros_image.height)
    w = int(ros_image.width)
    step = int(ros_image.step)
    enc = (ros_image.encoding or "").lower()
    data = np.frombuffer(ros_image.data, dtype=np.uint8)

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size: h=%d w=%d" % (h, w))
    if step <= 0:
        raise ValueError("Invalid image step: %d" % step)
    if data.size < h * step:
        raise ValueError("Image data too short: bytes=%d expected>=%d" % (data.size, h * step))

    row_view = data[: h * step].reshape((h, step))
    if enc in ("bgr8", "rgb8"):
        row_bytes = w * 3
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        frame = row_view[:, :row_bytes].reshape((h, w, 3))
        if enc == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    if enc in ("bgra8", "rgba8"):
        row_bytes = w * 4
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        frame = row_view[:, :row_bytes].reshape((h, w, 4))
        if enc == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if enc in ("mono8", "8uc1"):
        row_bytes = w
        if step < row_bytes:
            raise ValueError("Image step too small for %s: step=%d need=%d" % (enc, step, row_bytes))
        gray = row_view[:, :row_bytes].reshape((h, w))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unsupported image encoding: %s" % ros_image.encoding)
    return decoded


def _cv2_to_ros_image_fallback(image_bgr: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height = int(image_bgr.shape[0])
    msg.width = int(image_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(image_bgr.shape[1] * 3)
    msg.data = np.ascontiguousarray(image_bgr, dtype=np.uint8).tobytes()
    return msg

   
def _read_yaml_matrix(
    section: Dict[str, Any],
    key: str,
    allow_legacy_inline: bool = False,
) -> np.ndarray:
    block = section.get(key)
    if not isinstance(block, dict):
        if allow_legacy_inline and {"rows", "cols", "data"}.issubset(set(section.keys())):
            block = {
                "rows": section.get("rows"),
                "cols": section.get("cols"),
                "data": section.get("data"),
            }
        else:
            raise KeyError("Missing '%s' section in camera yaml" % key)

    data = np.asarray(block.get("data", []), dtype=np.float64).reshape(-1)
    rows = int(block.get("rows", 0))
    cols = int(block.get("cols", 0))
    if rows <= 0 or cols <= 0:
        raise ValueError("Invalid rows/cols for '%s'" % key)
    if data.size != rows * cols:
        raise ValueError(
            "Invalid '%s' data length: got %d, expected %d"
            % (key, int(data.size), int(rows * cols))
        )
    return data.reshape(rows, cols)   

   
def load_camera_params_from_yaml(
    yaml_path: str = DEFAULT_CAMERA_PARAM_YAML,
    camera_model: str = "fisheye",
) -> Dict[str, Any]:
    """Read camera intrinsics from param_camera.yaml.

    Returns a dict with:
    - camera_name
    - distortion_model
    - K: (3, 3) float64
    - D: (N,) float64
    - R_rect: (3, 3) float64 or identity if missing
    - P: (3, 4) float64 or None if missing
    """
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "load_camera_params_from_yaml requires PyYAML (pip install pyyaml)"
        ) from exc

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if camera_model not in cfg:
        available = ", ".join(sorted(cfg.keys()))
        raise KeyError(
            "Camera model '%s' not found in %s. Available: [%s]"
            % (camera_model, yaml_path, available)
        )

    model_cfg = cfg[camera_model]
    if not isinstance(model_cfg, dict):
        raise ValueError("Camera model '%s' config is not a mapping" % camera_model)

    K_loaded = _read_yaml_matrix(model_cfg, "camera_matrix")
    if K_loaded.shape != (3, 3):
        raise ValueError(
            "camera_matrix for '%s' must be 3x3, got %s"
            % (camera_model, str(K_loaded.shape))
        )

    D_loaded = _read_yaml_matrix(model_cfg, "distortion_coefficients").reshape(-1)

    if "rectification_matrix" in model_cfg:
        R_rect = _read_yaml_matrix(model_cfg, "rectification_matrix")
        if R_rect.shape != (3, 3):
            raise ValueError(
                "rectification_matrix for '%s' must be 3x3, got %s"
                % (camera_model, str(R_rect.shape))
            )
    else:
        R_rect = np.eye(3, dtype=np.float64)

    P = None
    if "projection_matrix" in model_cfg:
        P_loaded = _read_yaml_matrix(
            model_cfg,
            "projection_matrix",
            allow_legacy_inline=True,
        )
        if P_loaded.shape != (3, 4):
            raise ValueError(
                "projection_matrix for '%s' must be 3x4, got %s"
                % (camera_model, str(P_loaded.shape))
            )
        P = P_loaded

    return {
        "camera_name": str(model_cfg.get("camera_name", "")),
        "distortion_model": str(model_cfg.get("distortion_model", camera_model)),
        "K": K_loaded,
        "D": D_loaded,
        "R_rect": R_rect,
        "P": P,
    }

def _estimate_fisheye_theta_limit(
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
) -> float:
    """Estimate the effective half-FOV angle (radians) for current fisheye intrinsics."""
    K_use = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)
    D_all = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if D_all.size < 4:
        return np.deg2rad(89.0)

    D_use = D_all[:4].reshape(4, 1)
    if width <= 4 or height <= 4:
        return np.deg2rad(89.0)

    x_samples = np.linspace(1.0, float(width) - 2.0, 240, dtype=np.float64)
    y_samples = np.linspace(1.0, float(height) - 2.0, 160, dtype=np.float64)
    top = np.stack([x_samples, np.full_like(x_samples, 1.0)], axis=1)
    bottom = np.stack([x_samples, np.full_like(x_samples, float(height) - 2.0)], axis=1)
    left = np.stack([np.full_like(y_samples, 1.0), y_samples], axis=1)
    right = np.stack([np.full_like(y_samples, float(width) - 2.0), y_samples], axis=1)
    border = np.concatenate([top, bottom, left, right], axis=0).reshape(-1, 1, 2)

    undist = cv2.fisheye.undistortPoints(border, K_use, D_use)
    xy = undist.reshape(-1, 2)
    theta = np.arctan(np.linalg.norm(xy, axis=1))
    theta = theta[np.isfinite(theta)]
    if theta.size == 0:
        return np.deg2rad(89.0)

    theta_limit = float(np.percentile(theta, 99.0))
    return float(np.clip(theta_limit, np.deg2rad(5.0), np.deg2rad(89.0)))