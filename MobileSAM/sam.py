from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry .* name being registered conflicts with an existing name\.",
    category=UserWarning,
)

from mobile_sam import SamPredictor, sam_model_registry


ArrayLike = np.ndarray
PathLike = Union[str, Path]


class Sam:
    """Thin wrapper around MobileSAM for image + prompt inference."""

    def __init__(
        self,
        checkpoint: Optional[PathLike] = None,
        model_type: str = "vit_t",
        device: Optional[str] = None,
        use_amp: Optional[bool] = None,
    ) -> None:
        ckpt = Path(checkpoint) if checkpoint is not None else root / "weights" / "mobile_sam.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_amp is None:
            env_flag = str(os.environ.get("MOBILESAM_USE_AMP", "1")).strip().lower()
            use_amp = env_flag not in ("0", "false", "no", "off")

        self.device = device
        self.use_amp = bool(use_amp) and str(self.device).startswith("cuda") and torch.cuda.is_available()
        self.model = sam_model_registry[model_type](checkpoint=str(ckpt))
        self.model.to(device=self.device).eval()
        self.predictor = SamPredictor(self.model)
        self.image_rgb: Optional[ArrayLike] = None

    def _autocast_ctx(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except Exception:
            return torch.amp.autocast("cuda", enabled=True)

    @staticmethod
    def _read_image(image: Union[PathLike, ArrayLike], image_format: str = "BGR") -> ArrayLike:
        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image))
            if image_bgr is None:
                raise FileNotFoundError(f"Image not found: {image}")
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a path or np.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got shape={image.shape}")

        fmt = image_format.upper()
        if fmt == "RGB":
            return image
        if fmt == "BGR":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise ValueError("image_format must be 'RGB' or 'BGR'")

    def set_image(self, image: Union[PathLike, ArrayLike], image_format: str = "BGR") -> None:
        self.image_rgb = self._read_image(image, image_format=image_format)
        with torch.inference_mode():
            with self._autocast_ctx():
                self.predictor.set_image(self.image_rgb)

    @staticmethod
    def _ensure_xyxy(box_xyxy: Union[Tuple[float, float, float, float], ArrayLike]) -> ArrayLike:
        box = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if box.size != 4:
            raise ValueError(f"box must have 4 values [x1,y1,x2,y2], got shape={box.shape}")
        x1, y1, x2, y2 = box.tolist()
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid box: {box.tolist()}, require x2>x1 and y2>y1")
        return box

    @staticmethod
    def best_mask(masks: ArrayLike, scores: ArrayLike) -> Tuple[ArrayLike, float, int]:
        idx = int(np.argmax(scores))
        return masks[idx], float(scores[idx]), idx

    def predict_with_box(
        self,
        box_xyxy: Union[Tuple[float, float, float, float], ArrayLike],
        image: Optional[Union[PathLike, ArrayLike]] = None,
        image_format: str = "BGR",
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        if image is not None:
            self.set_image(image, image_format=image_format)
        if self.image_rgb is None:
            raise RuntimeError("Call set_image(...) first, or pass image=... to predict_with_box().")

        box = self._ensure_xyxy(box_xyxy)
        with torch.inference_mode():
            with self._autocast_ctx():
                return self.predictor.predict(
                    box=box,
                    multimask_output=multimask_output,
                    return_logits=return_logits,
                )

    def get_mask_by_box(
        self,
        box_xyxy: Union[Tuple[float, float, float, float], ArrayLike],
        image: Optional[Union[PathLike, ArrayLike]] = None,
        image_format: str = "BGR",
        multimask_output: bool = False,
    ) -> Tuple[ArrayLike, float, int]:
        masks, scores, _ = self.predict_with_box(
            box_xyxy=box_xyxy,
            image=image,
            image_format=image_format,
            multimask_output=multimask_output,
            return_logits=False,
        )
        return self.best_mask(masks, scores)

    def predict_with_points(
        self,
        point_coords: ArrayLike,
        point_labels: ArrayLike,
        image: Optional[Union[PathLike, ArrayLike]] = None,
        image_format: str = "BGR",
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        if image is not None:
            self.set_image(image, image_format=image_format)
        if self.image_rgb is None:
            raise RuntimeError("Call set_image(...) first, or pass image=... to predict_with_points().")

        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("point_coords must be Nx2")
        if labels.ndim != 1 or labels.shape[0] != coords.shape[0]:
            raise ValueError("point_labels must be N and match point_coords")

        with torch.inference_mode():
            with self._autocast_ctx():
                return self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=multimask_output,
                    return_logits=return_logits,
                )

    def get_mask_by_points(
        self,
        point_coords: ArrayLike,
        point_labels: ArrayLike,
        image: Optional[Union[PathLike, ArrayLike]] = None,
        image_format: str = "BGR",
        multimask_output: bool = True,
    ) -> Tuple[ArrayLike, float, int]:
        masks, scores, _ = self.predict_with_points(
            point_coords=point_coords,
            point_labels=point_labels,
            image=image,
            image_format=image_format,
            multimask_output=multimask_output,
            return_logits=False,
        )
        return self.best_mask(masks, scores)

    @staticmethod
    def object_ratio(mask: ArrayLike) -> float:
        return float(mask.sum()) / float(mask.shape[0] * mask.shape[1])

    @staticmethod
    def render_object(image_rgb: ArrayLike, mask: ArrayLike) -> ArrayLike:
        out = image_rgb.copy()
        out[~mask] = 0
        return out

    @staticmethod
    def crop_by_mask(image_rgb: ArrayLike, mask: ArrayLike) -> ArrayLike:
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return np.zeros((1, 1, 3), dtype=image_rgb.dtype)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return image_rgb[y0 : y1 + 1, x0 : x1 + 1]

    @staticmethod
    def save_mask(mask: ArrayLike, path: PathLike) -> None:
        mask_u8 = mask.astype(np.uint8) * 255
        cv2.imwrite(str(path), mask_u8)

    @staticmethod
    def save_rgb(image_rgb: ArrayLike, path: PathLike) -> None:
        cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
