#!/usr/bin/env python3
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import torch

try:
    import torchvision

    disable_beta_warning = getattr(torchvision, "disable_beta_transforms_warning", None)
    if callable(disable_beta_warning):
        disable_beta_warning()
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r"The `device` argument is deprecated and will be removed in v5 of Transformers\.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"None of the inputs have requires_grad=True\. Gradients will be None",
    category=UserWarning,
)

root_gdino = os.path.dirname(os.path.abspath(__file__))
if root_gdino not in sys.path:
    sys.path.insert(0, root_gdino)

from groundingdino.util.inference import Model


PathLike = Union[str, os.PathLike]
ImageLike = Union[np.ndarray, PathLike]


class GroundingDINO(Model):
    """GroundingDINO wrapper ."""

    def __init__(
        self,
        model_config_path: Optional[PathLike] = None,
        model_checkpoint_path: Optional[PathLike] = None,
        device: Optional[str] = None,
        caption: str = "black box",
    ) -> None:
        model_config = Path(model_config_path) if model_config_path is not None else self.default_model_config_path()
        model_checkpoint = (
            Path(model_checkpoint_path)
            if model_checkpoint_path is not None
            else self.default_model_checkpoint_path()
        )
        if not model_config.exists():
            raise FileNotFoundError(f"Model config not found: {model_config}")
        if not model_checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            model_config_path=str(model_config),
            model_checkpoint_path=str(model_checkpoint),
            device=device,
        )
        self.model_config_path = str(model_config)
        self.model_checkpoint_path = str(model_checkpoint)
        self.caption = caption
        self.box_threshold = 0.40
        self.text_threshold = 0.25
        self.return_labels = True
        self.max_detections = 5

        self.image: Optional[np.ndarray] = None
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.name = "gdino"
    def setparameters(
        self,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        return_labels: Optional[bool] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        if caption is not None:
            self.caption = caption
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold
        if return_labels is not None:
            self.return_labels = return_labels
        if max_detections is not None:
            self.max_detections = max_detections
            
    @staticmethod
    def default_model_config_path() -> Path:
        return Path(root_gdino) / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"

    @staticmethod
    def default_model_checkpoint_path() -> Path:
        return Path(root_gdino) / "weights" / "groundingdino_swint_ogc.pth"

    @staticmethod
    def read_image(image_path: PathLike) -> np.ndarray:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return image

    @staticmethod
    def _ensure_bgr(image: ImageLike, image_format: str = "BGR") -> np.ndarray:
        if isinstance(image, (str, os.PathLike)):
            return GroundingDINO.read_image(image)

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a path or np.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got shape={image.shape}")

        fmt = image_format.upper()
        if fmt == "BGR":
            return image
        if fmt == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError("image_format must be 'BGR' or 'RGB'")

    def set_image(self, image: ImageLike, image_format: str = "BGR") -> np.ndarray:
        self.image = self._ensure_bgr(image, image_format=image_format)
        return self.image

    def predict(
        self,
        image: Optional[ImageLike] = None,
        caption: Optional[str] = None,
    ) -> Tuple[sv.Detections, List[str]]:
        if image is not None:
            current_image = self.set_image(image, image_format="BGR")
        elif self.image is not None:
            current_image = self.image
        else:
            raise RuntimeError("Call set_image(...) first, or pass image=... to predict().")

        caption_text = caption if caption is not None else self.caption
        return self.predict_with_caption(
            image=current_image,
            caption=caption_text,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            return_phrases=self.return_labels,
            max_detections=self.max_detections,
        )

    def predict_from_path(
        self,
        image_path: PathLike,
        caption: Optional[str] = None,
        box_threshold: float = 0.40,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, sv.Detections, List[str]]:
        image = self.set_image(image_path)
        detections, labels = self.predict(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return image, detections, labels

    @staticmethod
    def extract_box_info(
        detections: sv.Detections,
        labels: List[str],
        image: np.ndarray,
    ) -> List[Dict]:
        h, w = image.shape[:2]
        boxes = np.asarray(detections.xyxy) if detections is not None else np.zeros((0, 4))
        confidence = (
            np.asarray(detections.confidence)
            if getattr(detections, "confidence", None) is not None
            else np.zeros((len(boxes),), dtype=np.float32)
        )

        results: List[Dict] = []
        for idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = [float(v) for v in box.tolist()]
            x_min = float(np.clip(x_min, 0, max(0, w - 1)))
            y_min = float(np.clip(y_min, 0, max(0, h - 1)))
            x_max = float(np.clip(x_max, 0, max(0, w - 1)))
            y_max = float(np.clip(y_max, 0, max(0, h - 1)))

            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            area = width * height
            score = float(confidence[idx]) if idx < len(confidence) else 0.0
            label = labels[idx] if idx < len(labels) else ""

            results.append(
                {
                    "id": idx,
                    "label": label,
                    "score": score,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": width,
                    "height": height,
                    "area": area,
                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                    "bbox_cxcywh": [
                        x_min + width / 2.0,
                        y_min + height / 2.0,
                        width,
                        height,
                    ],
                    "bbox_xyxy_norm": [
                        x_min / w if w > 0 else 0.0,
                        y_min / h if h > 0 else 0.0,
                        x_max / w if w > 0 else 0.0,
                        y_max / h if h > 0 else 0.0,
                    ],
                }
            )
        return results

    def build_bbox_payload(
        self,
        detections: sv.Detections,
        labels: List[str],
        image: Optional[np.ndarray] = None,
        caption: Optional[str] = None,
        stamp: Optional[float] = None,
        frame_id: str = "",
    ) -> Dict:
        current_image = image if image is not None else self.image
        if current_image is None:
            raise ValueError("No current image available for bbox payload.")

        img_h, img_w = current_image.shape[:2]
        objects = self.extract_box_info(detections=detections, labels=labels, image=current_image)
        payload = {
            "stamp": float(stamp) if stamp is not None else None,
            "frame_id": frame_id,
            "image_width": int(img_w),
            "image_height": int(img_h),
            "caption": caption if caption is not None else self.caption,
            "num_detections": len(objects),
            "detections": objects,
        }
        return payload

    def annotate(self, image: np.ndarray, detections: sv.Detections, labels: List[str]) -> np.ndarray:
        confidence = (
            detections.confidence if getattr(detections, "confidence", None) is not None else np.zeros((len(labels),))
        )
        labels_with_conf = [
            f"{label} {score:.2f}"
            for label, score in zip(labels, confidence)
        ]
        annotated = self.box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels_with_conf,
        )
        return annotated

    def predict_and_annotate(
        self,
        image: Optional[ImageLike] = None,
        caption: Optional[str] = None,
        box_threshold: float = 0.40,
        text_threshold: float = 0.25,
        image_format: str = "BGR",
    ) -> Tuple[np.ndarray, sv.Detections, List[str], np.ndarray]:
        detections, labels = self.predict(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_format=image_format,
        )
        if self.image is None:
            raise RuntimeError("Image cache is empty after prediction.")
        annotated = self.annotate(self.image, detections, labels)
        return self.image, detections, labels, annotated

    def save_annotated(
        self,
        save_path: PathLike,
        image: Optional[ImageLike] = None,
        caption: Optional[str] = None,
        box_threshold: float = 0.40,
        text_threshold: float = 0.25,
        image_format: str = "BGR",
    ) -> str:
        _, _, _, annotated = self.predict_and_annotate(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_format=image_format,
        )
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), annotated):
            raise ValueError(f"Failed to write image: {save_path}")
        return str(save_path)
