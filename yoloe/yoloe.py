import os

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

from pathlib import Path
import torch
import supervision as sv
from typing import Union
import numpy as np
import cv2

from typing import Optional, List, Sequence


Yoloeroot = os.path.dirname(os.path.abspath(__file__))
from ultralytics.utils import SETTINGS
SETTINGS.update({"weights_dir": str(Path(os.path.join(Yoloeroot, "weights")))})

PathLike = Union[str, os.PathLike]
ImageLike = Union[np.ndarray, PathLike]

modes_ = ("11l", "11m", "11s", "v8l", "v8m", "v8s")

class Yoloe:
    def __init__(self, model_id="11l"):
        self.select_model(model_id)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.caption = "black box"
        self.return_labels = True

    def select_model(self, model_id):
        model_name = Path(os.path.join(Yoloeroot, f"weights/yoloe-{model_id}-seg.pt"))
        print(model_name)
        if model_name.exists():
            self.model = YOLOE(model_name)
        else:
            self.model = YOLOE(os.path.join(Yoloeroot, "weights/yoloe-11l-seg.pt"))
            print("Warning: input model {model} is not exist")

    def setparameters(
        self,
        caption: Optional[str] = None,
        threshold: Optional[float] = None,
        return_labels: Optional[bool] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        if caption is not None:
            self.caption = caption
        if threshold is not None:
            self.threshold = threshold
        if return_labels is not None:
            self.return_labels = return_labels
        if max_detections is not None:
            self.max_detections = max_detections

    @staticmethod
    def _normalize_names(names: Optional[Union[str, Sequence[str]]]) -> List[str]:
        """Normalize class names to the list[str] format expected by YOLOE text encoder."""
        if names is None:
            return []
        if isinstance(names, str):
            s = names.strip()
            return [s] if s else []
        out = [str(x).strip() for x in names if str(x).strip()]
        return out

    def predict(self, image, caption):
        names = self._normalize_names(caption if caption is not None else self.caption)
        if not names:
            names = ["object"]
        self.model.set_classes(names, self.model.get_text_pe(names))
        results = self.model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        labels = []
        if self.return_labels:
            labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)]
        return detections, labels
            
    def predict_with_text(self, image, names):
        if names is None:
            names = self.caption
        names = self._normalize_names(names)
        if not names:
            names = ["object"]
        self.model.set_classes(names, self.model.get_text_pe(names))
        results = self.model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        return detections
    
    def predict_with_visual_prompt(self, source_image, target_image, bbox):
        visual_prompts = {
            "bboxes": [bbox],
            "cls": [0],                        
        }

        results = self.model.predict(
            source=source_image,
            refer_image=target_image,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=0.45,
            save=True,
            project="yoloe/output",
            name="vp_demo",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        detections = sv.Detections.from_ultralytics(results[0])
        if self.return_labels:
            labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)]
        return detections, labels
    
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
    def visualize_detections(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        resolution_wh = image.size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]

        annotated_image = image.copy()
        annotated_image = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.4
        ).annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.BoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=thickness
        ).annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_scale=text_scale,
            smart_position=True
        ).annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
    
    def annotate(self, image: np.ndarray, detections: sv.Detections, labels: List[str]) -> np.ndarray:
        confidence = (
            detections.confidence if getattr(detections, "confidence", None) is not None else np.zeros((len(labels),))
        )
        labels_with_conf = [
            f"{label} {score:.2f}"
            for label, score in zip(labels, confidence)
        ]
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels_with_conf,
        )
        return annotated

def main():
    yoloe = Yoloe("11l")
    yoloe.select_model("11l")
    image = yoloe.read_image(os.path.join(Yoloeroot, "test.jpg"))
    names = ["box"]
    detections = yoloe.predict_with_text(image, names)
    
    annotated_image = yoloe.visualize_detections(image, detections)
    output_path = os.path.join(Yoloeroot, "output", "annotated_image.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main()
