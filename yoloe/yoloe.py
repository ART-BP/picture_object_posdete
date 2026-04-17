import argparse
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

from pathlib import Path
import torch

Yoloeroot = os.path.dirname(os.path.abspath(__file__))

class Yoloe:
    def __init__(self):

        self.model = YOLOE()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def select_model(self, model_id):
        model_name = Path(f"weights/yoloe-11{model_id}-seg.pt")
        if model_name.exists():
            self.model = YOLOE(model_name)
        else:
            self.model = YOLOE(os.path.join(Yoloeroot, "weights/yoloe-11l-seg.pt"))
            print("Warning: input model {model} is not exist")

    def predict_with_text(self, image, names):
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
            conf=0.25,
            save=True,
            project="yoloe/output",
            name="vp_demo",
            device="cpu",  # 或 cuda:0
        )
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

def main():
    yoloe = Yoloe()
    yoloe.select_model("l")

    image = Image.open(os.path.join(Yoloeroot, "data/demo.jpg"))
    names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"]
    detections = yoloe.predict_with_text(image, names)
    print(detections)

if __name__ == "__main__":
    main()