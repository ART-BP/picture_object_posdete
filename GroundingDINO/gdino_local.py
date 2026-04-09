#!/usr/bin/env python3
import argparse
import os

import cv2
import torch

try:
    from GroundingDINO.gdino import GroundingDINO, root_gdino
except ImportError:
    from gdino import GroundingDINO, root_gdino


class GroundingDINOLocal(GroundingDINO):
    pass


def parse_args():
    parser = argparse.ArgumentParser("GroundingDINO local-image box size extractor", add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(root_gdino, "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        help="path to model config",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(root_gdino, "weights/groundingdino_swint_ogc.pth"),
        help="path to model checkpoint",
    )
    parser.add_argument("--image", type=str, required=True, help="path to local image")
    parser.add_argument("--caption", type=str, default="black box", help="text prompt")
    parser.add_argument("--box-threshold", type=float, default=0.40, help="box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--save-annotated",
        type=str,
        required=True,
        help="optional output path for annotated image, e.g. demo_output/local_result.jpg",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image = os.path.join(root_gdino, "demo_input", args.image)
    save_annotate = os.path.join(root_gdino, "demo_output", args.save_annotated)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = GroundingDINOLocal(
        model_config_path=args.config,
        model_checkpoint_path=args.weights,
        device=device,
    )

    image = model.read_image(image)
    detections, labels = model.predict(
        image=image,
        caption=args.caption,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    box_info = model.extract_box_info(detections, labels, image)
    h, w = image.shape[:2]
    print(f"image: {image}")
    print(f"image_size: width={w}, height={h}")
    print(f"detections: {len(box_info)}")
    for item in box_info:
        print(
            f"[{item['id']}] label='{item['label']}' score={item['score']:.3f} "
            f"xyxy=({item['x_min']:.1f}, {item['y_min']:.1f}, {item['x_max']:.1f}, {item['y_max']:.1f}) "
            f"size=({item['width']:.1f} x {item['height']:.1f}) area={item['area']:.1f}"
        )

    out_dir = os.path.dirname(save_annotate)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    annotated = model.annotate(image=image, detections=detections, labels=labels)
    cv2.imwrite(save_annotate, annotated)
    print(f"annotated image saved: {save_annotate}")


if __name__ == "__main__":
    main()
