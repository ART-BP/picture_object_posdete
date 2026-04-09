from pathlib import Path

import argparse
import cv2
import numpy as np
import torch

from mobile_sam import SamPredictor, sam_model_registry

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "weights" / "mobile_sam.pt"
DEMO_INPUT_DIR = ROOT / "demo" / "demo_input"
DEMO_OUTPUT_DIR = ROOT / "demo" / "demo_output"
# 保留“框接口”：把 GroundingDINO 的 xyxy 直接填到这里即可
# 格式: [x1, y1, x2, y2]，坐标基于原图像素
BOX_XYXY = np.array([250, 140, 380, 300], dtype=np.float32)


def resolve_default_image() -> Path:
    fixed = DEMO_INPUT_DIR / "left0000.jpg"
    if fixed.exists():
        return fixed
    candidates = sorted(DEMO_INPUT_DIR.glob("*.jpg")) + sorted(DEMO_INPUT_DIR.glob("*.png"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No image found in {DEMO_INPUT_DIR}. "
        "Please add an image or pass --image /abs/path/to/image.jpg"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Box-prompt SAM inference test.")
    parser.add_argument("--image", type=str, default=None, help="Input image path.")
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_PATH), help="Model checkpoint.")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        default=BOX_XYXY.tolist(),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Box prompt in XYXY format on original image.",
    )
    return parser.parse_args()


args = parse_args()
img_path = Path(args.image) if args.image else resolve_default_image()
box_xyxy = np.array(args.box, dtype=np.float32)

# 1) 读图与模型
image_bgr = cv2.imread(str(img_path))
if image_bgr is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
sam.to(device=device).eval()
predictor = SamPredictor(sam)
predictor.set_image(image)

# 2) 用框提示分割（接口保留给检测框）
masks, scores, _ = predictor.predict(
    box=box_xyxy,
    multimask_output=False,
)
mask = masks[0]
score = float(scores[0])

# 3) 物体占整图比例
area = int(mask.sum())
ratio = area / (mask.shape[0] * mask.shape[1])
print(f"image = {img_path}")
print(f"box = {box_xyxy.tolist()}, score = {score:.4f}")
print(f"object area = {area}, ratio = {ratio:.4f}")

# 4) 导出“仅物体”图与二值 mask
obj_only = image.copy()
obj_only[~mask] = 0
mask_u8 = mask.astype(np.uint8) * 255
cv2.imwrite(str(DEMO_OUTPUT_DIR / "object_mask.png"), mask_u8)
cv2.imwrite(str(DEMO_OUTPUT_DIR / "object_only.png"), cv2.cvtColor(obj_only, cv2.COLOR_RGB2BGR))

# 5) 导出紧致裁剪图
ys, xs = np.where(mask)
y0, y1 = ys.min(), ys.max()
x0, x1 = xs.min(), xs.max()
obj_crop = obj_only[y0 : y1 + 1, x0 : x1 + 1]
cv2.imwrite(str(DEMO_OUTPUT_DIR / "object_crop.png"), cv2.cvtColor(obj_crop, cv2.COLOR_RGB2BGR))
