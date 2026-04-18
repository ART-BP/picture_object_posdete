import os
import time
from typing import List

import cv2
import numpy as np
import rospy
import torch

from GroundingDINO import gdino
from MobileSAM.sam import Sam
from yoloe.yoloe import Yoloe

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(vals: List[float]) -> str:
    if not vals:
        return "n=0"
    arr = np.asarray(vals, dtype=np.float64)
    return (
        f"n={arr.size}, avg={arr.mean():.4f}s, median={np.median(arr):.4f}s, "
        f"p90={np.percentile(arr, 90):.4f}s, min={arr.min():.4f}s, max={arr.max():.4f}s"
    )


def main():
    rospy.init_node("test_gdino_node")

    image_path = rospy.get_param("~image", default=os.path.join(rootdir, "bag/test.jpg"))
    caption = rospy.get_param("~caption", "black box")
    box_threshold = float(rospy.get_param("~box_threshold", 0.45))
    text_threshold = float(rospy.get_param("~text_threshold", 0.35))
    warmup_iters = int(rospy.get_param("~warmup_iters", 5))
    test_iters = int(rospy.get_param("~test_iters", 20))
    sleep_s = float(rospy.get_param("~sleep_s", 0.0))
    resize_scale = float(rospy.get_param("~resize_scale", 1.0))
    reuse_sam_image_embedding = _as_bool(rospy.get_param("~reuse_sam_image_embedding", False))

    t0 = time.perf_counter()
    model = gdino.GroundingDINO()
    model.setparameters(caption=caption, box_threshold=box_threshold, text_threshold=text_threshold)
    
    sam_model = Sam()
    t1 = time.perf_counter()
    rospy.loginfo("construct model cost: %.4fs", t1 - t0)

    image = model.read_image(image_path)
    if resize_scale > 0.0 and resize_scale != 1.0:
        h, w = image.shape[:2]
        new_w = max(1, int(round(w * resize_scale)))
        new_h = max(1, int(round(h * resize_scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rospy.loginfo("resize image: (%d, %d) -> (%d, %d)", w, h, new_w, new_h)

    # Optional: only valid for static-image benchmarking.
    if reuse_sam_image_embedding:
        sam_model.set_image(image=image, image_format="BGR")

    rospy.loginfo(
        "benchmark config: warmup=%d, test_iters=%d, caption='%s', reuse_sam_image_embedding=%s",
        warmup_iters,
        test_iters,
        caption,
        str(reuse_sam_image_embedding),
    )

    for _ in range(max(0, warmup_iters)):
        if rospy.is_shutdown():
            return
        _sync_cuda()
        detections, _ = model.predict(
            image=image,
            caption=caption,
        )
        if len(detections.xyxy) > 0:
            xyxy = detections.xyxy[0]
            if reuse_sam_image_embedding:
                sam_model.get_mask_by_box(box_xyxy=xyxy, multimask_output=False)
            else:
                sam_model.get_mask_by_box(
                    box_xyxy=xyxy,
                    image=image,
                    image_format="BGR",
                    multimask_output=False,
                )
        _sync_cuda()

    total_costs: List[float] = []
    gdino_costs: List[float] = []
    sam_costs: List[float] = []

    for i in range(max(0, test_iters)):
        if rospy.is_shutdown():
            break

        _sync_cuda()
        t_start = time.perf_counter()

        detections, _ = model.predict(
            image=image,
            caption=caption,
        )

        _sync_cuda()
        t_gdino = time.perf_counter()

        if len(detections.xyxy) > 0:
            xyxy = detections.xyxy[0]
            if reuse_sam_image_embedding:
                sam_model.get_mask_by_box(box_xyxy=xyxy, multimask_output=False)
            else:
                sam_model.get_mask_by_box(
                    box_xyxy=xyxy,
                    image=image,
                    image_format="BGR",
                    multimask_output=False,
                )

        _sync_cuda()
        t_end = time.perf_counter()

        gdino_dt = t_gdino - t_start
        sam_dt = t_end - t_gdino
        total_dt = t_end - t_start
        gdino_costs.append(gdino_dt)
        sam_costs.append(sam_dt)
        total_costs.append(total_dt)

        rospy.loginfo(
            "[%02d/%02d] gdino=%.4fs sam=%.4fs total=%.4fs",
            i + 1,
            test_iters,
            gdino_dt,
            sam_dt,
            total_dt,
        )

        if sleep_s > 0.0:
            rospy.sleep(sleep_s)

    rospy.loginfo("GDINO stats: %s", _stats(gdino_costs))
    rospy.loginfo("SAM stats:   %s", _stats(sam_costs))
    rospy.loginfo("TOTAL stats: %s", _stats(total_costs))


if __name__ == "__main__":
    main()
