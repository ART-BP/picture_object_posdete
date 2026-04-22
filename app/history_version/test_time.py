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

    image_path = rospy.get_param("~image", default=os.path.join(rootdir, "bag/test0.jpg"))
    caption = rospy.get_param("~caption", "black box")
    box_threshold = float(rospy.get_param("~box_threshold", 0.45))
    text_threshold = float(rospy.get_param("~text_threshold", 0.35))
    warmup_iters = int(rospy.get_param("~warmup_iters", 0))
    test_iters = int(rospy.get_param("~test_iters", 20))
    sleep_s = float(rospy.get_param("~sleep_s", 0.0))
    reuse_sam_image_embedding = _as_bool(rospy.get_param("~reuse_sam_image_embedding", False))
    select_model = rospy.get_param("~select_model", "gdino").strip().lower()

    print(select_model)
    t0 = time.perf_counter()
    if select_model.startswith("yoloe"):
        if select_model.endswith("v8m"):
            model = Yoloe("v8m")
        elif select_model.endswith("v8l"):
            model = Yoloe("v8l")
        elif select_model.endswith("v8s"):
            model = Yoloe("v8s")
        elif select_model.endswith("11l"): 
            model = Yoloe("11l")
        elif select_model.endswith("11m"):
            model = Yoloe("11m")
        else:
            model = Yoloe("11s")
    else:
        model = gdino.GroundingDINO()
        model.setparameters(caption=caption, box_threshold=box_threshold, text_threshold=text_threshold)
    
    rospy.loginfo("select model is %s", model.name)
    sam_model = Sam()
    t1 = time.perf_counter()
    rospy.loginfo("construct model cost: %.4fs", t1 - t0)

    image = model.read_image(image_path)

    rospy.loginfo(
        "benchmark config: warmup=%d, test_iters=%d, caption='%s', reuse_sam_image_embedding=%s",
        warmup_iters,
        test_iters,
        caption,
        str(reuse_sam_image_embedding),
    )

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

        detection_dt = t_gdino - t_start
        sam_dt = t_end - t_gdino
        total_dt = t_end - t_start
        gdino_costs.append(detection_dt)
        sam_costs.append(sam_dt)
        total_costs.append(total_dt)

        rospy.loginfo(
            "[%02d/%02d] detection_dt=%.4fs sam=%.4fs total=%.4fs",
            i + 1,
            test_iters,
            detection_dt,
            sam_dt,
            total_dt,
        )

        if sleep_s > 0.0:
            rospy.sleep(sleep_s)

    rospy.loginfo("detection stats: %s", _stats(gdino_costs))
    rospy.loginfo("SAM stats:   %s", _stats(sam_costs))
    rospy.loginfo("TOTAL stats: %s", _stats(total_costs))


if __name__ == "__main__":
    main()
