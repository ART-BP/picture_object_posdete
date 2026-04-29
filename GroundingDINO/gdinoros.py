#!/usr/bin/env python3
import json
import os
import sys
import threading

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from GroundingDINO.gdino import GroundingDINO, root_gdino
except ImportError:
    from gdino import GroundingDINO, root_gdino

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from camdepthfusion.camera_handle import DEFAULT_CAMERA_PARAM_YAML, load_camera_params_from_yaml


class GroundingDINORos(GroundingDINO):
    def __init__(
        self,
        model_config_path: str = None,
        model_checkpoint_path: str = None,
        device: str = None,
        topic_image: str = "/camera/go2/front/image_raw",
        topic_caption: str = "/grounding_dino/caption",
        caption: str = "black box",
        box_threshold: float = 0.45,
        text_threshold: float = 0.45,
        camera_params_yaml: str = DEFAULT_CAMERA_PARAM_YAML,
        camera_model: str = "fisheye",
    ):
        super().__init__(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
            caption=caption,
        )
        self.setparameters(caption=caption, box_threshold=box_threshold, text_threshold=text_threshold)

        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_cv = threading.Condition(self._frame_lock)
        self._latest_image_msg = None
        self._running = True
        self._topic_image = topic_image
        self._topic_caption = topic_caption
        self._camera_params_yaml = camera_params_yaml
        self._camera_model = camera_model

        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_size = None
        self._load_undistort_params()

        self.pub_image = rospy.Publisher("/grounding_dino/annotated_image", Image, queue_size=1, latch=True)
        self.pub_bbox_json = rospy.Publisher("/grounding_dino/detections_json", String, queue_size=2)

        self.sub_image = rospy.Subscriber(
            self._topic_image,
            Image,
            self.image_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_caption = rospy.Subscriber(self._topic_caption, String, self.caption_callback, queue_size=1)
        self._worker = threading.Thread(target=self._infer_loop, name="gdino_infer_loop", daemon=True)
        self._worker.start()

        rospy.loginfo("GroundingDINO subscribed image topic: %s", self._topic_image)
        rospy.loginfo("GroundingDINO subscribed caption topic: %s", self._topic_caption)
        rospy.loginfo("GroundingDINO initial caption: %s", self.caption)

    def _load_undistort_params(self):
        params = load_camera_params_from_yaml(
            yaml_path=self._camera_params_yaml,
            camera_model=self._camera_model,
        )
        self._cam_distortion_model = str(params.get("distortion_model", self._camera_model)).lower()
        self._cam_K = np.asarray(params["K"], dtype=np.float64).reshape(3, 3)
        self._cam_D = np.asarray(params["D"], dtype=np.float64).reshape(-1)
        self._cam_R = np.asarray(params.get("R_rect", np.eye(3)), dtype=np.float64).reshape(3, 3)
        P = params.get("P")
        if P is not None:
            P_arr = np.asarray(P, dtype=np.float64).reshape(3, 4)
            self._cam_newK = P_arr[:, :3]
        else:
            self._cam_newK = self._cam_K.copy()

        rospy.loginfo(
            "GroundingDINO undistort params loaded: yaml=%s model=%s distortion_model=%s",
            self._camera_params_yaml,
            self._camera_model,
            self._cam_distortion_model,
        )

    def _ensure_undistort_maps(self, width: int, height: int):
        size = (int(width), int(height))
        if self._undistort_size == size and self._undistort_map1 is not None and self._undistort_map2 is not None:
            return

        if "fisheye" in self._cam_distortion_model:
            if self._cam_D.size < 4:
                raise ValueError(
                    f"Fisheye distortion coefficients need >=4 values, got {self._cam_D.size}"
                )
            D = self._cam_D[:4].reshape(4, 1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self._cam_K,
                D,
                self._cam_R,
                self._cam_newK,
                size,
                cv2.CV_16SC2,
            )
        else:
            # Works for standard/rational_polynomial style distortion vectors.
            map1, map2 = cv2.initUndistortRectifyMap(
                self._cam_K,
                self._cam_D,
                self._cam_R,
                self._cam_newK,
                size,
                cv2.CV_16SC2,
            )

        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_size = size

    def _undistort_image(self, image: np.ndarray):
        h, w = image.shape[:2]
        self._ensure_undistort_maps(w, h)
        return cv2.remap(
            image,
            self._undistort_map1,
            self._undistort_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    @staticmethod
    def _parse_caption(data: str):
        text = (data or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("caption") is not None:
                cap = str(payload["caption"]).strip()
                return cap if cap else None
        except Exception:
            pass
        return text

    def caption_callback(self, msg: String):
        new_caption = self._parse_caption(msg.data)
        if not new_caption:
            return
        with self._lock:
            if new_caption == self.caption:
                return
            self.setparameters(caption=new_caption)
            current = self.caption
        rospy.loginfo("GroundingDINO caption updated: %s", current)

    @staticmethod
    def ros_image_to_cv2(ros_image: Image):
        data = np.frombuffer(ros_image.data, dtype=np.uint8)
        h, w = ros_image.height, ros_image.width
        enc = (ros_image.encoding or "").lower()

        if h > 0 and w > 0:
            if enc in ("bgr8", "rgb8"):
                expected = h * w * 3
                if data.size >= expected:
                    frame = data[:expected].reshape((h, w, 3))
                    if enc == "rgb8":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return frame
            elif enc in ("bgra8", "rgba8"):
                expected = h * w * 4
                if data.size >= expected:
                    frame = data[:expected].reshape((h, w, 4))
                    if enc == "rgba8":
                        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif enc in ("mono8", "8uc1"):
                expected = h * w
                if data.size >= expected:
                    gray = data[:expected].reshape((h, w))
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(
                "Unsupported image data "
                f"(encoding={ros_image.encoding}, height={h}, width={w}, bytes={len(ros_image.data)})"
            )
        return decoded

    @staticmethod
    def cv2_to_ros_image(bgr: np.ndarray, header):
        h, w, c = bgr.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel BGR image, got {c}.")
        out = Image()
        out.header = header
        out.height = h
        out.width = w
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = w * c
        out.data = bgr.tobytes()
        return out

    def image_callback(self, msg: Image):
        # Producer: callback never runs detection; it only updates the newest frame.
        with self._frame_cv:
            self._latest_image_msg = msg
            self._frame_cv.notify()

    def _infer_loop(self):
        while not rospy.is_shutdown():
            with self._frame_cv:
                while self._running and self._latest_image_msg is None and not rospy.is_shutdown():
                    self._frame_cv.wait(timeout=0.2)
                if not self._running or rospy.is_shutdown():
                    break
                # Consumer: take one latest frame. New arrivals during inference overwrite this slot.
                msg = self._latest_image_msg
                self._latest_image_msg = None
            self._process_one_image(msg)

    def _process_one_image(self, msg: Image):
        try:
            image = self.ros_image_to_cv2(msg)
            # image = self._undistort_image(image_raw)
            with self._lock:
                current_caption = self.caption

            detections, labels = self.predict(image=image, caption=current_caption)
            annotated = self.annotate(image, detections, labels)
            self.pub_image.publish(self.cv2_to_ros_image(annotated, msg.header))

            stamp = msg.header.stamp.to_sec() if msg.header is not None else rospy.get_time()
            frame_id = msg.header.frame_id if msg.header is not None else ""
            payload = self.build_bbox_payload(
                detections=detections,
                labels=labels,
                image=image,
                caption=current_caption,
                stamp=stamp,
                frame_id=frame_id,
            )
            self.pub_bbox_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception as e:
            rospy.logerr("GroundingDINO image processing failed: %s", e)

    def shutdown(self):
        with self._frame_cv:
            self._running = False
            self._frame_cv.notify_all()
        if self.sub_image is not None:
            self.sub_image.unregister()
            self.sub_image = None
        if self.sub_caption is not None:
            self.sub_caption.unregister()
            self.sub_caption = None
        if getattr(self, "_worker", None) is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        rospy.loginfo("GroundingDINO node shutdown complete.")


def main():
    rospy.init_node("grounding_dino_node")
    model_config_path = rospy.get_param(
        "~model_config_path",
        os.path.join(root_gdino, "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    model_checkpoint_path = rospy.get_param(
        "~model_checkpoint_path",
        os.path.join(root_gdino, "weights/groundingdino_swint_ogc.pth"),
    )
    topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
    topic_caption = rospy.get_param("~topic_caption", "/grounding_dino/caption")
    caption = rospy.get_param("~caption", "black box")
    box_threshold = float(rospy.get_param("~box_threshold", 0.55))
    text_threshold = float(rospy.get_param("~text_threshold", 0.65))
    camera_params_yaml = rospy.get_param("~camera_params_yaml", DEFAULT_CAMERA_PARAM_YAML)
    camera_model = "fisheye"

    model = GroundingDINORos(
        model_config_path=model_config_path,
        model_checkpoint_path=model_checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        topic_image=topic_image,
        topic_caption=topic_caption,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        camera_params_yaml=camera_params_yaml,
        camera_model=camera_model,
    )
    rospy.on_shutdown(model.shutdown)
    rospy.loginfo("GroundingDINO node ready.")
    rospy.spin()


if __name__ == "__main__":
    main()


GroundingDINO = GroundingDINORos
