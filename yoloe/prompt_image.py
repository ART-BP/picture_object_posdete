from yoloe import Yoloe
from camdepthfusion import camera_handle
import rospy
from sensor_msgs.msg import Image
from pathlib import Path
import os
import numpy as np
import cv2
import queue
import threading


class image_prompt:
    def __init__(self,model_id):
        self.model = Yoloe(model_id)
        self.sub_ = None
        self.pub_ = rospy.Publisher("/image_prompt/image", Image, queue_size=1)
        self.prompt_image = None
        self.xyxy = None
        self.frame_queue: "queue.Queue[Image]" = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, name="prompt_worker", daemon=True)
        self.worker_thread.start()
        rospy.on_shutdown(self._on_shutdown)

    def callback(self, msg: Image):
        if self.prompt_image is None or self.xyxy is None:
            rospy.logwarn("image prompt is no ready")
            return

        # Keep only the newest frame to avoid callback backlog.
        try:
            self.frame_queue.put_nowait(msg)
            return
        except queue.Full:
            pass

        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.frame_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _worker_loop(self):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                msg = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if msg is None:
                continue

            try:
                image = camera_handle._ros_image_to_cv2_fallback(msg)
                detetions, labels = self.model.predict_with_visual_prompt(image, self.prompt_image, self.xyxy)
                detetion_image = self.model.annotate(image, detetions, labels)
                image_msg = camera_handle._cv2_to_ros_image_fallback(detetion_image, msg.header)
                self.pub_.publish(image_msg)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "prompt worker failed: %s", str(exc))

    def _on_shutdown(self):
        self.stop_event.set()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)

    def prepare_detetion(self, path: Path, caption: str):
        self.prompt_image = self.model.read_image(path)
        detections = self.model.predict_with_text(self.prompt_image, caption)
        if len(detections.xyxy) == 0:
            rospy.logwarn("No detections for prompt image, caption=%s", caption)
            return
        box = np.asarray(detections.xyxy[0], dtype=np.float32).reshape(-1)
        if box.size != 4:
            rospy.logwarn("Invalid bbox shape from prompt image: %s", str(detections.xyxy.shape))
            return
        self.xyxy = box.tolist()

        if self.prompt_image is None or self.xyxy is None:
            rospy.logwarn("image prompt is no ready")
            return
        self.sub_ = rospy.Subscriber("/camera/go2/front/image_raw", Image, self.callback, queue_size=1)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]
        detection_image = self.model.annotate(self.prompt_image, detections, labels)
        self.prompt_image = detection_image
        cv2.imwrite("test.jpg", detection_image)

def main():
    object_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rospy.init_node("test_yoloe")
    model_id = rospy.get_param("~model", "11l")
    image_p = image_prompt(model_id)
    image_p.prepare_detetion(os.path.join(object_dir, "bag/test0.jpg"),  "person")
    rospy.spin()

if __name__ == "__main__":
    main()
