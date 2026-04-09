import rospy
import os
import numpy as np
import cv2

from sensor_msgs.msg import Image
from GroundingDINO.gdino import GroundingDINO
from MobileSAM.sam import Sam

class ObjectDetection:
    def __init__(self):
        self.gdino_model = GroundingDINO()
        self.sam_model = Sam()
        self.sub = None
        self.image = None
        self.box_threshold = 0.45
        self.text_threshold =0.35
        self.image_header = None
        self.caption = "black box"

    def image_callback(self, msg):
        """

        Callback function for ROS Image messages. 
        Converts the incoming ROS Image to OpenCV format and updates the state.
        Args:
            msg: sensor_msgs/Image message received from the subscribed topic.

        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.ros_image_to_cv2(msg)
            self.image = cv_image
            self.image_header = msg.header
            rospy.loginfo("Image received and converted to OpenCV format.")
            if self.sub is not None:
                self.sub.unregister()
                self.sub = None
        except Exception as e:
            rospy.logerr(f"Failed to convert ROS Image to OpenCV: {e}")

    def ros_image_to_cv2(self, ros_image):
        """Convert sensor_msgs/Image to OpenCV BGR image."""
        try:
            data = np.frombuffer(ros_image.data, dtype=np.uint8)
            h, w = ros_image.height, ros_image.width
            enc = (ros_image.encoding or "").lower()

            # Prefer raw-image decoding for sensor_msgs/Image.
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

            # Fallback for compressed payloads accidentally published as Image.
            decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(
                    f"Unsupported/invalid image data (encoding={ros_image.encoding}, "
                    f"height={h}, width={w}, bytes={len(ros_image.data)})"
                )
            return decoded
        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to OpenCV: {e}")
            raise


    def cv2_to_ros_image(self, annotated_image):
        """

        Convert OpenCV BGR image to sensor_msgs/Image.
        args:
            annotated_image: The annotated image in OpenCV BGR format to convert to ROS Image message.
        returns:
            A sensor_msgs/Image message containing the annotated image data.

        """
        try:
            h, w, c = annotated_image.shape
            if c != 3:
                raise ValueError(f"Expected 3-channel BGR image, got {c} channels.")

            ros_image = Image()
            ros_image.header.stamp = rospy.Time.now()
            ros_image.height = h
            ros_image.width = w
            ros_image.encoding = "bgr8"
            ros_image.is_bigendian = 0
            ros_image.step = w * c
            ros_image.data = annotated_image.tobytes()
            return ros_image
        except Exception as e:
            rospy.logerr(f"Error converting OpenCV image to ROS Image: {e}")
            raise
    
    def mask_detection(self, caption):
        detections, labels = self.gdino_model.predict(
            image=self.image,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(detections.xyxy) == 0:
            rospy.logerr(f"gdino 检测错误")
            return
        
        max_score = int(np.argmax(detections.confidence))
        box_xyxy = detections.xyxy[max_score]

        mask, score, idx = self.sam_model.get_mask_by_box(
            box_xyxy=box_xyxy,
            image=self.image,
            image_format="BGR",
            multimask_output=False,
        )
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)

        now = self.image_header.stamp
        cv2.imwrite(os.path.join(output_dir, f"{str(now)}orin.jpg"), self.image)
        cv2.imwrite(os.path.join(output_dir, f"{str(now)}annotate.jpg"), self.gdino_model.annotate(self.image, detections, labels))
        self.sam_model.save_mask(mask, os.path.join(output_dir, f"{str(now)}mask.jpg"))
        obj_rgb = self.sam_model.render_object(self.sam_model.image_rgb, mask)
        self.sam_model.save_rgb(obj_rgb, os.path.join(output_dir, f"{str(now)}rgb.jpg"))
        crop_rgb = self.sam_model.crop_by_mask(self.sam_model.image_rgb, mask)
        self.sam_model.save_rgb(crop_rgb, os.path.join(output_dir, f"{str(now)}crop.jpg"))

    def start(self, caption, timeout=3.0):
        self.image = None
        self.image_header = None

        if self.sub is not None:
            self.sub.unregister()
            self.sub = None

        self.sub = rospy.Subscriber("/camera/go2/front/image_raw", Image, self.image_callback, queue_size=1)

        start_time = rospy.Time.now().to_sec()
        while self.image is None and not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - start_time > timeout:
                if self.sub is not None:
                    self.sub.unregister()
                    self.sub = None
                rospy.logwarn("等待图像超时")
                return
            rospy.sleep(0.01)

        self.mask_detection(caption)


    def update_runtime_params(self):
        self.caption = rospy.get_param("~caption", self.caption)
        run_once = rospy.get_param("~run_once", False)

        if run_once:
            rospy.set_param("~run_once", False)
            rospy.loginfo(f"Run once with caption={self.caption}")
            self.start(self.caption)


def main():
    rospy.init_node("get_objectmask_node")
    odetection = ObjectDetection()
    odetection.caption = rospy.get_param("~caption", "black box")
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        odetection.update_runtime_params()
        rate.sleep()

if __name__ == "__main__":
    main()