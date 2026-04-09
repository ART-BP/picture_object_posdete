#!/usr/bin/env python3
import json
import os 
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import torch

try:
    from GroundingDINO.gdino import GroundingDINO, root_gdino
except ImportError:
    from gdino import GroundingDINO, root_gdino


class State:
    INITIAL = 0
    READY = 1
    IMAGE_RECEIVED = 2
    PREDICTION_COMPLETE = 3
    ANNOTATION_COMPLETE = 4

class GroundingDINORos(GroundingDINO):
    def __init__(
        self, 
        model_config_path: str = None,
        model_checkpoint_path: str = None,
        device: str = None,
        topic_image: str = "/camera/go2/front/image_raw",
        caption: str = "black box",
        ):
        super().__init__(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
            caption=caption,
        )
        self.image_header = None
        self.topic_image = None
        self.sub = None
        self.state = State.INITIAL
        self.set_topic_image(topic_image)
        self.pub_ = rospy.Publisher("/grounding_dino/annotated_image", Image, queue_size=1)
        self.pub_bbox_json = rospy.Publisher("/grounding_dino/detections_json", String, latch=True, queue_size=2)

    def set_topic_image(self, topic_image: str):
        """ 

            Subscribe to the specified ROS topic for images.
            If already subscribed to a different topic, it will unsubscribe first.
        
            Args:        
            topic_image: ROS topic name to subscribe to for images.
        
        """
        topic_image = (topic_image or "").strip()
        if not topic_image:
            rospy.logwarn("Ignoring empty ~topic_image.")
            return
        if self.topic_image == topic_image and self.sub is not None:
            return

        if self.sub is not None:
            self.sub.unregister()

        self.sub = rospy.Subscriber(topic_image, Image, self.image_callback, queue_size=1)
        self.topic_image = topic_image
        self.image = None
        self.state = State.READY
        rospy.loginfo(f"Subscribed to image topic: {self.topic_image}")

    def update_runtime_params(self):
        """

        Update caption and topic_image from ROS parameters at runtime.
        This allows dynamic reconfiguration of the caption and image topic without restarting the node.

        """
        latest_caption = rospy.get_param("~caption", self.caption)
        if latest_caption != self.caption:
            self.caption = latest_caption
            rospy.loginfo(f"Updated caption to: {self.caption}")

        latest_topic_image = rospy.get_param("~topic_image", self.topic_image)
        if latest_topic_image != self.topic_image:
            self.set_topic_image(latest_topic_image)

    def shutdown(self):
        if self.sub is not None:
            self.sub.unregister()
            self.sub = None
        rospy.loginfo("GroundingDINO node shutdown complete.")

    def image_callback(self, msg):
        """

        Callback function for ROS Image messages. 
        Converts the incoming ROS Image to OpenCV format and updates the state.
        Args:
            msg: sensor_msgs/Image message received from the subscribed topic.

        """
        if self.state != State.READY:
            return
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.ros_image_to_cv2(msg)
            self.image = cv_image
            self.image_header = msg.header
            self.state = State.IMAGE_RECEIVED
            rospy.loginfo("Image received and converted to OpenCV format.")
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

    def build_bbox_payload(self, detections, labels):
        """Build JSON payload with pixel/normalized bbox info for sensor alignment."""
        if self.image is None:
            raise ValueError("No current image available for bbox payload.")

        stamp = rospy.get_time()
        frame_id = ""
        if self.image_header is not None:
            frame_id = self.image_header.frame_id
            stamp = self.image_header.stamp.to_sec()

        return super().build_bbox_payload(
            detections=detections,
            labels=labels,
            image=self.image,
            caption=self.caption,
            stamp=stamp,
            frame_id=frame_id,
        )

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

if __name__ == "__main__":
    rospy.init_node("grounding_dino_node")
    model_config_path = rospy.get_param("~model_config_path", os.path.join(root_gdino, "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    model_checkpoint_path = rospy.get_param("~model_checkpoint_path", os.path.join(root_gdino, "weights/groundingdino_swint_ogc.pth"))
    #topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_small")
    topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
    caption = rospy.get_param("~caption", "black box")
    box_threshold = float(rospy.get_param("~box_threshold", 0.40))
    text_threshold = float(rospy.get_param("~text_threshold", 0.25))

    model = GroundingDINORos(
        model_config_path=model_config_path,
        model_checkpoint_path=model_checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        topic_image=topic_image,
        caption=caption,
    )
    rospy.on_shutdown(model.shutdown)

    model.state = State.READY
    rospy.loginfo("GroundingDINO node is ready and waiting for images...")
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            model.update_runtime_params()
            if model.state == State.IMAGE_RECEIVED:
                try:
                    now = rospy.get_time()
                    rospy.loginfo(f"Processing image at time: {now:.4f} seconds")

                    # Perform prediction and annotation
                    detections, labels = model.predict(
                        image=model.image,
                        caption=model.caption,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )
                    annotated_image = model.annotate(model.image, detections, labels)
                    model.state = State.ANNOTATION_COMPLETE

                    # save picture to local
                    name_p = "demo_output/result" + str(now) + "jpg"
                    os.makedirs("demo_output", exist_ok=True)
                    cv2.imwrite(name_p, annotated_image)
                
                    # publish annotated image as ROS Image message
                    image_message = model.cv2_to_ros_image(annotated_image)
                    model.pub_.publish(image_message)

                    # publish bbox boundaries for lidar/camera alignment
                    bbox_payload = model.build_bbox_payload(detections, labels)
                    model.pub_bbox_json.publish(String(data=json.dumps(bbox_payload, ensure_ascii=False)))

                    model.state = State.READY  # Reset state to wait for the next image
                except Exception as e:
                    rospy.logerr(f"Error during prediction or annotation: {e}")
                    model.state = State.READY  # Reset state to wait for the next image
            rate.sleep()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        rospy.loginfo("Interrupt received, exiting node loop.")


GroundingDINO = GroundingDINORos
