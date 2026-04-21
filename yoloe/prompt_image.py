from yoloe import Yoloe
from camdepthfusion import camera_handle
import rospy
from sensor_msgs.msg import Image
from pathlib import Path

class image_prompt:
    def __init__(self):
        self.model = Yoloe("11l")
        self.sub_ = None
        self.pub_ = rospy.Publisher("/image_prompt/image", Image)
        self.prompt_image :Image = None
        self.xyxy = None

    def callback(self, msg: Image):
        if self.prompt_image == None or self.xyxy == None:
            rospy.logwarn("image prompt is no ready")
            return
        
        image = camera_handle._ros_image_to_cv2_fallback(msg)
        detetions, labels = self.model.predict_with_visual_prompt(image, self.prompt_image, self.xyxy)

        detetion_image = self.model.annotate(image, detetions, labels)
        
        image = camera_handle._cv2_to_ros_image_fallback(detetion_image, msg.header)
        self.pub_.publish(image)

    def prepare_detetion(self, path: Path, caption: str):
        self.prompt_image = self.model.read_image(path)
        detections, _ = self.model.predict_with_text(self.prompt_image, caption)

        self.xyxy = detections.xyxy

        if self.prompt_image == None or self.xyxy == None:
            rospy.logwarn("image prompt is no ready")
            return
        
        self.sub_ = rospy.Subscriber("/camera/go2/front/image_raw", Image, self.callback)
