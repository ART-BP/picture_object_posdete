from camdepthfusion import camera_handle
import rospy
from sensor_msgs.msg import Image
import cv2
i = 0
def callback(msg):
    global i
    if i > 5:
        return
    image = camera_handle._ros_image_to_cv2_fallback(msg)
    cv2.imwrite(f"test{i}.jpg", image)
    i = i+1

def main():
    rospy.init_node("save_image")
    rospy.Subscriber("/camera/go2/front/image_raw", Image, callback)
    rospy.spin()

if __name__ == "__main__":
    main()