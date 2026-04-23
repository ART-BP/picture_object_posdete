from camdepthfusion import points_project
from camdepthfusion import cloudpoints_handle
from camdepthfusion import camera_handle
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

def callback(image_msg, lidar_msg, pub_):
    xyzuv_lidar = cloudpoints_handle._read_xyzuv(lidar_msg)
    uv = xyzuv_lidar[:, 3:5]

    uv = uv[:,::-1]  # Convert (u,v) to (v,u) for visualization
    image = camera_handle._ros_image_to_cv2_fallback(image_msg)
    h, w = image.shape[:2]
    
    depth = np.linalg.norm(xyzuv_lidar[:, :3], axis=1)
    project_image = points_project.draw_overlay(image, uv, depth)
    
    pub_.publish(camera_handle._cv2_to_ros_image_fallback(project_image, image_msg.header))

if __name__ == "__main__":
    rospy.init_node("lidar_image_test_node")

    subcam_ = Subscriber("/camera/go2/front/image_raw", Image)
    sublidar_ = Subscriber("/visual_points/cam_front_lidar", PointCloud2)
    pub_ = rospy.Publisher("/projected_image", Image, queue_size=10)
    sync = ApproximateTimeSynchronizer(
        [subcam_, sublidar_], queue_size=10, slop=0.5
    )
    sync.registerCallback(callback, pub_)
    rospy.spin()