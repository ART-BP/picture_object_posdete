import os

import cv2
import rospy
from sensor_msgs.msg import Image

try:
    from camdepthfusion import camera_handle
except ImportError:
    import camera_handle


save_requested = False
image_index = 0
save_dir = ""


def callback(msg):
    global save_requested, image_index
    if not save_requested:
        return

    try:
        image = camera_handle._ros_image_to_cv2_fallback(msg)
        out_path = os.path.join(save_dir, "test%d.jpg" % image_index)
        ok = cv2.imwrite(out_path, image)
        if ok:
            rospy.loginfo("Saved image: %s", out_path)
            image_index += 1
        else:
            rospy.logerr("Failed to save image: %s", out_path)
    except Exception as exc:
        rospy.logerr("Image save failed: %s", str(exc))
    finally:
        # one-shot save per user confirmation
        save_requested = False


def main():
    global save_requested, save_dir
    rospy.init_node("save_image")

    topic_image = rospy.get_param("~topic_image", "/camera/go2/front/image_raw")
    save_dir = rospy.get_param("~save_dir", os.path.join(os.getcwd(), "saved_images"))
    os.makedirs(save_dir, exist_ok=True)

    rospy.Subscriber(topic_image, Image, callback, queue_size=1)
    rospy.loginfo("Subscribed image topic: %s", topic_image)
    rospy.loginfo("Save directory: %s", save_dir)
    rospy.loginfo("Type 'y' to save next frame, 'q' to quit.")

    while not rospy.is_shutdown():
        try:
            cmd = input("Save next frame? [y/n/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            rospy.signal_shutdown("User interrupted")
            break

        if cmd == "y":
            save_requested = True
        elif cmd == "q":
            rospy.signal_shutdown("User requested quit")
            break

    rospy.loginfo("Shutting down save_image node.")


if __name__ == "__main__":
    main()
