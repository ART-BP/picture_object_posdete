from GroundingDINO import gdino
import rospy
import os
import cv2
rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rospy.init_node("test_gdino_node")
    image_path = rospy.get_param("~image", default=os.path.join(rootdir, "bag/test.jpg"))
    time_start = rospy.Time.now().to_sec()
    model = gdino.GroundingDINO()
    time_addmodel = rospy.Time.now().to_sec()
    rospy.loginfo("time_start: %f, time_addmodel: %f", time_start, time_addmodel)
    rospy.loginfo("deltatime_addmodel: %f", time_addmodel - time_start)
    while not rospy.is_shutdown():
        time_addmodel = rospy.Time.now().to_sec()
        image = model.read_image(image_path)
        model.set_image(image=image)
        detections, phrases = model.predict(caption="black box")
        annotation = model.annotate(image=image, detections= detections, labels=phrases)
        cv2.imwrite("test.jpg", annotation)
        time_predict = rospy.Time.now().to_sec()
        rospy.loginfo("time_addmodel: %f, time_predict: %f", time_addmodel, time_predict)
        rospy.loginfo("deltatime_predict: %f", time_predict - time_addmodel)
        rospy.sleep(1)

if __name__ == "__main__":
    main()
