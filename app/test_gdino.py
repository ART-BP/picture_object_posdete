from GroundingDINO import gdino
import rospy
import os

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    rospy.init_node("test_gdino_node")
    while not rospy.is_shutdown():
        image_path = rospy.get_param("~image", default=os.path.join(rootdir, "bag/test.jpg"))
        time_start = rospy.Time.now().to_sec()
        model = gdino.GroundingDINO()
        time_addmodel = rospy.Time.now().to_sec()
        detections, phrases = model.predict(image_path, "black box")
        time_predict = rospy.Time.now().to_sec()
        rospy.loginfo("time_start: %f, time_addmodel: %f, time_predict: %f", time_start, time_addmodel, time_predict)
        rospy.loginfo("delta_time_addmodel: %f, deltatime_predict: %f", time_addmodel - time_start, time_predict - time_addmodel)
        rospy.sleep(1)

if __name__ == "__main__":
    main()