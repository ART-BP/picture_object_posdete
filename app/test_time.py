from MobileSAM.sam import Sam

from GroundingDINO import gdino
import rospy
import os

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    rospy.init_node("test_gdino_node")
    image_path = rospy.get_param("~image", default=os.path.join(rootdir, "bag/test.jpg"))
    
    var = "_" + str(os.environ.get("GROUNDINGDINO_USE_AMP", "1"))
    var = var + "_" + str(os.environ.get("GROUNDINGDINO_RESIZE_SHORT", "800"))
    var = var + "_" + str(os.environ.get("GROUNDINGDINO_RESIZE_MAX", "1333"))
    var = var + "_" + str(os.environ.get("MOBILESAM_USE_AMP", "1")) + "_"


    time_start = rospy.Time.now().to_sec()
    model = gdino.GroundingDINO()
    sam_model = Sam()
    time_constructmodel = rospy.Time.now().to_sec()
    rospy.loginfo("time_start: %f, time_constructmodel: %f", time_start, time_constructmodel)
    rospy.loginfo("deltatime_constructmodel: %f", time_constructmodel - time_start)   
    rospy.sleep(1) 
    n = 0
    with open("time_data.txt", "a") as f:
        f.write(f"{var}\n")
    while not rospy.is_shutdown():
        n += 1
        if n > 20:
            break
        time_constructmodel = rospy.Time.now().to_sec()
        image = model.read_image(image_path)
        model.set_image(image=image)
        detections, phrases = model.predict(caption="black box")
        annotation = model.annotate(image=image, detections= detections, labels=phrases)
        # cv2.imwrite("test.jpg", annotation)
        best_idx = detections.confidence.argmax()
        xyxy = sam_model._ensure_xyxy(detections.xyxy[best_idx])
        masks = sam_model.get_mask_by_box(xyxy, image=image)
        debug = annotation
        mask = masks[0]
        debug[~mask] = 0
        # cv2.imwrite("debug.jpg", debug)
        time_detection = rospy.Time.now().to_sec()
        rospy.loginfo("time_constructmodel: %f, time_detection: %f", time_constructmodel, time_detection)
        rospy.loginfo("deltatime_detection: %f", time_detection - time_constructmodel)

        with open("time_data.txt", "a") as f:
            f.write(f"deltatime_detection: {time_detection - time_constructmodel}\n")

if __name__ == "__main__":
    main()
