#include <sstream>
#include <string>

#include <ros/ros.h>
#include <std_msgs/String.h>

#if defined(__has_include)
#if __has_include(<jsoncpp/json/json.h>)
#include <jsoncpp/json/json.h>
#elif __has_include(<json/json.h>)
#include <json/json.h>
#else
#error "jsoncpp header not found. Please install libjsoncpp-dev"
#endif
#else
#include <jsoncpp/json/json.h>
#endif

std::string PointOrNull(const Json::Value& v) {
  if (v.isNull()) {
    return "null";
  }
  if (!v.isArray() || v.size() < 2 || !v[0].isNumeric() || !v[1].isNumeric()) {
    return "invalid";
  }

  std::ostringstream oss;
  oss << "[" << v[0].asDouble() << ", " << v[1].asDouble() << "]";
  return oss.str();
}

void Callback(const std_msgs::String::ConstPtr& msg) {
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;

  Json::Value root;
  std::string err;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  const bool ok = reader->parse(
      msg->data.data(), msg->data.data() + msg->data.size(), &root, &err);

  if (!ok || !root.isObject()) {
    ROS_WARN_STREAM("JSON parse failed: " << err << " | raw=" << msg->data);
    return;
  }

  const double stamp = root.get("stamp", 0.0).asDouble();
  const std::string frame_id = root.get("frame_id", "").asString();
  const std::string caption = root.get("caption", "").asString();
  const double gdino_score = root.get("gdino_score", 0.0).asDouble();
  const int num_points = root.get("num_points", 0).asInt();

  std::ostringstream bbox_ss;
  const Json::Value bbox = root["bbox_xyxy"];
  if (bbox.isArray() && bbox.size() == 4) {
    bbox_ss << "[" << bbox[0].asDouble() << ", " << bbox[1].asDouble() << ", "
            << bbox[2].asDouble() << ", " << bbox[3].asDouble() << "]";
  } else {
    bbox_ss << "invalid";
  }

  std::string nearest_dist = "null";
  const Json::Value& nd = root["nearest_surface_dist_m"];
  if (nd.isNumeric()) {
    nearest_dist = std::to_string(nd.asDouble());
  } else if (!nd.isNull()) {
    nearest_dist = "invalid";
  }

  ROS_INFO_STREAM("stamp=" << stamp
                  << " frame_id=" << frame_id
                  << " caption=\"" << caption << "\""
                  << " gdino_score=" << gdino_score
                  << " num_points=" << num_points
                  << " bbox=" << bbox_ss.str()
                  << " centroid_xy_m=" << PointOrNull(root["centroid_xy_m"])
                  << " nearest_surface_xy_m=" << PointOrNull(root["nearest_surface_xy_m"])
                  << " nearest_surface_dist_m=" << nearest_dist);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_depth_json_unpacker");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string topic = "/fusion_lidar_camera/object_depth_json";
  pnh.param<std::string>("topic", topic, topic);

  ros::Subscriber sub = nh.subscribe(topic, 20, Callback);
  ROS_INFO_STREAM("Subscribed: " << topic);
  ros::spin();
  return 0;
}
