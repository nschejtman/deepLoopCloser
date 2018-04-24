#include <vector>
#include <cv_bridge/cv_bridge.h>
#include "ros/ros.h"
#include <std_msgs/Bool.h>
#include <cvPerceptualHash.h>

ros::Publisher loop_closure_publisher;
std::vector<float> hashes;


void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    // Format & compute the hash
    const cv_bridge::CvImagePtr &cv_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    float hash = cvPerceptualHash::hash(cv_img->image);

    // Compare with previous hashes
    std::vector<float>::iterator it;
    for (it = hashes.begin(); it != hashes.end(); it++) {
        float similarity = cvPerceptualHash::compareHashes(hash, *it);
        if(similarity > 99.0f){
            ROS_INFO("Detected a loop");
            std_msgs::Bool pmessage;
            pmessage.data = static_cast<unsigned char>(true);
            loop_closure_publisher.publish(pmessage);
        } else {
            ROS_INFO("No loop...");
        }
    }

    // Store the vector
    hashes.push_back(hash);
}


int main(int argc, char **argv) {
    // Set up the ROS node
    ros::init(argc, argv, "loop_closer");
    ros::NodeHandle n;
    ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/images", 100, imageCallback);
    loop_closure_publisher = n.advertise<std_msgs::Bool>("/loopClosures", 100);
    ros::spin();
    return 1;
}