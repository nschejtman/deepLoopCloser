#include "ros/ros.h"

#include "std_msgs/Time.h"
#include <fstream>

using namespace std;
using namespace ros;

/**
 * This node publishes a timestamp to /timeStamp at a specified rate
 * @param argc
 * @param argv loopRate
 * @return
 */
int main(int argc, char **argv) {
    std_msgs::Time t;
    ros::Time actualTime;
    actualTime.useSystemTime();
    if (argc < 2) {
        ROS_ERROR("Missing loop rate! Correct usage is ./timeStamp loopRate");
        return -1;
    }
    int loopRate = atoi(argv[1]);

    ros::init(argc, argv, "timeStamp");
    NodeHandle n;
    Publisher ts_pub = n.advertise<std_msgs::Time>("/timeStamp", 1);
    if (ts_pub) {
        Rate loop_rate(loopRate);
        while (ts_pub.getNumSubscribers() < 1 && ok())
            ROS_INFO("Waiting for subscribers...");

        ROS_INFO("Starting to publish timestamp...");

        while (ts_pub.getNumSubscribers() > 0 && ros::ok()) {
            t.data = actualTime.now();
            ts_pub.publish(t);
            cout << "new timestamp published" << endl;
            loop_rate.sleep();
        }
        if (ts_pub.getNumSubscribers() == 0) {
            ROS_ERROR("No more subscriber! Quitting!");
            return 0;
        } else if (!ok()) {
            ROS_ERROR("Error in ROS! Quitting!");
            return -1;
        }
    } else {
        ROS_ERROR("Error in creating publisher!");
        return -1;
    }
    return 0;
}
