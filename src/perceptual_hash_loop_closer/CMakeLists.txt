cmake_minimum_required(VERSION 2.8.3)
project(perceptual_hash_loop_closer)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
)

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
include_directories("include/perceptual_hash_loop_closer/cvPerceptualHash")
include_directories( ${OpenCV_INCLUDE_DIRS} )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES perceptual_hash_loop_closer
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


add_executable(loop_closer_node src/loop_closer_node.cpp include/perceptual_hash_loop_closer/cvPerceptualHash/cvPerceptualHash.h include/perceptual_hash_loop_closer/cvPerceptualHash/cvPerceptualHash.cpp)

add_executable(camera_simulator_node src/camera_simulator_node.cpp)

add_executable(time_stamp_node src/time_stamp_node.cpp)

target_link_libraries(loop_closer_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(camera_simulator_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(time_stamp_node ${catkin_LIBRARIES} )