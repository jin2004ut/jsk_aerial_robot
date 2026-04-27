#!/bin/bash

rosrun aerial_robot_base rosbag_control_data.sh ${1:-arm_kinniku} arm_kinniku/livox/lidar arm_kinniku/livox/imu arm_kinniku/Odometry arm_kinniku/Odometry_precede ${@:2}
