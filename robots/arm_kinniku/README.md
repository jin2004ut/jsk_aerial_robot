# Arm Kinniku

# How to use

In one terminal, run

`roslaunch arm_kinniku bringup.launch real_machine:=false simulation:=True headless:=False`

In another terminal, run

`rosrun aerial_robot_base keyboard_command.py`

In this terminal, input `r` to arm the arm_kinniku, then input `t` to takeoff. The arm_kinniku will takeoff and hover.

Input `l` to land the arm_kinniku.
