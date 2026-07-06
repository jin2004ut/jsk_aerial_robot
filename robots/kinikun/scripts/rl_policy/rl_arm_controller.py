#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kinikun 空圧アーム RL ポリシー制御ノード.

学習済み RL ポリシーで arm1_joint を目標角に追従させる。
GUI (rl_arm_gui.py) が publish する目標角 (/rl/target_deg) を受け取り、
200Hz でポリシーを回して 2ch 圧力コマンド (/mpa_cmd) を出す。

契約は rl_policy/kinikun_policy_realrobot_handoff.md に準拠:
  obs = [joint_pos, joint_vel, target, target-joint_pos,
         last_action0, last_action1, p1-p2]
  a in [-1,1] -> p = 0.5*(a+1)*pressure_limit_mpa
  /mpa_cmd (Quaternion): x=p1*4096/0.9, y=p2*4096/0.9  (既存 MPPI/PID と同じ DAC 換算)
"""
import math
import os
import sys
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_infer import load_policy  # noqa: E402


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class RLArmController:
    def __init__(self):
        rospy.init_node("rl_arm_controller")

        # ---- モデル場所 ----
        default_dir = self._default_rl_dir()
        self.rl_dir = rospy.get_param("~rl_policy_dir", default_dir)
        backend = rospy.get_param("~backend", "auto")

        # ---- 契約パラメータ (handoff §2/§4) ----
        self.rate_hz = float(rospy.get_param("~rate", 200.0))  # 学習時 200Hz
        self.pressure_limit_mpa = float(rospy.get_param("~pressure_limit_mpa", 0.6))
        self.dac_per_mpa = float(rospy.get_param("~dac_per_mpa", 4096.0 / 0.9))

        # 可動域と安全な目標範囲 (center ± target_scale*half)
        self.arm_lower = float(rospy.get_param("~arm_lower_rad", -math.pi / 4))
        self.arm_upper = float(rospy.get_param("~arm_upper_rad", math.pi / 4))
        self.target_scale = float(rospy.get_param("~target_scale", 0.7))
        center = 0.5 * (self.arm_lower + self.arm_upper)
        half = 0.5 * (self.arm_upper - self.arm_lower)
        self.target_min = center - self.target_scale * half
        self.target_max = center + self.target_scale * half

        # 実機とsimの関節定義合わせ (handoff §4)
        self.theta_sign = float(rospy.get_param("~theta_sign", 1.0))
        self.theta_offset = float(rospy.get_param("~theta_offset", 0.0))

        # ---- センサ (JointState) ----
        self.theta_topic = rospy.get_param("~theta_topic", "/kinikun1/joint_states")
        self.theta_index = int(rospy.get_param("~theta_index", 2))
        self.vel_lpf_alpha = float(rospy.get_param("~vel_lpf_alpha", 0.2))

        # ---- 安全 ----
        # 圧力レート制限 (MPa/step)。dp_max_mpa_s [MPa/s] を dt で割る
        self.dp_max_mpa_s = float(rospy.get_param("~dp_max_mpa_s", 5.0))
        self.cmd_timeout = float(rospy.get_param("~sensor_timeout", 0.2))  # [s]
        self.start_enabled = bool(rospy.get_param("~start_enabled", False))

        # ---- ポリシー ----
        self.policy, used = load_policy(self.rl_dir, backend)
        rospy.loginfo("[RL] policy loaded (backend=%s) from %s", used, self.rl_dir)

        # ---- 状態 ----
        self.lock = threading.Lock()
        self.enabled = self.start_enabled
        self.have_theta = False
        self.theta = 0.0            # rad (sim系に整合済み)
        self.theta_vel = 0.0        # rad/s
        self._theta_prev = None
        self._theta_stamp = None
        self.last_sensor_time = None
        self.target = center        # rad, 既定は中心姿勢

        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.pressure = np.array([0.0, 0.0], dtype=np.float32)  # MPa (前ステップ)

        # ---- ROS I/F ----
        self.pub_cmd = rospy.Publisher("/mpa_cmd", Quaternion, queue_size=1)
        self.pub_theta_deg = rospy.Publisher("/rl/theta_deg", Float32, queue_size=1)
        rospy.Subscriber(self.theta_topic, JointState, self.cb_theta, queue_size=10)
        rospy.Subscriber("/rl/target_deg", Float32, self.cb_target, queue_size=1)
        rospy.Subscriber("/rl/enable", Bool, self.cb_enable, queue_size=1)

        rospy.loginfo("[RL] target range: [%.1f, %.1f] deg (%.3f, %.3f rad)",
                      math.degrees(self.target_min), math.degrees(self.target_max),
                      self.target_min, self.target_max)
        rospy.loginfo("[RL] rate=%.0fHz  pressure_limit=%.2fMPa  enabled=%s",
                      self.rate_hz, self.pressure_limit_mpa, self.enabled)
        rospy.on_shutdown(self._on_shutdown)

    @staticmethod
    def _default_rl_dir():
        try:
            import rospkg
            return os.path.join(rospkg.RosPack().get_path("kinikun"), "rl_policy")
        except Exception:  # noqa: BLE001
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.normpath(os.path.join(here, "..", "..", "rl_policy"))

    # ---------------- callbacks ----------------
    def cb_theta(self, msg):
        if len(msg.position) <= self.theta_index:
            return
        raw = float(msg.position[self.theta_index])
        theta = self.theta_sign * raw + self.theta_offset
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        with self.lock:
            # 位置差分で速度推定 + 一次LPF
            if self._theta_prev is not None and self._theta_stamp is not None:
                dt = now - self._theta_stamp
                if dt > 1e-4:
                    v = (theta - self._theta_prev) / dt
                    a = self.vel_lpf_alpha
                    self.theta_vel = a * v + (1.0 - a) * self.theta_vel
            self.theta = theta
            self._theta_prev = theta
            self._theta_stamp = now
            self.last_sensor_time = rospy.get_time()
            self.have_theta = True

    def cb_target(self, msg):
        t = math.radians(float(msg.data))
        with self.lock:
            self.target = clamp(t, self.target_min, self.target_max)

    def cb_enable(self, msg):
        with self.lock:
            self.enabled = bool(msg.data)
        rospy.loginfo("[RL] enable = %s", msg.data)

    # ---------------- control loop ----------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz
        dp_max = self.dp_max_mpa_s * dt
        plim = self.pressure_limit_mpa
        while not rospy.is_shutdown():
            with self.lock:
                enabled = self.enabled
                have = self.have_theta
                theta = self.theta
                theta_vel = self.theta_vel
                target = self.target
                last_action = self.last_action.copy()
                pressure = self.pressure.copy()
                last_sensor = self.last_sensor_time

            # 安全: センサ未着 / タイムアウト / 無効 なら 0 圧力
            stale = (last_sensor is None or
                     (rospy.get_time() - last_sensor) > self.cmd_timeout)
            if not enabled or not have or stale:
                if stale and enabled and have:
                    rospy.logwarn_throttle(1.0, "[RL] sensor stale -> zero pressure")
                self._publish_pressure(0.0, 0.0)
                with self.lock:
                    self.pressure[:] = 0.0
                    self.last_action[:] = 0.0
                rate.sleep()
                continue

            # 観測 (前ステップの pressure / last_action を使う: handoff §3)
            obs = np.array([
                theta,
                theta_vel,
                target,
                target - theta,
                last_action[0],
                last_action[1],
                pressure[0] - pressure[1],
            ], dtype=np.float32)

            a = np.clip(np.asarray(self.policy(obs)).reshape(-1), -1.0, 1.0)

            p_new = 0.5 * (a + 1.0) * plim  # [0, plim] MPa
            # レート制限
            p_new = np.clip(p_new, pressure - dp_max, pressure + dp_max)
            p_new = np.clip(p_new, 0.0, plim)

            self._publish_pressure(float(p_new[0]), float(p_new[1]))
            self.pub_theta_deg.publish(Float32(math.degrees(theta)))

            with self.lock:
                self.pressure[:] = p_new
                self.last_action[:] = a
            rate.sleep()

    def _publish_pressure(self, p1, p2):
        msg = Quaternion()
        msg.x = p1 * self.dac_per_mpa
        msg.y = p2 * self.dac_per_mpa
        msg.z = 0.0
        msg.w = 0.0
        self.pub_cmd.publish(msg)

    def _on_shutdown(self):
        try:
            self._publish_pressure(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        rospy.loginfo("[RL] shutdown: zero pressure sent")


if __name__ == "__main__":
    try:
        RLArmController().spin()
    except rospy.ROSInterruptException:
        pass
