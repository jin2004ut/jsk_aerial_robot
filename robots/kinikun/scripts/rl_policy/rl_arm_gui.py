#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kinikun RL アーム用の簡易 GUI.

arm1_joint の目標角[deg]をスライダ/入力で指定して /rl/target_deg に publish する。
現在角 (/rl/theta_deg) を表示し、Start/Stop で /rl/enable を切り替える。
制御本体は rl_arm_controller.py。
"""
import math
import tkinter as tk

import rospy
from std_msgs.msg import Bool, Float32


class RLArmGUI:
    def __init__(self):
        rospy.init_node("rl_arm_gui", anonymous=True)

        # 安全な目標範囲 (controller と同じ既定: center ± target_scale*half)
        lower = float(rospy.get_param("~arm_lower_rad", -math.pi / 4))
        upper = float(rospy.get_param("~arm_upper_rad", math.pi / 4))
        scale = float(rospy.get_param("~target_scale", 0.7))
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        self.min_deg = math.degrees(center - scale * half)
        self.max_deg = math.degrees(center + scale * half)

        self.pub_target = rospy.Publisher("/rl/target_deg", Float32, queue_size=1)
        self.pub_enable = rospy.Publisher("/rl/enable", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/rl/theta_deg", Float32, self._cb_theta, queue_size=1)

        self.current_deg = None
        self.enabled = False
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("kinikun RL arm")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        pad = {"padx": 8, "pady": 4}

        tk.Label(self.root, text="目標角 arm1_joint [deg]",
                 font=("", 12, "bold")).grid(row=0, column=0, columnspan=2, **pad)

        self.var_target = tk.DoubleVar(value=0.0)
        self.scale = tk.Scale(
            self.root, from_=round(self.min_deg, 1), to=round(self.max_deg, 1),
            resolution=0.5, orient=tk.HORIZONTAL, length=340,
            variable=self.var_target, command=self._on_slider)
        self.scale.grid(row=1, column=0, columnspan=2, **pad)

        tk.Label(self.root, text="直接入力:").grid(row=2, column=0, sticky="e", **pad)
        self.entry = tk.Entry(self.root, width=8)
        self.entry.insert(0, "0.0")
        self.entry.grid(row=2, column=1, sticky="w", **pad)
        self.entry.bind("<Return>", self._on_entry)

        tk.Label(self.root,
                 text="範囲 [{:.1f}, {:.1f}] deg".format(self.min_deg, self.max_deg),
                 fg="gray").grid(row=3, column=0, columnspan=2, **pad)

        self.lbl_current = tk.Label(self.root, text="現在角: -- deg",
                                    font=("", 12))
        self.lbl_current.grid(row=4, column=0, columnspan=2, **pad)

        self.btn_enable = tk.Button(self.root, text="Start", width=10,
                                    bg="#3a7", fg="white", command=self._toggle_enable)
        self.btn_enable.grid(row=5, column=0, **pad)

        tk.Button(self.root, text="STOP", width=10, bg="#c33", fg="white",
                  command=self._stop).grid(row=5, column=1, **pad)

        tk.Button(self.root, text="0 にリセット", command=self._reset_zero).grid(
            row=6, column=0, columnspan=2, **pad)

        # 初回 target を publish
        self._publish_target(0.0)
        self.root.after(100, self._tick)

    # ---------------- events ----------------
    def _clamp(self, v):
        return max(self.min_deg, min(self.max_deg, v))

    def _publish_target(self, deg):
        deg = self._clamp(float(deg))
        self.pub_target.publish(Float32(deg))
        return deg

    def _on_slider(self, _val):
        deg = self._publish_target(self.var_target.get())
        self.entry.delete(0, tk.END)
        self.entry.insert(0, "{:.1f}".format(deg))

    def _on_entry(self, _evt):
        try:
            deg = float(self.entry.get())
        except ValueError:
            return
        deg = self._publish_target(deg)
        self.var_target.set(deg)

    def _reset_zero(self):
        self.var_target.set(0.0)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, "0.0")
        self._publish_target(0.0)

    def _toggle_enable(self):
        self._set_enable(not self.enabled)

    def _stop(self):
        self._set_enable(False)

    def _set_enable(self, val):
        self.enabled = bool(val)
        self.pub_enable.publish(Bool(self.enabled))
        if self.enabled:
            self.btn_enable.config(text="Running", bg="#888")
        else:
            self.btn_enable.config(text="Start", bg="#3a7")

    def _cb_theta(self, msg):
        self.current_deg = float(msg.data)

    def _tick(self):
        if rospy.is_shutdown():
            self.root.destroy()
            return
        if self.current_deg is not None:
            self.lbl_current.config(text="現在角: {:+.1f} deg".format(self.current_deg))
        self.root.after(100, self._tick)

    def _on_close(self):
        self._set_enable(False)  # 閉じる時は必ず停止
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        RLArmGUI().run()
    except rospy.ROSInterruptException:
        pass
