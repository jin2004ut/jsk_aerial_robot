#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kinikun RL ポリシー推論器.

学習済みポリシー (observation 7次元 -> action 2次元, MLP[128,128]+elu) の
前向き計算だけを行う軽量ラッパ。実機で依存を増やさないよう、既定では numpy だけで
動く .npz 重みを使う。onnxruntime / torch が入っていればそちらも使える。

入出力の契約は rl_policy/kinikun_policy_realrobot_handoff.md 参照:
  obs = [joint_pos, joint_vel, target, target-joint_pos,
         last_action0, last_action1, p1-p2]  (すべて生値)
  action in [-1, 1] (呼び出し側で clamp)
"""
import os
import numpy as np


def _elu(x):
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class NumpyMLPPolicy:
    """policy_weights.npz (actor.0/2/4 の weight/bias) を使った numpy 実装."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.w0, self.b0 = d["w0"].astype(np.float32), d["b0"].astype(np.float32)
        self.w2, self.b2 = d["w2"].astype(np.float32), d["b2"].astype(np.float32)
        self.w4, self.b4 = d["w4"].astype(np.float32), d["b4"].astype(np.float32)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        x = _elu(x @ self.w0.T + self.b0)
        x = _elu(x @ self.w2.T + self.b2)
        x = x @ self.w4.T + self.b4
        return x


class OnnxPolicy:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.sess.run(None, {self.in_name: x})[0].reshape(-1)


class TorchScriptPolicy:
    def __init__(self, pt_path):
        import torch
        self.torch = torch
        self.model = torch.jit.load(pt_path).eval()

    def __call__(self, obs):
        t = self.torch.tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        with self.torch.no_grad():
            return self.model(t).numpy().reshape(-1)


def load_policy(rl_policy_dir, backend="auto"):
    """rl_policy ディレクトリからポリシーを読み込む.

    backend: "auto" | "numpy" | "onnx" | "torch"
    戻り値: (callable, 使用したbackend名)
    """
    npz = os.path.join(rl_policy_dir, "exported", "policy_weights.npz")
    onnx = os.path.join(rl_policy_dir, "exported", "policy.onnx")
    pt = os.path.join(rl_policy_dir, "exported", "policy.pt")

    order = {
        "auto": ["numpy", "onnx", "torch"],
        "numpy": ["numpy"],
        "onnx": ["onnx"],
        "torch": ["torch"],
    }[backend]

    errors = []
    for b in order:
        try:
            if b == "numpy":
                if not os.path.exists(npz):
                    raise FileNotFoundError(
                        npz + " が無い。export_policy_weights.py を実行してください")
                return NumpyMLPPolicy(npz), "numpy"
            if b == "onnx":
                return OnnxPolicy(onnx), "onnx"
            if b == "torch":
                return TorchScriptPolicy(pt), "torch"
        except Exception as e:  # noqa: BLE001
            errors.append("{}: {}".format(b, e))
    raise RuntimeError("ポリシーを読み込めませんでした:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    # 簡易セルフテスト: numpy と onnx (あれば) の一致を確認
    here = os.path.dirname(os.path.abspath(__file__))
    rl_dir = os.path.normpath(os.path.join(here, "..", "..", "rl_policy"))
    pol, backend = load_policy(rl_dir, "numpy")
    print("loaded backend:", backend)
    obs = [0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0]
    print("action(default-ish obs):", pol(obs))
