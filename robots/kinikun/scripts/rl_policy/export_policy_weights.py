#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""policy.onnx から numpy 用の重み policy_weights.npz を生成する.

実機で torch / onnxruntime を入れずに numpy だけで推論するための一度きりの変換。
生成物: rl_policy/exported/policy_weights.npz (actor.0/2/4 の weight/bias)

使い方:
    pip install onnx   # 変換時のみ必要。実行時は不要
    python3 export_policy_weights.py [rl_policy_dir]
"""
import os
import sys
import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, "..", "..", "rl_policy"))
    rl_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir

    onnx_path = os.path.join(rl_dir, "exported", "policy.onnx")
    out_path = os.path.join(rl_dir, "exported", "policy_weights.npz")

    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    w = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    np.savez(
        out_path,
        w0=w["actor.0.weight"], b0=w["actor.0.bias"],
        w2=w["actor.2.weight"], b2=w["actor.2.bias"],
        w4=w["actor.4.weight"], b4=w["actor.4.bias"],
    )
    print("wrote", out_path)

    # onnx リファレンス実装との一致を確認
    try:
        from onnx.reference import ReferenceEvaluator
        sess = ReferenceEvaluator(onnx_path)
        d = np.load(out_path)

        def elu(x):
            return np.where(x > 0, x, np.exp(x) - 1.0)

        rng = np.random.RandomState(0)
        max_diff = 0.0
        for _ in range(20):
            obs = rng.randn(1, 7).astype(np.float32)
            ref = sess.run(None, {"obs": obs})[0]
            x = elu(obs @ d["w0"].T + d["b0"])
            x = elu(x @ d["w2"].T + d["b2"])
            mine = x @ d["w4"].T + d["b4"]
            max_diff = max(max_diff, float(np.abs(ref - mine).max()))
        print("numpy vs onnx max abs diff:", max_diff)
    except Exception as e:  # noqa: BLE001
        print("(検証スキップ:", e, ")")


if __name__ == "__main__":
    main()
