# kinikun RL ポリシー実機ランナー + GUI

学習済み RL ポリシー (`rl_policy/exported/policy.*`) で空圧アーム `arm1_joint` を
目標角に追従させる。GUI で角度を指定するだけの簡素な構成。

契約の詳細は [`../../rl_policy/kinikun_policy_realrobot_handoff.md`](../../rl_policy/kinikun_policy_realrobot_handoff.md) を参照。

## 構成

| ファイル | 役割 |
|---|---|
| `rl_arm_controller.py` | 制御本体。200Hz でポリシーを回し `/mpa_cmd` を出す ROS ノード |
| `rl_arm_gui.py` | 目標角[deg]を指定する簡易 Tkinter GUI |
| `policy_infer.py` | ポリシー推論器 (numpy / onnx / torch 対応、既定は numpy) |
| `export_policy_weights.py` | `policy.onnx` → numpy 用 `policy_weights.npz` の変換 (変換時のみ onnx が必要) |
| `../../launch/modeling/rl_arm.launch` | controller + GUI をまとめて起動 |

**依存**: 実行時は `numpy` と `rospy` だけ（`policy_weights.npz` を使うため torch/onnxruntime 不要）。

## トピック I/F

- 入力 `~theta_topic` (`/kinikun1/joint_states`, `sensor_msgs/JointState`): `position[~theta_index]` を関節角に使用
- 入力 `/rl/target_deg` (`std_msgs/Float32`): 目標角 [deg]（GUI が publish）
- 入力 `/rl/enable` (`std_msgs/Bool`): 制御の有効/無効（GUI の Start/Stop）
- 出力 `/mpa_cmd` (`geometry_msgs/Quaternion`): `x=p1*4096/0.9, y=p2*4096/0.9` の DAC 値（既存 MPPI/PID と同じ換算）
- 出力 `/rl/theta_deg` (`std_msgs/Float32`): 現在角 [deg]（GUI 表示用）

## 使い方

```bash
# 実機のエンコーダ & /mpa_cmd ブリッジは別途起動しておく (bringup.launch 等)
roslaunch kinikun rl_arm.launch
```

GUI のスライダ/入力で目標角を指定し、**Start** を押すと制御開始。**STOP** または
ウィンドウを閉じると圧力ゼロで停止する。

主な引数:

```bash
roslaunch kinikun rl_arm.launch \
    theta_index:=2 \        # JointState 内の関節角インデックス
    theta_sign:=1.0 \       # 実機とsimの符号合わせ (handoff §4)
    theta_offset:=0.0 \     # 0点合わせ [rad]
    start_enabled:=false \  # true で GUI なしでも即制御
    gui:=true
```

## 安全機構

- 出力圧力を `[0, pressure_limit_mpa=0.6] MPa` にハードクランプ
- 圧力レート制限 `dp_max_mpa_s`（既定 5 MPa/s）
- 目標角を学習範囲 `center ± target_scale*half`（既定 ≈ ±31.5°）にクランプ
- エンコーダが `sensor_timeout`（既定 0.2s）途切れると自動でゼロ圧力
- 既定は無効状態で起動（GUI の Start 待ち）、ノード終了時にゼロ圧力送信

## npz の再生成（ポリシー差し替え時）

```bash
pip install onnx          # 変換時のみ
python3 export_policy_weights.py
```

> 使用チェックポイントは `model_400.pt`（`model_1999.pt` は不可、handoff §4）。
