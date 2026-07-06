# Kinikun RL ポリシー 実機デプロイ 引き継ぎ

Kinikun の空圧アーム（`arm1_joint`）を、学習済み RL ポリシーで関節角追従させるための引き継ぎ資料です。
**このリポジトリ（`aerial_lab`）で学習したポリシーを、実機コマンド送信スクリプトを持つ別リポジトリ側で動かす**のが目的です。

---

## 0. 前提：model は2つあるが、実機に載せるのは「ポリシー」だけ

| model | 実体 | 役割 | 実機で必要？ |
|---|---|---|---|
| **NARXモデル** (`narx_model.pt` + `narx_meta.json`) | 空圧アームの代理ダイナミクス（圧力→関節角を予測） | 学習時にsim内で実機の代わりをする「疑似実機」 | ❌ **不要** |
| **RLポリシー** (`policy.pt` / `policy.onnx`) | 学習したコントローラ（観測→圧力コマンド） | 目標角に追従するよう2ch圧力を出す本体 | ✅ **これを載せる** |

実機そのものが NARX の代わりになるので、NARX は持っていきません。

---

## 1. コピーするファイル

学習ログ内の `exported/` にある**エクスポート済みポリシー**をコピーします（isaaclab / torch 依存を切り離せる）。

- `logs/rsl_rl/kinikun_arm/2026-07-02_06-57-01/exported/policy.pt` … TorchScript（`torch.jit.load`）
- `logs/rsl_rl/kinikun_arm/2026-07-02_06-57-01/exported/policy.onnx` … ONNX（`onnxruntime`、torch不要）
- `logs/rsl_rl/kinikun_arm/2026-07-02_06-57-01/params/env.yaml` … 学習時の環境設定（可動域・スケール確認用）

> **使うチェックポイント：`model_400.pt`（最良）**。`model_1999.pt` は学習後半で崩壊しているため使わないこと。

### エクスポート方法（未生成の場合）

`aerial_lab` リポジトリで play.py を1回走らせると `exported/` が生成される：

```bash
python scripts/rsl_rl/play.py --task Kinikun-Arm-Direct-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/kinikun_arm/2026-07-02_06-57-01/model_400.pt
```

---

## 2. ポリシーの入出力（実機で厳密に再現する「契約」）

ポリシーは**瞬時値だけの feedforward MLP**（隠れ層 [128,128], activation=elu）。
**履歴バッファは不要**（履歴が要るのはNARX側だけ）。観測の正規化も無し（`actor_obs_normalization=False`）＝生の7次元をそのまま入力する。

### 制御周期
**200 Hz（dt = 0.005 s）**。学習時と同じレートで回すこと（NARX の `dt_est ≈ 0.004997 s`）。

### 観測（入力）7次元（すべて生値、単位はrad系）

| idx | 中身 | 単位 |
|---|---|---|
| 0 | `joint_pos`（現在の関節角） | rad |
| 1 | `joint_vel`（関節角速度） | rad/s |
| 2 | `target_joint_pos`（目標角） | rad |
| 3 | `position_error = target − joint_pos` | rad |
| 4 | `last_action[0]`（前ステップ action ch0, [-1,1]） | – |
| 5 | `last_action[1]`（前ステップ action ch1, [-1,1]） | – |
| 6 | `pressure_delta = p1 − p2`（今ステップの圧力コマンド差） | MPa |

### 行動（出力）2次元 `a ∈ [-1, 1]`（必ず clamp）→ 圧力コマンドへ写像

```
pressure_limit_mpa = 0.6
p1 = 0.5 * (a[0] + 1.0) * pressure_limit_mpa   # MPa, 範囲 [0, 0.6]
p2 = 0.5 * (a[1] + 1.0) * pressure_limit_mpa   # MPa, 範囲 [0, 0.6]
```
この `p1, p2` を実機の2ch空圧コマンドとして送る。

---

## 3. デプロイループ（擬似コード）

**変数更新順が sim の観測生成と一致すること**が肝（`last_action` は obs を作った後に更新、`pressure_delta` は今ステップの圧力を使う）。

```python
import torch
policy = torch.jit.load("policy.pt").eval()   # or onnxruntime.InferenceSession("policy.onnx")

last_action = [0.0, 0.0]
pressure    = [0.0, 0.0]   # MPa

while running:                                   # 200 Hz ループ
    joint_pos, joint_vel = read_encoder()        # rad, rad/s ← 実機センサ
    target = get_target(t)                        # rad（§4の範囲制約に従う）

    obs = [joint_pos, joint_vel, target, target - joint_pos,
           last_action[0], last_action[1], pressure[0] - pressure[1]]

    a = policy(torch.tensor(obs).float()).clamp(-1, 1).tolist()   # 2次元
    pressure = [0.5*(a[0]+1)*0.6, 0.5*(a[1]+1)*0.6]               # MPa
    send_pressure(pressure)                       # ← 実機コマンド送信スクリプト

    last_action = a
    wait_next_tick()                              # 5 ms
```

`read_encoder` / `send_pressure` / `get_target` は実機リポジトリ側のAPI（トピック名等）に差し替える。

---

## 4. sim-to-real の注意点（重要）

- **目標角の範囲**：ポリシーは「可動域の中心 ± 0.7×半幅」の目標しか学習していない（`target_scale = 0.7`）。この外はOOD（挙動保証なし）。
  - 可動域 `[lower, upper]` は `params/env.yaml`、または play 実行時ログの `Arm lower/upper limit` で確認。
  - `center = 0.5*(lower+upper)`, `half = 0.5*(upper-lower)`。許容目標域 = `[center - 0.7*half, center + 0.7*half]`。
  - **実機の関節0点(origin)・符号・単位(rad)を sim の定義に必ず合わせる**。
- **初期条件**：起動時は default 姿勢近く、`last_action = 0`、`pressure = 0` から始める前提。
- **ダイナミクスギャップ**：ポリシーは NARX 代理モデル相手に学習。NARX は実データfitなので近いはずだが、ズレは出る。最初は低速・小振幅の目標で検証開始。
- **使用チェックポイント**：`model_400.pt`（final_tracking_mae ≈ 0.09 rad ≒ 5°）。`model_1999.pt` は不可。
- **安全**：
  - 圧力を `[0, 0.6] MPa` にハードクランプ。
  - 圧力の急変にはレート制限を推奨。
  - E-stop / タイムアウトを実機側に必ず実装。

---

## 5. 参考：学習時の設定（再現・デバッグ用）

- タスク: `Kinikun-Arm-Direct-v0`（`aerial_lab` 内 `tasks/direct/kinikun/kinikun_arm_env.py`）
- 制御: 固定ベース、`sim_dt = 1/200`, `decimation = 1`, `episode_length_s = 10`
- action_space = 2（2ch圧力）, observation_space = 7
- 報酬（すべてペナルティ, ≤0）: `track`(追従誤差²), `joint_vel`(速度²×0.05), `action_rate`(action変化²×0.01)
- アルゴリズム: rsl_rl PPO, hidden [128,128] elu, `learning_rate=3e-4` adaptive(`desired_kl=0.01`), `entropy_coef=0.005`
- 既知の学習挙動: iter≈425 で最良（MAE 0.09）、後半で noise_std が再膨張して崩壊 → **中盤チェックポイントを使う**。

---

## 6. 質問があれば `aerial_lab` 側（このリポジトリ）へ

- ポリシー入出力の定義元: `source/aerial_lab/aerial_lab/tasks/direct/kinikun/kinikun_arm_env.py`
  - 観測: `_get_observations`（L222-239）
  - 行動→圧力写像: `_pre_physics_step`（L174-180）
- リポジトリ: `https://github.com/jin2004ut/aerial_lab.git`（branch: `kinikun`）
