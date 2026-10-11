# G0.5 YAM ActionCodec 微调

在预训练的跨具身 ActionCodec（RVQ-VAE，Qwen3.5-2B 的动作 tokenizer 部分）上，
用 I2RT YAM 双臂数据做**自监督重建微调**，让码本适配 YAM 的 27 维动作布局。

训练脚本：`scripts/train_tokenizer.py`（官方 `train_vq.py` 未开源，此文件补齐）。
入口：`run_finetune.sh`。

---

## 目录结构

```
ActionCodec-finetune-pkg/
├── src/                          # g05 源码包（微调 + 数据管线）
├── configs/                      # hydra 配置（task/yam.yaml 等 34 个文件）
│   ├── task/yam.yaml             # ← YAM 任务配置（含码本稳定性超参，见下）
│   ├── data/yam.yaml             # ← 数据集定义，指向 data/yam/TASK-YAM-0001_lerobot
│   ├── data/parts_meta/r1lite.yaml
│   └── tokenizer/actioncodec.yaml
├── scripts/train_tokenizer.py    # 微调/评估脚本
├── checkpoints/
│   └── action_tokenizer.pt       # 预训练初始权重（507MB，--init-ckpt）
├── data/stats/yam_stats.json     # YAM 动作 z-score 统计量（datastatics_path）
├── pyproject.toml                # 依赖清单（环境重建参考）
├── run_finetune.sh               # ★ 微调入口（含训练后评估）
└── run_eval.sh                   # 评估入口（基线对照 / 微调后验收）
```

---

## 数据集

数据集**不打包**，需自行准备 LeRobot v3.0 数据集目录 `TASK-YAM-0001_lerobot`，
放到 `data/yam/` 下，或用 `YAM_DATA_ROOT` 指向其父目录（脚本会自动软链）：

```
data/yam/
└── TASK-YAM-0001_lerobot/        # LeRobot v3.0 数据集
    ├── meta/
    ├── data/
    └── videos/
```

```bash
export YAM_DATA_ROOT=/path/to/your/data    # 指向包含 TASK-YAM-0001_lerobot 的父目录
bash run_finetune.sh
```

---

## 环境要求

- Python 3.10（`requires-python = ">=3.10.16,<3.11"`）
- CUDA GPU（codec 约 484MB，单卡即可）
- 关键依赖：`torch==2.7.1(+cu128)`、`hydra-core==1.3.2`、`omegaconf`、`transformers`
  完整清单见 `pyproject.toml`（本地用 miniforge env `g05` 验证通过）。

---

## 快速开始

### 1) 微调

```bash
bash run_finetune.sh
# 等价于：
# python scripts/train_tokenizer.py --task yam \
#   --init-ckpt checkpoints/action_tokenizer.pt \
#   --out      checkpoints/action_tokenizer_yam.pt \
#   --freeze none --epochs 2 --micro-batch 64 --grad-accum 4 --lr 5e-5 --workers 16
```

可覆盖的超参（环境变量）：`EPOCHS / MICRO_BATCH / GRAD_ACCUM / LR / WORKERS`。

训练每 epoch 结束时自动评估，结束后再跑一次完整评估，输出逐键 RMSE + 码本利用率。

### 2) 评估（基线对照 / 验收）

```bash
bash run_eval.sh                                          # 预训练基线（微调前）
bash run_eval.sh checkpoints/action_tokenizer_yam.pt      # 微调后
```

---

## 码本稳定性超参（重要）

`configs/task/yam.yaml` → `tokenizer.vq_config.model_arch` 已固化三个覆盖值，
**不要改回默认**：

| 参数 | 默认值 | 本包 | 作用 |
|---|---|---|---|
| `ema_decay` | 0.95 | **0.99** | 减缓码本漂移 |
| `threshold_ema_dead` | 2 | **0** | `<=0` 禁用死码重置（窄域微调根因） |
| `quantizer_dropout` | 0.5 | **0.0** | 关码本 dropout |

原因：预训练码本利用率接近 100%，在 YAM 单任务窄域上微调时，
默认 `threshold_ema_dead=2` 会让 ~96% 的码字瞬间判死并被随机重置，
导致 commitment_loss 震荡、利用率坍缩至 ~0.7%、RMSE 劣于基线。
关闭死码重置后（`--freeze none`，1 epoch，lr 5e-5）：

| 指标 | 基线 | 微调后 |
|---|---|---|
| rmse_left_control | 0.0431 | **0.0246**（↓43%） |
| rmse_right_control | 0.0123 | 0.0141 |
| 利用率 | 100% | 100% 维持 |
| commitment_loss | — | 稳定 ~0.001 |

> 注意：`utilization=1.0` 在 `threshold_ema_dead=0` 下是度量假象（`cluster_size>=0`
> 恒真）。真实证据是 commitment_loss 稳定 + RMSE 下降。

---

## 下游衔接（VLA 全模型微调）

`configs/task/yam.yaml` 中 `tokenizer.vq_config.ckpt_dir` 已指向
`checkpoints/action_tokenizer_yam.pt`。本次微调产物恰好落在这个路径，
因此之后跑 `scripts/finetune.py`（VLA 全模型）时会自动加载微调后的码本。
注意：`train_tokenizer.py` 内部用 `--init-ckpt` 覆盖 `ckpt_dir`，二者不冲突。
