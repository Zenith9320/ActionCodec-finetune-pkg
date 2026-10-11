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
│   ├── data/yam.yaml             # ← 数据集定义，列出 10 个 data/yam/TASK-YAM-*_lerobot
│   ├── data/parts_meta/r1lite.yaml
│   └── tokenizer/actioncodec.yaml
├── scripts/train_tokenizer.py    # 微调/评估脚本
├── tools/yam/                    # 原始 MCAP → LeRobot 转换
│   ├── mcap_to_lerobot.py        # 转换脚本（流式写盘，防 OOM）
│   └── tasks_yam.json            # task id → 指令 映射（下游 VLA 前需填真实指令）
├── checkpoints/
│   └── action_tokenizer.pt       # 预训练初始权重（507MB，--init-ckpt，git 不打包）
├── data/stats/yam_stats.json     # YAM 动作 z-score 统计量（缺失时自动重算）
├── pyproject.toml                # 依赖清单（环境重建参考）
├── download_ckpt.sh              # 下载预训练码本 action_tokenizer.pt（约 484MB）
├── convert_data.sh               # 原始 MCAP → data/yam/*_lerobot（微调前先跑）
├── run_finetune.sh               # ★ 微调入口（含训练后评估）
├── run_eval.sh                   # 评估入口（基线对照 / 微调后验收）
└── upload_output.sh              # 微调后上传产物到 HF（私有仓，防临时容器销毁）
```

---

## 预训练权重

`checkpoints/action_tokenizer.pt` 是预训练 ActionCodec 码本，**git 不打包**（`.gitignore`
忽略 `checkpoints/` 与 `*.pt`），需从 Hugging Face 单独下载：

```bash
bash download_ckpt.sh    # 只下 action_tokenizer.pt 一个文件（约 484MB）
```

来源是官方 `OpenGalaxea/G05` 仓库（`action_tokenizer.pt` 在根目录）。脚本默认走国内
镜像 `hf-mirror.com`，海外环境用 `HF_ENDPOINT= bash download_ckpt.sh` 切回官方源。

---

## 数据集

平台上的原始数据是 **MCAP**（`TASK-YAM-XXXX/mcap/success/*.mcap`），微调读取的是
转换后的 **LeRobot v3.0**（`TASK-YAM-XXXX_lerobot/`）。二者不能直接互用，需先转换。

流程：**原始 MCAP →（convert_data.sh）→ LeRobot `_lerobot` →（run_finetune.sh）微调**

```bash
# 在算力平台上（原始数据已通过 PFS 挂载，直接指过去转，无需拷贝）：
RAW_DATA_ROOT=/mnt/workspace/handzero-share/DATASET/TeleopData/YAM bash convert_data.sh
# 产物：data/yam/TASK-YAM-{0001,0002,0004,0006,0007,0008,0010,0013,0014,0015}_lerobot/
```

若想把原始数据先拷到本地再转，把 `RAW_DATA_ROOT` 指向本地目录即可。
转换后会删除旧的 `data/stats/yam_stats.json`，微调启动时自动遍历全量数据重算统计量。

> 说明：`tools/yam/tasks_yam.json` 里除 `TASK-YAM-0001` 外的指令是占位符（task id 本身）。
> 码本微调（自监督重建）不看指令，占位即可；但后续 VLA 全模型微调前需换成真实自然语言指令。

---

## 环境要求

- Python 3.10（`requires-python = ">=3.10.16,<3.11"`）
- CUDA GPU（codec 约 484MB，单卡即可）
- 关键依赖：`torch==2.7.1(+cu128)`、`hydra-core==1.3.2`、`omegaconf`、`transformers`
  完整清单见 `pyproject.toml`（本地用 miniforge env `g05` 验证通过）。

---

## 快速开始

### 0) 下载预训练码本（首次）

```bash
bash download_ckpt.sh    # checkpoints/action_tokenizer.pt（约 484MB）
```

### 1) 准备数据（首次）

```bash
RAW_DATA_ROOT=/mnt/workspace/handzero-share/DATASET/TeleopData/YAM bash convert_data.sh
```

### 2) 微调

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

### 3) 评估（基线对照 / 验收）

```bash
bash run_eval.sh                                          # 预训练基线（微调前）
bash run_eval.sh checkpoints/action_tokenizer_yam.pt      # 微调后
```

### 4) 上传产物（收尾，防临时容器销毁）

平台的本地盘任务结束即释放，微调产物必须赶在结束前传到持久化仓库：

```bash
export HF_REPO_ID=Zenith6666/yam-actioncodec   # 目标仓库（脚本已内置此默认值，可不设）
export HF_TOKEN=hf_xxxx                          # 写权限 token（平台密钥注入，勿硬编码）
bash upload_output.sh
```

默认上传 `action_tokenizer_yam.pt` + `yam_stats.json` 两个文件；仓库不存在会自动建私有仓。
改文件清单用 `export UPLOAD_FILES="checkpoints/xxx data/yyy"`。

> 注意：上传走官方 `https://huggingface.co`（`hf-mirror.com` 只读、不支持 push）。
> 若平台直连官方源不通，需自行配代理或改用 BOS 上传。

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
