# G0.5 YAM ActionCodec 微调 · 百度智能云训练包

在预训练的跨具身 ActionCodec（RVQ-VAE，Qwen3.5-2B 的动作 tokenizer 部分）上，
用 I2RT YAM 双臂数据做**自监督重建微调**，让码本适配 YAM 的 27 维动作布局。

训练脚本：`scripts/train_tokenizer.py`（官方 `train_vq.py` 未开源，此文件补齐）。
入口：`run_finetune.sh`。

---

## 目录结构

```
G05_finetune_pkg/
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

## 数据集挂载（PFS）

数据集**不打包**，通过百度智能云 PFS（并行文件系统）挂载。

挂载点下需包含 LeRobot 数据集目录 `TASK-YAM-0001_lerobot`，例如：

```
/mnt/pfs/yam_data/
└── TASK-YAM-0001_lerobot/        # LeRobot v3.0 数据集
    ├── meta/
    ├── data/
    └── videos/
```

脚本通过环境变量 `YAM_DATA_ROOT` 定位挂载点，并自动软链到
`data/yam/TASK-YAM-0001_lerobot`（`configs/data/yam.yaml` 期望的路径）：

```bash
export YAM_DATA_ROOT=/mnt/pfs/yam_data     # 不设置则默认 /mnt/pfs/yam_data
bash run_finetune.sh
```

---

## 环境要求

- Python 3.10（`requires-python = ">=3.10.16,<3.11"`）
- CUDA GPU（codec 约 484MB，单卡即可）
- 关键依赖：`torch==2.7.1(+cu128)`、`hydra-core==1.3.2`、`omegaconf`、`transformers`
  完整清单见 `pyproject.toml`（本地用 miniforge env `g05` 验证通过）。

---

## 自定义训练镜像（可选）

包里已附 `Dockerfile` + 精简 `requirements.txt`，用来构建「只含依赖、不含代码/数据」的训练镜像：

```bash
docker build -t g05-finetune:1.0 .
```

流程：构建 → 推到百度云 **CCR（容器镜像服务）** → 训练任务里选「自定义镜像」填入地址。
详见文末「百度智能云自定义镜像操作」。

---

## 预设镜像 + git clone（免构建镜像）

不构建自定义镜像时，把本目录推成 git 仓库（`.gitignore` 已忽略 `checkpoints/`、
`data/yam/`、`__pycache__`，权重放 PFS），训练任务启动命令：

```bash
cd /workspace && git clone <仓库地址> g05 && cd g05 && \
bash setup_env.sh && \
export YAM_DATA_ROOT=/mnt/pfs/yam_data INIT_CKPT=/mnt/pfs/checkpoints/action_tokenizer.pt && \
bash run_finetune.sh
```

`setup_env.sh` 负责装 torch cu128 + 精简依赖（默认阿里云镜像源，可用
`PYPI_INDEX_URL` / `PYTORCH_INDEX_URL` 覆盖）。预设镜像要求：**CUDA 12.x（驱动 ≥ 570）
+ Python 3.10**，且不要选 PaddlePaddle。

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

---

## 安全注意

- 本包**不含任何 AK/SK / 密钥**。百度智能云 AK/SK 请通过环境变量或平台侧注入，
  不要写进脚本或 Markdown。
- 数据集凭 PFS 挂载访问，无需在脚本里配置对象存储 AK/SK。
- 实验室内网凭据（`chenyuan` / `xiangyushun` 等）不要出现在本包或日志中。

---

## 百度智能云自定义镜像操作

三步：**构建 → 推到 CCR → 训练任务里选用**。

**1) 构建**（本地或云端构建均可）：

```bash
cd G05_finetune_pkg
docker build -t g05-finetune:1.0 .
```

**2) 推到 CCR**（容器镜像服务）：

```bash
# 开通 CCR，拿到 registry 地址（如 registry.baidubce.com），建命名空间 ns + 仓库 repo
docker login registry.baidubce.com          # 交互输入，或从环境变量读凭证，勿硬编码 AK/SK
docker tag  g05-finetune:1.0 registry.baidubce.com/ns/repo:1.0
docker push registry.baidubce.com/ns/repo:1.0
```

**3) 训练任务选用**：

- 创建训练任务 → 框架/镜像选「**自定义镜像**」→ 填入 `registry.baidubce.com/ns/repo:1.0`；
- 用 PFS 挂载数据集 + 代码（本包 `G05_finetune_pkg` 整个目录）；
- 启动命令：`bash run_finetune.sh`（先 `export YAM_DATA_ROOT=...` 指向 PFS 挂载点）。

> 注意：`docker login` 需要凭证（用户名 + 令牌/AK），务必交互输入或走环境变量，
> 不要写进脚本或 `docker build` 的历史层里。
