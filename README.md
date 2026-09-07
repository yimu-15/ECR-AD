# ECR-AD 2.0
# Evidence-Calibrated Cross-Modal Routing for Zero-Shot Anomaly Detection

## 项目概述

ECR-AD 2.0 是一种用于零样本工业异常检测 (ZSAD) 的计算机视觉算法。
核心思想从传统的 "Fuse everything" 转向 "Trust selectively" ——
当视觉证据 (DINOv2) 与语义先验 (CLIP) 发生冲突时，
模型通过 **Asymmetric Reliability Arbitration** 动态分配信任权重。

## 核心贡献

1. **Cross-Modal Evidence Reliability**: 首次从"模态可靠性"而非简单特征融合的角度建模视觉与语义信息。
2. **Evidence-Calibrated Router**: 联合不确定性、分歧度与原型距离，动态决定每个 Patch 应信任视觉还是语义。
3. **Conflict-Aware Prototype Learning**: 利用模态间的分歧来发现模糊或异常区域，增强检测鲁棒性。

## 项目结构

```
ECR-AD/
├── configs/              # 配置文件
│   └── visa.yaml         # VisA 数据集配置
├── models/               # 模型定义
│   ├── __init__.py
│   ├── dino_encoder.py   # DINOv2 ViT-S/14 视觉编码器
│   ├── clip_encoder.py   # CLIP ViT-B/32 图文编码器
│   ├── projection.py     # Projection Head + 对比损失
│   ├── evidence.py       # 视觉/语义证据提取
│   ├── uncertainty.py    # 不确定性计算
│   ├── disagreement.py   # 跨模态分歧度
│   ├── reliability_router.py  # 核心路由仲裁器
│   └── ecr_ad.py         # 完整 ECR-AD Pipeline
├── training/             # 训练脚本
│   ├── __init__.py
│   └── train_projection.py   # 跨模态对齐预训练
├── inference/            # 推理脚本
│   ├── __init__.py
│   └── zero_shot.py      # Zero-Shot 推理器
├── evaluation/           # 评估脚本
│   ├── __init__.py
│   └── metrics.py        # AUROC/AUPR 指标
└── visualization/        # 可视化脚本
    ├── __init__.py
    └── anomaly_map.py    # 热力图/权重可视化
```

## 环境配置

### 依赖

- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- timm >= 1.0
- open_clip_torch
- scikit-learn
- opencv-python
- Pillow
- matplotlib

### 安装

```bash
pip install torch torchvision timm open_clip_torch scikit-learn opencv-python Pillow matplotlib
```

### 模型权重

首次运行时，timm 和 open_clip 会自动从 HuggingFace 下载预训练权重。
国内用户请设置镜像:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## 使用方法

### Step 1: 跨模态对齐预训练 (Projection Head)

```bash
python training/train_projection.py \
    --source_dir /path/to/ImageNet-1K/train \
    --output checkpoints/projection_head.pth \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3
```

### Step 2: Zero-Shot 推理

```python
from models.ecr_ad import ECRAD
from inference.zero_shot import ZeroShotInference
from PIL import Image

# 加载模型
model = ECRAD(device="cuda", prompt_mode="class_agnostic")
model.projection.load_state_dict(
    torch.load("checkpoints/projection_head.pth")['projection_state']
)
model.eval()

# 推理
inference = ZeroShotInference(model, device="cuda")
images = [Image.open("test_image.jpg")]
results = inference.infer_batch(images)

print(f"Image-level anomaly score: {results['image_scores'][0]:.4f}")
```

### Step 3: 可视化

```python
from visualization.anomaly_map import visualize_anomaly_map
import numpy as np

# 读取原始图像和预测结果
rgb = np.array(Image.open("test_image.jpg"))
anomaly_map = results['pixel_maps'][0]

# 可视化
fig = visualize_anomaly_map(rgb, anomaly_map, save_path="output/anomaly_map.png")
```

## 数据集下载与放置 (Dataset Download)

由于各数据集体积较大（合计约 10GB+），**本仓库不直接包含数据集文件**。
请使用 `scripts/download_datasets.py` 一键下载、解压并整理到仓库约定的目录结构
（与 [configs/dataset.yaml](./dataset.yaml) 中的 `root` 路径一致）。

### 支持的基准

| 数据集 | 类别数 | 官方链接 | 体积 | 许可 |
|--------|-------|----------|------|------|
| **MVTec AD** | 15 | https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz | ~4.9GB | CC BY-NC-SA 4.0 |
| **VisA** | 12 | https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar | ~2GB | CC BY-NC-SA 4.0 |
| **BTAD** | 3 | https://avires.dimi.uniud.it/papers/btad/btad.zip | ~1GB | Research only |
| **MPDD** | 6 | 需官网申请，无匿名直链 | ~1.7GB | Research only |

> 说明：MVTec AD 链接来自官网下载页的整包地址，若失效可用 `--url` 指定镜像；
> VisA / BTAD 均为官方 S3 / 机构直链；MPDD 官方按网页申请分发，脚本仅提供目录自检与引导。

### 使用

```bash
# 在项目根目录执行。下载到 datasets/_archives/（已 gitignore），支持断点续传。

# 下载并整理单个数据集
python -m scripts.download_datasets --dataset mvtec
python -m scripts.download_datasets --dataset visa
python -m scripts.download_datasets --dataset btad

# 全部数据集依次处理
python -m scripts.download_datasets --all

# 指定镜像链接（如官方链接失效）
python -m scripts.download_datasets --dataset mvtec --url <镜像URL>

# 仅自检本地目录结构（不联网）
python -m scripts.download_datasets --check
```

脚本会在**已就绪时自动跳过**下载；压缩包默认保留（便于断点续传 / 重装），
可用 `--no-keep-archive` 在整理成功后删除。MPDD 需先从官网获取压缩包后执行
`python -m scripts.download_datasets --dataset mpdd --url <下载链接>`，或手动放置。

### 目标目录结构

脚本整理后的结构与加载器 [models/dataset.py](../models/dataset.py) 的读取口径一致：

```
datasets/
├── MVTec AD/                  # mvtec
│   └── <类别 15>/             # train/good, test/<缺陷>, ground_truth/<缺陷>
├── visa/VisA/data/VisA_20220922/   # visa（保留官方原始目录树）
│   ├── <类别 12>/             # Data/Images/{Normal,Anomaly}, Data/Masks/Anomaly
│   └── split_csv/1cls.csv     # 官方划分文件
├── BTAD/                      # btad
│   └── <01|02|03>/            # train/ok, test/{ok,ko}, ground_truth/ko
└── MPDD/                      # mpdd（手动放置）
    └── <类别 6>/              # train/good, test/<缺陷>, ground_truth/<缺陷>
```

每个类别内部与 `configs/dataset.yaml` 的 `train_dir / test_dir / gt_dir` 一一对应。

## 实验设置

### 数据集

- **MVTec AD**: 15 个工业类别
- **VisA**: 12 个工业类别
- **BTAD**: 3 个类别
- **MPDD**: 6 个类别

### 实验设定

- **Setting A (Class-Agnostic ZSAD)**: 使用 "a normal object" / "an anomalous object"
- **Setting B (Class-Aware ZSAD)**: 使用 "a normal bottle" / "an anomalous bottle" (仅类别名)
- **Setting C (Few-Shot Extension)**: 使用 K 张正常样本构建 Prototype Bank

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lambda_weight | 1.0 | 分歧度放大系数 |
| temperature | 1.0 | Router softmax 温度 |
| sigma | 1.0 | Visual typicality 温度 |
| dropout_rate | 0.25 | Modality dropout 概率 |
| latent_dim | 256 | Projection latent 维度 |

## 论文结构

1. Introduction
2. Related Work
   - 2.1 Vision Foundation Models
   - 2.2 Vision-Language Anomaly Detection
   - 2.3 Prototype Learning
   - 2.4 Uncertainty-aware Learning
3. Method
   - 3.1 Problem Formulation
   - 3.2 Visual Evidence Modeling
   - 3.3 Semantic Prototype Modeling
   - 3.4 Cross-Modal Disagreement
   - 3.5 Reliability Estimation
   - 3.6 Evidence-Calibrated Routing
   - 3.7 Prototype Consistency Loss
4. Experiments
   - 4.1 Datasets
   - 4.2 Implementation Details
   - 4.3 Comparison with SOTA
   - 4.4 Ablation Study
   - 4.5 Reliability Analysis
   - 4.6 Cross-Modal Conflict Analysis
   - 4.7 Visualization
5. Discussion
6. Conclusion

## 硬件要求

- GPU: 8GB+ VRAM (推理模式)
- 训练 Projection Head: 建议 11GB+ (ImageNet-1K, batch=32)
- 内存: 16GB+
- 磁盘: 50GB+ (数据集 + 模型权重)

## License

MIT License
