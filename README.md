# 棉田杂草分割 (CottonWeed-YOLOv8-Seg)

基于 YOLOv8-seg 的棉田杂草实例分割项目。

## 数据集

- **来源**：数据集下载地址：https://www.kaggle.com/datasets/omarhesham189/cottonweeddet12 ，下载解压后放置在项目根目录下，还需运行 tools/auto_annotate_all.py 生成多边形掩模。
- **类别数**：12 类（包含棉花作物及多种杂草）
- **标注格式**：YOLO polygon 格式（由 SAM 辅助生成，未进行人工修正，但基本满足部署需求。）
- **图像尺寸**：原始分辨率拍摄，训练时统一缩放至 640×640  
- **模型权重**：百度网盘： https://pan.baidu.com/s/16BY9oPAP_NEanqOtRiRfRw ，提取码: cpdd ，权重下载后放到项目根目录。
## 项目结构

```
CottonWeed-YOLOv8-Seg/
├── CottonWeedDet12/          # 棉花-杂草12类数据集
│   └── Dataset/
│       ├── data.yaml         # 数据集配置文件（路径/类别数/names）
│       ├── images/
│       │   ├── train/        # 训练集图像
│       │   ├── val/          # 验证集图像
│       │   └── test/         # 测试集图像
│       └── labels/
│           ├── train/        # 训练集标签（TXT 格式，包含 polygon 分割标注）
            ├── val/          # 验证集标签
│           └── test/         # 测试集标签
├── deploy/                   # 部署模型到 jetson nano
├── train/                    # 训练/推理脚本
│   ├── train_yolo_seg.py              # ① 基线训练脚本
│   ├── train_yolo_seg_albumentations.py # ② 强增强训练（Albumentations）
│   ├── train_yolo_seg_aug_only.py     # ③ 强增强训练（仅 MixUp/mosaic）
│   ├── train_yolo_seg_cbam.py         # ④ 嵌入 CBAM 注意力（消融实验）
│   ├── train_yolo_seg_cbam_aug.py     # ⑤ CBAM + 强数据增强
│   ├── train_yolo_seg_cbam_step2.py   # ⑥ CBAM + 强增强 + 低学习率微调
│   ├── train_yolo_seg_eca.py          # ⑦ 嵌入 ECA 注意力机制
│   ├── inference_test.py              # 推理测试脚本
│   └── test_seg.py                    # 分割测试脚本
├── tools/                    # 辅助工具
│   └── auto_annotate_all.py  # SAM 自动标注工具
├── yolov8n-seg.pt            # YOLOv8n-seg 预训练权重
├── mobile_sam.pt             # MobileSAM 权重（用于自动标注）
├── runs/segment/             # 训练结果（按实验分组）
│   ├── base_train/           # 实验①：基线训练
│   ├── train_albumentations2/# 实验②：强增强（Albumentations）
│   ├── train_aug_only/       # 实验③：强增强（仅增强，不换模型）
│   ├── train_cbam_only/      # 实验④：CBAM 注意力机制（无增强）
│   ├── train_cbam_aug/       # 实验⑤：CBAM + 强增强
│   ├── train_cbam_step2/     # 实验⑥：CBAM 微调（lr=0.001）
│   └── train_eca_only/       # 实验⑦：ECA 注意力机制
└── README.md
```

## 实验总览

| # | 实验名称 | 模型 | 描述 |
|---|---------|------|------|
| ① | `base_train` | `yolov8n-seg.pt` | 基线训练，默认超参数 |
| ② | `train_albumentations2` | `yolov8n-seg.pt` | 强增强训练（HSV/mosaic/mixup=0.2 + 关闭mosaic轮数=15） |
| ③ | `train_aug_only` | `yolov8n-seg.pt` | 仅使用强增强策略训练 |
| ④ | `train_cbam_only` | `yolov8n-seg-cbam.yaml` | 嵌入 CBAM 注意力模块，默认增强 |
| ⑤ | `train_cbam_aug` | `yolov8n-seg-cbam.yaml` | CBAM + 强增强 |
| ⑥ | `train_cbam_step2` | `yolov8n-seg-cbam.yaml` | CBAM + 强增强 + 低学习率微调（lr0=0.001） |
| ⑦ | `train_eca_only` | `yolov8n-seg-eca.yaml` | 嵌入 ECA 注意力模块，默认增强 |

## 训练配置对比

| 参数 | ① base_train | ② albumentations2 | ③ aug_only | ④ cbam_only | ⑤ cbam_aug | ⑥ cbam_step2 | ⑦ eca_only |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **基础模型** | yolov8n-seg | yolov8n-seg | yolov8n-seg | yolov8n-seg-cbam | yolov8n-seg-cbam | yolov8n-seg-cbam | yolov8n-seg-eca |
| **Epochs** | 100 | 100 | 100 | 100 | 100 | **150** | 100 |
| **Batch Size** | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| **Image Size** | 640 | 640 | 640 | 640 | 640 | 640 | 640 |
| **学习率 (lr0)** | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | **0.001** | 0.01 |
| **最终学习率 (lrf)** | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **余弦退火 (cos_lr)** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | ✗ |
| **Dropout** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.1** | 0.0 |
| **预热轮数** | 3 | 3 | 3 | 3 | 3 | **5** | 3 |
| **保存间隔** | -1 | -1 | -1 | -1 | -1 | **10** | -1 |
| **mosaic** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **mixup** | 0.0 | 0.2 | 0.2 | 0.0 | 0.2 | 0.2 | 0.0 |
| **close_mosaic** | 10 | 15 | 15 | 10 | 15 | 15 | 10 |
| **HSV-H** | 0.015 | 0.015 | 0.015 | 0.015 | 0.01 | 0.01 | 0.015 |
| **HSV-S** | 0.7 | 0.7 | 0.7 | 0.7 | 0.5 | 0.5 | 0.7 |
| **HSV-V** | 0.4 | 0.4 | 0.4 | 0.4 | 0.3 | 0.3 | 0.4 |
| **scale** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| **erasing** | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| **训练耗时** | 68 min | 80 min | 78 min | 69 min | 78 min | 118 min | 68 min |

## 最佳验证集指标（按 Box mAP50-95 降序排列）

| 排名 | 实验 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | 最佳 Epoch | Avg15 Box | Avg15 Mask |
|:---:|------|:---------:|:------------:|:----------:|:-------------:|:----------:|:---------:|:----------:|
| 🥇 | **③ aug_only** | **0.9320** | **0.8351** | 0.9182 | 0.7348 | 90 | 0.8307 | 0.7303 |
| 🥈 | **② albumentations2** | 0.9246 | 0.8259 | 0.9197 | 0.7327 | 92 | 0.8227 | 0.7277 |
| 🥉 | **⑥ cbam_step2** | 0.9284 | 0.8253 | **0.9263** | **0.7363** | 93 | 0.8219 | 0.7348 |
| 4 | ① base_train | 0.9232 | 0.8218 | 0.9127 | 0.7234 | 98 | 0.8178 | 0.7204 |
| 5 | ⑤ cbam_aug | 0.8847 | 0.7696 | 0.8721 | 0.6740 | 100 | 0.7655 | 0.6694 |
| 6 | ⑦ eca_only | 0.8843 | 0.7668 | 0.8770 | 0.6767 | 99 | 0.7608 | 0.6722 |
| 7 | ④ cbam_only | 0.8813 | 0.7580 | 0.8744 | 0.6713 | 94 | 0.7530 | 0.6646 |

> **说明：**
> - **Box mAP50-95**：目标检测框（Bounding Box）在不同 IoU 阈值（0.50~0.95，步长 0.05）下的平均精度
> - **Mask mAP50-95**：实例分割掩码（Mask）在所有 IoU 阈值下的平均精度
> - **Avg15**：最后 15 个 epoch 的平均值，反映训练的稳定性

## 各实验详细分析

### 🥇 实验③ `train_aug_only` — 最佳检测精度

- **Box mAP50-95 = 0.8351 ⬆**，优于基线 **+0.0133**
- **Mask mAP50-95 = 0.7348 ⬆**，优于基线 **+0.0114**
- **分析**：在不修改模型结构的前提下，仅通过增强数据增强策略（mosaic + mixup=0.2 + 延迟关闭 mosaic 至 epoch 15）就取得了最佳检测效果。这说明针对棉田杂草场景，更丰富的数据增强有助于提升模型的泛化能力，尤其是在杂草形态多变、光照条件复杂的田间场景中。
- **训练耗时**：78 分钟（比基线增加约 15%）

### 🥈 实验② `train_albumentations2` — 强增强（Albumentations）

- **Box mAP50-95 = 0.8259 ⬆**，优于基线 **+0.0041**
- **Mask mAP50-95 = 0.7327 ⬆**，优于基线 **+0.0093**
- **分析**：引入 Albumentations 库的额外增强手段（如 RandomBrightnessContrast、CLAHE、RandomGamma 等）进一步提升了分割效果。相比之下，aug_only 无需额外依赖即取得更优结果，因此推荐优先使用 aug_only 方案。
- **训练耗时**：80 分钟

### 🥉 实验⑥ `train_cbam_step2` — 最佳分割精度

- **Box mAP50-95 = 0.8253 ⬆**，优于基线 **+0.0035**
- **Mask mAP50-95 = 0.7363 ⬆**，**所有实验中最高！** 优于基线 **+0.0129**
- **分析**：在 CBAM 注意力机制基础上，使用低学习率（lr0=0.001）微调。虽然 box 检测精度略低于 aug_only，但 mask 分割精度达到了所有实验的最佳值 0.7363。Avg15 指标也显示出良好的训练稳定性。
- **训练耗时**：118 分钟（由于低学习率，收敛较慢，耗时最长）

### 实验① `base_train` — 基线

- **Box mAP50-95 = 0.8218**
- **Mask mAP50-95 = 0.7234**
- 使用 YOLOv8n-seg 默认配置训练 100 epoch，是所有实验的对比基线。

### 实验④⑤⑦ — CBAM/ECA 注意力机制（较低性能）

| 实验 | Box mAP50-95 | Mask mAP50-95 | 与基线差距 |
|------|:---:|:---:|:---:|
| ④ cbam_only | 0.7580 | 0.6713 | -0.0638 / -0.0521 |
| ⑤ cbam_aug | 0.7696 | 0.6740 | -0.0522 / -0.0494 |
| ⑦ eca_only | 0.7668 | 0.6767 | -0.0550 / -0.0467 |

- **分析**：直接嵌入 CBAM/ECA 注意力模块后，精度明显低于基线。可能原因：
  1. YOLOv8n 本身是 nano 级别模型，参数量极小（约 3.2M），额外插入注意力模块后可能导致特征维度不匹配或梯度不稳定
  2. CBAM 的参数量约增加 15%，对于小模型而言，这些额外参数可能难以在有限数据上有效训练
  3. 然而，实验⑥（cbam_step2）通过低学习率微调取得了出色的结果，说明 CBAM 模块本身有潜力，但训练策略至关重要

## 关键结论

1. **数据增强 > 模型结构修改**：在 YOLOv8n 这个轻量级模型上，增强训练数据（增广策略）的效果显著优于插入注意力机制。实验③仅调整增强策略就取得了最佳 box mAP。

2. **CBAM 需特殊训练策略**：CBAM 注意力模块在正常学习率下表现不佳（低于基线），但采用低学习率（lr0=0.001）微调后（实验⑥），Mask 分割精度达到最佳。这说明注意力模块需要更精细的训练策略。

3. **训练稳定性**：所有实验的 Avg15 指标与最佳值非常接近（差距 < 0.01），表明训练过程稳定，未出现过拟合现象。

4. **推荐方案**：
   - **追求检测精度**：使用 `train_aug_only` 方案（Box mAP50-95 = 0.8351）
   - **追求分割精度**：使用 `train_cbam_step2` 方案（Mask mAP50-95 = 0.7363）
   - **快速部署**：基线方案仅需 68 分钟，精度已相当不错（0.8218/0.7234）

## 测试集指标（独立测试集评估）

以下为验证集性能排名前三的模型与基线模型在**独立测试集**上的最终评估结果：

| 排名 | 实验 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Box Precision | Box Recall | Mask Precision | Mask Recall |
|:---:|------|:---------:|:------------:|:----------:|:-------------:|:-------------:|:----------:|:--------------:|:-----------:|
| 🥇 | **② albumentations2** | 0.8846 | **0.7831** | 0.8781 | 0.6868 | 0.8535 | 0.8324 | 0.8822 | 0.8571 |
| 🥈 | **⑥ cbam_step2** | **0.8906** | 0.7805 | **0.8880** | **0.6927** | 0.8832 | 0.8571 | 0.9109 | 0.8857 |
| 🥉 | **① base_train** | 0.8815 | 0.7792 | 0.8774 | 0.6836 | 0.9195 | 0.8571 | 0.9388 | 0.8766 |
| 4 | ③ aug_only | 0.8849 | 0.7751 | 0.8734 | 0.6793 | 0.8430 | 0.8857 | 0.8437 | 0.8857 |

> **测试集说明：**
> - 测试集图像位于 `CottonWeedDet12/Dataset/images/test/`，与训练/验证集完全独立
> - 测试结果为 `runs/segment/test_results/<实验名>/metric.txt` 中的记录值
> - 混淆矩阵、P-R 曲线、F1 曲线等可视化结果保存在对应的 `test_results/<实验名>/` 目录下

### 验证集 vs 测试集对比分析

| 实验 | 验证 Box mAP50-95 | 测试 Box mAP50-95 | 泛化差距 | 验证 Mask mAP50-95 | 测试 Mask mAP50-95 | 泛化差距 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| ① base_train | 0.8218 | 0.7792 | -0.0426 | 0.7234 | 0.6836 | -0.0398 |
| ② albumentations2 | 0.8259 | **0.7831** | **-0.0428** | 0.7327 | 0.6868 | -0.0459 |
| ③ aug_only | 0.8351 | 0.7751 | -0.0600 | 0.7348 | 0.6793 | -0.0555 |
| ⑥ cbam_step2 | 0.8253 | 0.7805 | -0.0448 | 0.7363 | **0.6927** | -0.0436 |

### 测试集关键发现

1. **albumentations2 泛化最优**：虽然 aug_only 在验证集上排名第一，但在测试集上 albumentations2 的 Box mAP50-95 最高（0.7831），说明 Albumentations 增强策略带来了更好的泛化能力。

2. **cbam_step2 分割最强**：在测试集上 cbam_step2 的 Mask mAP50-95 达到 0.6927，Box mAP50 达到 0.8906，均为所有模型最高，且泛化差距较小（-0.0436），印证了低学习率微调策略的有效性。

3. **aug_only 泛化最弱**：验证集表现最好的 aug_only 在测试集上下降幅度最大（Box mAP 下降 0.0600），存在一定过拟合风险。

4. **基线模型表现稳健**：base_train 在测试集上 Box mAP50-95 = 0.7792，尽管验证集排名第四，但测试表现与第二名仅差 0.0013，显示出良好的稳定性。

### 最终推荐

| 应用场景 | 推荐模型 | 测试 Box mAP50-95 | 测试 Mask mAP50-95 |
|------|------|:---:|:---:|
| 追求检测精度与泛化能力 | ② albumentations2 | **0.7831** | 0.6868 |
| 追求分割精度 | ⑥ cbam_step2 | 0.7805 | **0.6927** |
| 快速部署（训练最快） | ① base_train | 0.7792 | 0.6836 |

测试结果文件位于 `runs/segment/test_results/`，每个子目录包含：
- `metric.txt` — 测试指标数值
- `BoxPR_curve.png` / `MaskPR_curve.png` — P-R 曲线
- `BoxF1_curve.png` / `MaskF1_curve.png` — F1 曲线
- `confusion_matrix.png` / `confusion_matrix_normalized.png` — 混淆矩阵
- `val_batch*_pred.jpg` / `val_batch*_labels.jpg` — 预测/真实标签对比

## 生成结果文件说明

每个实验目录 (`runs/segment/<experiment>/`) 下包含：

| 文件 | 说明 |
|------|------|
| `results.csv` | 每个 epoch 的训练/验证指标 |
| `results.png` | 训练曲线图（loss + mAP 随 epoch 变化） |
| `BoxPR_curve.png` | Box 检测的 P-R 曲线 |
| `MaskPR_curve.png` | Mask 分割的 P-R 曲线 |
| `BoxF1_curve.png` | Box 检测的 F1 置信度曲线 |
| `MaskF1_curve.png` | Mask 分割的 F1 置信度曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `labels.jpg` | 标签分布统计 |
| `train_batch*.jpg` | 训练批次样本可视化 |
| `val_batch*_pred.jpg` | 验证批次的预测结果 |
| `val_batch*_labels.jpg` | 验证批次的真实标签 |
| `weights/best.pt` | 最佳权重文件 |
| `weights/last.pt` | 最后一轮权重文件 |
| `args.yaml` | 训练超参数配置 |

## 复现训练

```bash
# 环境安装
git clone https://github.com/Dingshichun/CottonWeed-YOLOv8-Seg.git
cd CottonWeed-YOLOv8-Seg
pip install -r requirements.txt
```

```bash
# 快速开始

# 首先将 CottonWeedDet12/Dataset/data.yaml 复制到自己下载的数据集目录 CottonWeedDet12/Dataset 中，并修改其中的 path 为自己的路径

# 使用 SAM 生成多边形掩码，会覆盖原来的矩形框数据
python tools/auto_annotate_all.py

# 基线训练
python train/train_yolo_seg.py

# 仅增强训练
python train/train_yolo_seg_aug_only.py

# 强增强训练（Albumentations）
python train/train_yolo_seg_albumentations.py

# 评估模型
python train/test_seg.py

# 单图推理
python train/inference_test.py --source <image_path> --weights <weights_path>
```


