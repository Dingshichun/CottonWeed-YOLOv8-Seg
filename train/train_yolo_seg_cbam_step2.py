"""
YOLOv8-seg + CBAM 注意力 + 强数据增强 + 低学习率微调训练脚本
在 CBAM + 强增强基础上，采用低学习率（lr0=0.001）和余弦退火策略，
延长训练至 150 epoch，并引入 dropout=0.1 正则化，
用于验证注意力机制在精细微调下的潜力。
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 将 CBAM 动态注册到 ultralytics.nn.tasks 的全局命名空间 ──
import ultralytics.nn.tasks as tasks
from models.cbam import CBAM
tasks.CBAM = CBAM  # 注入到 tasks 模块的全局空间

from ultralytics import YOLO


def main():
    print('[INFO] CBAM + 强增强 + 低学习率微调训练启动 ...')

    # 数据路径
    yaml_path = PROJECT_ROOT / 'CottonWeedDet12' / 'Dataset' / 'data.yaml'
    if not yaml_path.exists():
        print(f"[ERROR] 数据集配置文件不存在: {yaml_path}")
        return

    # CBAM 模型配置文件
    cbam_yaml = PROJECT_ROOT / 'models' / 'yolov8n-seg-cbam.yaml'
    if not cbam_yaml.exists():
        print(f"[ERROR] CBAM 模型配置文件不存在: {cbam_yaml}")
        return

    print(f'[INFO] 使用数据集: {yaml_path}')
    print(f'[INFO] 使用模型配置: {cbam_yaml}')

    # 从 YAML 构建 CBAM 模型
    model = YOLO(str(cbam_yaml))

    # 加载预训练权重（仅加载兼容层的参数）
    pretrained_weights = PROJECT_ROOT / 'yolov8n-seg.pt'
    if pretrained_weights.exists():
        print(f'[INFO] 从预训练权重加载: {pretrained_weights}')
        pretrained_model = YOLO(str(pretrained_weights))
        pretrained_dict = pretrained_model.ckpt['model'].float().state_dict()
        model_dict = model.model.state_dict()

        # 过滤：只加载形状完全匹配的参数（跳过 CBAM 新增层和索引偏移的不兼容层）
        filtered_dict = {}
        skipped_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append(k)

        model.model.load_state_dict(filtered_dict, strict=False)
        print(f'[INFO] 预训练权重迁移完成: 加载 {len(filtered_dict)}/{len(pretrained_dict)} 层, '
              f'跳过 {len(skipped_keys)} 层')
    else:
        print('[WARN] 未找到预训练权重，从 scratch 训练')

    # ---------- 训练参数 (CBAM + 强增强 + 低学习率微调) ----------
    results = model.train(
        data=str(yaml_path),
        epochs=150,                    # 延长训练轮数
        imgsz=640,
        batch=8,
        device='0',
        amp=False,
        task='segment',

        # ── 低学习率微调 ──
        lr0=0.001,                     # 低初始学习率（baseline 为 0.01）
        lrf=0.01,                      # 最终学习率因子
        cos_lr=True,                   # 余弦退火学习率调度
        warmup_epochs=5,               # 更长的预热轮数（baseline 默认 3）
        dropout=0.1,                   # 添加 dropout 正则化，防止过拟合

        # ── 强数据增强 ──
        mixup=0.2,
        copy_paste=0.3,
        close_mosaic=15,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        # ── 预训练与保存 ──
        pretrained=True,
        project=str(PROJECT_ROOT / 'runs' / 'segment'),
        name='train_cbam_step2',
        exist_ok=False,
    )

    print('[INFO] CBAM + 强增强 + 低学习率微调训练结束！')
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_path.exists():
        print(f'[INFO] 最佳模型保存在: {best_path}')
    else:
        print('[WARN] 未找到 best.pt，请检查训练输出')


if __name__ == '__main__':
    main()