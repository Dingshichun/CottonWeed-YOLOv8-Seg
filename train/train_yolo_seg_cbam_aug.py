"""
YOLOv8-seg + CBAM 注意力 + 强数据增强训练脚本
变量：
  1. CBAM 通道+空间注意力
  2. 强数据增强（mixup=0.2 + copy_paste=0.3 + close_mosaic=15）
独立验证 CBAM + 数据增强的联合贡献。
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
    print('[INFO] CBAM + 强数据增强训练启动 ...')

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

    # ---------- 训练参数 (CBAM + 强增强) ----------
    results = model.train(
        data=str(yaml_path),
        epochs=100,                    # 与 baseline 一致
        imgsz=640,
        batch=8,
        device='0',
        amp=False,
        task='segment',

        # ── 强数据增强 ──
        mixup=0.2,                     # baseline 默认 0.0
        copy_paste=0.3,                # baseline 默认 0.0
        close_mosaic=15,               # baseline 默认 10
        hsv_h=0.01,                    # HSV 增强（降低幅度，避免过强颜色变换）
        hsv_s=0.5,
        hsv_v=0.3,

        # ── 预训练与保存 ──
        pretrained=True,
        project=str(PROJECT_ROOT / 'runs' / 'segment'),
        name='train_cbam_aug',
        exist_ok=False,
    )

    print('[INFO] CBAM + 强数据增强训练结束！')
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_path.exists():
        print(f'[INFO] 最佳模型保存在: {best_path}')
    else:
        print('[WARN] 未找到 best.pt，请检查训练输出')


if __name__ == '__main__':
    main()