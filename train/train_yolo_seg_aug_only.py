"""
YOLOv8-seg 仅数据增强消融实验
与 baseline 唯一变量：开启 mixup + copy_paste 数据增强，
使用标准 yolov8n-seg.pt 模型，不添加任何注意力模块，
用于独立验证数据增强的贡献。
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    print('[INFO] 仅数据增强消融实验启动 ...')

    # 数据路径
    yaml_path = PROJECT_ROOT / 'CottonWeedDet12' / 'Dataset' / 'data.yaml'
    if not yaml_path.exists():
        print(f"[ERROR] 数据集配置文件不存在: {yaml_path}")
        return

    # 使用标准 yolov8n-seg 模型（无任何注意力模块）
    pretrained_weights = PROJECT_ROOT / 'yolov8n-seg.pt'
    if not pretrained_weights.exists():
        print(f"[ERROR] 预训练权重不存在: {pretrained_weights}")
        return

    print(f'[INFO] 使用数据集: {yaml_path}')
    print(f'[INFO] 使用标准模型: {pretrained_weights}')

    model = YOLO(str(pretrained_weights))

    # ---------- 训练参数 (消融实验版：仅数据增强) ----------
    results = model.train(
        data=str(yaml_path),
        epochs=100,                    # 与 baseline 一致
        imgsz=640,
        batch=8,
        device='0',
        amp=False,
        task='segment',

        # ── 数据增强 (唯一变量：开启 mixup + copy_paste) ──
        mixup=0.2,                     # baseline 默认 0.0
        copy_paste=0.3,                # baseline 默认 0.0
        close_mosaic=15,               # baseline 默认 10

        # ── 预训练与保存 ──
        pretrained=True,
        project=str(PROJECT_ROOT / 'runs' / 'segment'),
        name='train_aug_only',
        exist_ok=False,
    )

    print('[INFO] 仅数据增强消融实验训练结束！')
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_path.exists():
        print(f'[INFO] 最佳模型保存在: {best_path}')
    else:
        print('[WARN] 未找到 best.pt，请检查训练输出')


if __name__ == '__main__':
    main()