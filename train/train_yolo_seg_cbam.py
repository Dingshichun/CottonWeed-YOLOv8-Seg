"""
YOLOv8-seg + CBAM 注意力训练脚本 (消融实验版 - 仅 CBAM)
仅保留一个变量与 baseline 不同：
  1. CBAM 通道+空间注意力 → 验证注意力机制的效果
数据增强设置与 baseline 完全一致（使用 ultralytics 默认值），
用于独立对比 CBAM 的贡献。
"""
import sys 
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 将 CBAM 动态注册到 ultralytics.nn.tasks 的全局命名空间 ──
# ultralytics 的 parse_model 通过 globals() 查找模块类名，
# 因此必须在 import YOLO 之前完成注入
import ultralytics.nn.tasks as tasks
from models.cbam import CBAM
tasks.CBAM = CBAM  # 注入到 tasks 模块的全局空间

from ultralytics import YOLO


def main():
    print('[INFO] CBAM 改进模型训练启动 (仅 CBAM 消融实验) ...')

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

    # ---------- 训练参数 (消融实验版：仅 CBAM，数据增强与 baseline 一致) ----------
    results = model.train(
        data=str(yaml_path),
        epochs=100,                    # 与 baseline 一致
        imgsz=640,
        batch=8,
        device='0',
        amp=False,
        task='segment',

        # ── 预训练与保存 ──
        pretrained=True,
        project=str(PROJECT_ROOT / 'runs' / 'segment'),
        name='train_cbam_only',
        exist_ok=False,
    )

    print('[INFO] CBAM 消融实验训练结束 (仅 CBAM)！')
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_path.exists():
        print(f'[INFO] 最佳模型保存在: {best_path}')
    else:
        print('[WARN] 未找到 best.pt，请检查训练输出')


if __name__ == '__main__':
    main()