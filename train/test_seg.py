"""
在测试集上评估 YOLOv8-seg 分割模型
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便正确导入模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    models_name=['base_train','train_aug_only','train_cbam_step2','train_albumentations2']
    # 模型路径
    model_path = PROJECT_ROOT / 'runs' / 'segment' / models_name[3] / 'weights' / 'best.pt'
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 数据集配置文件
    data_yaml = PROJECT_ROOT / 'CottonWeedDet12' / 'Dataset' / 'data.yaml'
    if not data_yaml.exists():
        print(f"[ERROR] 数据配置文件不存在: {data_yaml}")
        return

    print(f"[INFO] 数据集配置: {data_yaml}")
    print("[INFO] 开始在测试集上评估...")
    print("-" * 60)

    # 在测试集上验证
    results = model.val(
        data=str(data_yaml),
        split='test',
        imgsz=640,
        batch=8,
        device='0',
        # 路径与结果保存相关参数
        project='test_results', # 根目录
        name=models_name[3],    # 子目录名
        exist_ok=False,         # 如果目录已存在，自动创建新的递增目录
        plots=True              # 保存可视化的验证图
    )

    print("-" * 60)
    print("[INFO] 测试集评估完成！")
    print(f"Box mAP@50:     {results.box.map50:.4f}")
    print(f"Box mAP@50-95:  {results.box.map:.4f}")
    print(f"Mask mAP@50:    {results.seg.map50:.4f}")
    print(f"Mask mAP@50-95: {results.seg.map:.4f}")
    print(f"Box Precision:  {results.box.p[1]:.4f}")
    print(f"Box Recall:     {results.box.r[1]:.4f}")
    print(f"Mask Precision: {results.seg.p[1]:.4f}")
    print(f"Mask Recall:    {results.seg.r[1]:.4f}")
    


if __name__ == '__main__':
    main()