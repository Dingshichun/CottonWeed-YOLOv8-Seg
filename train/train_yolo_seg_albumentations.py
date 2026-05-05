"""
YOLOv8-seg 强数据增强训练：Albumentations + Mosaic + MixUp

针对"光照变换"和"叶片遮挡"问题，在输入端做强增强：

光照对抗增强（Albumentations）：
  - RandomBrightnessContrast：模拟暴晒和阴天（亮度/对比度随机变化）
  - HueSaturationValue：模拟不同光照条件下的色调/饱和度/明度变化
  - CLAHE：直方图均衡化，提升局部对比度
  - RandomGamma：模拟过曝/欠曝场景
  - ImageCompression：模拟有损压缩引入的噪声
  - Blur / MedianBlur：模拟运动模糊

遮挡对抗增强（Albumentations）：
  - CoarseDropout：模拟叶片/杂草被部分遮挡，随机遮盖矩形区域
  - PixelDropout：模拟椒盐噪声/像素级遮挡

数据融合增强（YOLO 内置）：
  - Mosaic：4 张图拼接训练，提升复杂背景鲁棒性
  - MixUp：两张图按比例混合，缓解过拟合

"""
import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def inject_custom_albumentations():
    """
    将自定义增强 transforms 注入到 YOLOv8 的 Albumentations 类中。

    YOLOv8 内置的 ultralytics.data.augment.Albumentations 类在 __init__ 中
    可通过 transforms 参数传入自定义 transform 列表。但 model.train() 没有直接
    暴露该参数，因此通过 monkey-patch 替换默认 transforms 来实现。

    注意：CoarseDropout / PixelDropout 属于空间变换（spatial transforms），
    需要 BboxParams 同步处理 bbox。非空间变换只修改像素值，不改变 bbox 坐标。
    """
    try:
        import albumentations as A
    except ImportError:
        print("[ERROR] Albumentations 未安装！请先执行：conda activate pytorch_gpu && pip install albumentations")
        return False

    from ultralytics.data.augment import Albumentations as YOLOAlbumentations
    from ultralytics.utils import LOGGER, colorstr

    # 保存原始 __init__
    _original_init = YOLOAlbumentations.__init__

    def _custom_init(self, p: float = 1.0, transforms: list | None = None):
        """
        自定义初始化：使用强增强 transforms 替换 YOLO 默认的弱增强。
        如果通过参数显式传入 transforms，则优先使用传入的。
        """
        if transforms is not None:
            # 用户显式传入 transforms，直接使用原始逻辑
            _original_init(self, p=p, transforms=transforms)
            return

        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
            import albumentations as A
            from ultralytics.utils.checks import check_version

            check_version(A.__version__, "1.0.3", hard=True)

            # ============================================================
            # 自定义强增强 transforms 列表
            # ============================================================

            # ── 光照对抗增强 ──
            light_transforms = [
                # 亮度/对比度随机变化：模拟暴晒(p>1)和阴天(p<1)
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3),  # 亮度变化 ±30%
                    contrast_limit=(-0.3, 0.3),    # 对比度变化 ±30%
                    p=0.5,
                ),
                # 色调/饱和度/明度变化：模拟不同光照条件下的颜色偏移
                A.HueSaturationValue(
                    hue_shift_limit=15,        # 色调偏移 ±15
                    sat_shift_limit=25,        # 饱和度偏移 ±25
                    val_shift_limit=25,        # 明度偏移 ±25
                    p=0.4,
                ),
                # CLAHE 直方图均衡：提升局部对比度，增强暗部细节
                A.CLAHE(
                    clip_limit=(1, 4),          # 裁剪限制
                    tile_grid_size=(8, 8),
                    p=0.3,
                ),
                # Gamma 校正：模拟过曝(gamma<1)和欠曝(gamma>1)
                A.RandomGamma(
                    gamma_limit=(70, 130),      # gamma 范围 [0.7, 1.3]
                    p=0.3,
                ),
                # JPEG 压缩模拟：引入压缩噪声
                A.ImageCompression(
                    quality_range=(50, 95),     # 压缩质量范围
                    p=0.2,
                ),
            ]

            # ── 模糊增强（模拟风吹/运动模糊） ──
            blur_transforms = [
                A.Blur(blur_limit=(3, 7), p=0.2),        # 均值模糊
                A.MedianBlur(blur_limit=(3, 7), p=0.1),   # 中值模糊
            ]

            # ── 遮挡对抗增强（像素级变换，不改变 bbox 坐标） ──
            # CoarseDropout 仅遮盖图像像素区域，目标边界框不应改变。
            # 这样模型才能学习从部分遮挡中识别完整目标。
            # 注意：虽然 Albumentations 将 CoarseDropout/PixelDropout 标记为空间变换，
            # 但我们强制 contains_spatial=False，仅修改像素不触发
            # BboxParams。若触发 BboxParams，YOLOv8 会尝试同步 bbox 但忽略
            # segments，导致 _format_segments 中索引越界崩溃。
            occlusion_transforms = [
                # 随机遮盖矩形区域：模拟叶片/杂草被部分遮挡
                # max_holes=8 表示最多遮盖 8 个矩形
                # max_height/max_width 以像素值给定，最大遮盖 ~15% 图像
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(0.15 * 640),   # 最大高度 96px
                    max_width=int(0.15 * 640),     # 最大宽度 96px
                    min_holes=1,
                    min_height=int(0.02 * 640),    # 最小高度 ~13px
                    min_width=int(0.02 * 640),     # 最小宽度 ~13px
                    fill_value=0,                  # 用黑色填充
                    p=0.3,
                ),
                # 像素级随机丢弃：模拟椒盐噪声/像素遮挡
                A.PixelDropout(
                    dropout_prob=0.02,             # 丢弃 2% 像素
                    per_channel=False,              # 所有通道一致
                    p=0.2,
                ),
            ]

            # 合并所有 transforms：光照 → 模糊 → 遮挡
            T = light_transforms + blur_transforms + occlusion_transforms

            # 注意：所有自定义增强都是像素级变换，不改变 bbox/segments 几何位置。
            # 强制 contains_spatial=False，避免触发 YOLOv8 的 BboxParams 同步逻辑，
            # 防止 segments 与 bbox 索引不一致导致的 IndexError。
            self.contains_spatial = False
            self.transform = A.Compose(T)

            # 设置随机种子保证可复现性
            if hasattr(self.transform, "set_random_seed"):
                import torch
                self.transform.set_random_seed(torch.initial_seed())

            # 打印实际启用的 transforms
            active_transforms = [
                f"{type(x).__name__}(p={x.p:.2f})"
                for x in T if getattr(x, 'p', 0) > 0
            ]
            LOGGER.info(prefix + ", ".join(active_transforms))

        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    # 应用 monkey-patch
    YOLOAlbumentations.__init__ = _custom_init
    print("[INFO] ✅ 强数据增强 Albumentations transforms 已注入到 YOLOv8 数据流水线")
    print("[INFO]    光照增强: RandomBrightnessContrast, HueSaturationValue, CLAHE, RandomGamma, ImageCompression")
    print("[INFO]    模糊增强: Blur, MedianBlur")
    print("[INFO]    遮挡增强: CoarseDropout, PixelDropout")
    return True


def main():
    print("=" * 70)
    print("[INFO] YOLOv8-seg 强数据增强训练启动")
    print("[INFO] Albumentations + Mosaic + MixUp")
    print("=" * 70)

    # ── 数据路径 ──
    yaml_path = PROJECT_ROOT / "CottonWeedDet12" / "Dataset" / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] 数据集配置文件不存在: {yaml_path}")
        return

    # ── 预训练权重 ──
    pretrained_weights = PROJECT_ROOT / "yolov8n-seg.pt"
    if not pretrained_weights.exists():
        print(f"[ERROR] 预训练权重不存在: {pretrained_weights}")
        return

    print(f"[INFO] 数据集配置: {yaml_path}")
    print(f"[INFO] 预训练权重: {pretrained_weights}")

    # ── 注入自定义 Albumentations 强增强 ──
    if not inject_custom_albumentations():
        print("[ERROR] Albumentations 注入失败，请检查环境")
        return

    # ── 加载模型 ──
    model = YOLO(str(pretrained_weights))

    # ── 训练参数 ──
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",
        amp=False,  # FP32 纯精度训练，避免混合精度带来不稳定
        task="segment",

        # ── 数据融合增强（YOLO 内置） ──
        mixup=0.2,              # MixUp 数据融合，两张图按比例混合
        copy_paste=0.3,         # Copy-Paste 实例粘贴增强，具体是将一个实例从一张图复制到另一张图
        close_mosaic=15,        # mosaic 是在训练时将4张图像拼接成一张图像进行训练，close_mosaic参数控制在训练的后期逐渐关闭mosaic增强，15表示在训练的最后15个epoch关闭mosaic增强，这样模型可以更好地适应单张图像的输入。

        # ── 预训练与保存 ──
        pretrained=True,
        project=str(PROJECT_ROOT / "runs" / "segment"),
        name="train_albumentations",
        exist_ok=False,
    )

    print("=" * 70)
    print("[INFO] 强数据增强训练结束！")
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        print(f"[INFO] 最佳模型保存在: {best_path}")
    else:
        print("[WARN] 未找到 best.pt，请检查训练输出")
    print("=" * 70)


if __name__ == "__main__":
    main()