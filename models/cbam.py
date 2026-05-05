"""
CBAM (Convolutional Block Attention Module)
通道注意力 + 空间注意力，兼容 ultralytics 模型构建接口。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：AvgPool + MaxPool → 共享 1x1 Conv → Sigmoid"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(x.mean(dim=[2, 3], keepdim=True))
        max_out = self.mlp(x.amax(dim=[2, 3], keepdim=True))
        return x * (avg_out + max_out).sigmoid()


class SpatialAttention(nn.Module):
    """空间注意力：沿通道维度取 Avg/Max → 7x7 Conv → Sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1)).sigmoid()


class CBAM(nn.Module):
    """
    CBAM 注意力模块（通道 → 空间，串行）。
    延迟初始化：ultralytics 对自定义模块不自动应用 width_multiplier，
    因此 CBAM 在首次 forward 时根据实际输入通道数动态构建子模块，
    避免 YAML 中写死的未缩放通道数与实际输入不匹配的问题。
    yaml 格式: [-1, 1, CBAM, []]  （args 被忽略，通道由输入决定）
    """
    def __init__(self, c1=0, c2=0, *args, **kwargs):
        super().__init__()
        # 延迟初始化
        self.ca = None
        self.sa = None
        self._built = False

        self.reduction = args[0] if args and isinstance(args[0], int) else \
            kwargs.get('reduction', 16)
        self.kernel_size = args[1] if len(args) > 1 and isinstance(args[1], int) else \
            kwargs.get('kernel_size', 7)

    def _build(self, channels):
        """根据实际输入通道数构建子模块"""
        self.ca = ChannelAttention(channels, self.reduction)
        self.sa = SpatialAttention(self.kernel_size)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.size(1))
        x = self.ca(x)
        x = self.sa(x)
        return x