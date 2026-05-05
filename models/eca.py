"""
ECA-Net (Efficient Channel Attention)
轻量级通道注意力机制，使用自适应 1D 卷积替代全连接 MLP，
参数开销远小于 CBAM/SE-Net，更适合 YOLOv8n 等 nano 模型。
兼容 ultralytics 模型构建接口，延迟初始化。
"""
import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    ECA-Net 注意力模块 (Efficient Channel Attention)。
    延迟初始化：首次 forward 时根据实际输入通道数动态构建 1D 卷积，
    自适应计算 kernel_size: k = |log2(C)/γ + b/γ| (γ=2, b=1)。
    yaml 格式: [-1, 1, ECA, []]  （args 被忽略，通道/核大小由输入决定）
    """

    def __init__(self, c1=0, c2=0, *args, **kwargs):
        super().__init__()
        # 延迟初始化
        self.conv = None
        self._built = False

        # gamma 和 bias 用于自适应 kernel_size 计算
        self.gamma = args[0] if args and isinstance(args[0], (int, float)) else \
            kwargs.get('gamma', 2)
        self.b = args[1] if len(args) > 1 and isinstance(args[1], int) else \
            kwargs.get('b', 1)

    def _build(self, channels: int):
        """根据输入通道数自适应计算 kernel_size 并构建 1D 卷积"""

        # 自适应核大小: k = |log2(C)/γ + b/γ|_odd
        t = int(abs(math.log2(channels) / self.gamma + self.b / self.gamma))
        k = t if t % 2 == 1 else t + 1

        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k,
            padding=k // 2,
            bias=False,
        )
        self._built = True

    @staticmethod
    def _eca_forward(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """ECA 前向传播核心逻辑：GAP → 1D Conv → Sigmoid → 通道加权"""
        # Global Average Pooling: (B, C, H, W) → (B, C, 1, 1)
        y = x.mean(dim=[2, 3], keepdim=True)

        # 调整形状: (B, C, 1, 1) → (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 1D 卷积分组通道交互 (共享卷积核，无通道间全连接)
        y = conv(y)

        # 恢复形状: (B, 1, C) → (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y.sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.size(1))
        return self._eca_forward(x, self.conv)