"""Lens-table 原生空间的多尺度特征提取模块。

本文件只处理 `[B, 64, 48, 67]` 的物理表格先验：
- 64 表示归一化视场半径 r；
- 48 表示方位角 theta；
- 67 表示 SFR 特征通道。

网络内部把表格变成 `[B, 67, 48, 64]`，其中 height 是 theta、
width 是 r。theta 方向使用 circular padding，r 方向使用普通 zero
padding，以保留角度周期性且不错误地把半径两端连起来。
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    """把卷积参数规范成二维 tuple，便于后续手动 padding 与 stride 对齐。"""

    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Expected a 2D convolution parameter")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class CircularConv2d(nn.Module):
    """只在 theta 维做 circular padding 的二维卷积。

    输入张量约定为 `[B, C, theta, r]`。

    为什么不用原生 Conv2d padding：
    - theta 维是角度，天然周期，应采用 circular 连接首尾；
    - r 维是半径，边界不应循环，应采用 zero padding。

    因此实现上先手动 circular 拼接 theta，再仅对 r 做零填充，最后执行
    `padding=0` 卷积。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size)
        stride_h, stride_w = _pair(stride)
        self.pad_h = kernel_h // 2
        self.pad_w = kernel_w // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=0,
            bias=bias,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_h > 0:
            x = torch.cat([x[:, :, -self.pad_h :, :], x, x[:, :, : self.pad_h, :]], dim=2)
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="constant", value=0.0)
        return self.conv(x)


class ZeroPaddedConv2d(nn.Module):
    """Drop-in Conv2d wrapper matching CircularConv2d's state_dict layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size)
        stride_h, stride_w = _pair(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(kernel_h // 2, kernel_w // 2),
            bias=bias,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LensTableResidualBlock(nn.Module):
    """Lens-table 特征空间的轻量残差块。

    由两层 CircularConv2d 与 GELU 组成，保持角向周期边界条件。
    """

    def __init__(self, channels: int, conv_layer=CircularConv2d):
        super().__init__()
        self.net = nn.Sequential(
            conv_layer(channels, channels, kernel_size=3),
            nn.GELU(),
            conv_layer(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LensTableEncoder(nn.Module):
    """把原生 lens-table 编码为三层多尺度物理特征。

    输入约定：`[B, 64, 48, 67]`（r, theta, channel）。
    内部布局：`[B, 67, 48, 64]`（channel, theta, r）。
    输出字典：
    - `F_1`: 高分辨率特征；
    - `F_2`: 下采样一层；
    - `F_3`: 下采样两层。

    与 PriorEstimator 的关系：
    - PriorEstimator 的输出是 `[B, 64, 48, 67]` 物理参数表；
    - 本模块（LensTableEncoder）负责接收这个表，提取多尺度物理特征；
    - 这些特征最后会喂给 Restoration 网络，通过 CrossAttention 注入主干。
    """

    def __init__(
        self,
        in_channels: int = 67,
        channels: Sequence[int] = (128, 192, 256),
        blocks_per_level: Sequence[int] = (2, 2, 3),
        padding_mode: str = "circular",
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("channels must contain three levels: F_1, F_2, F_3")
        if len(blocks_per_level) != 3:
            raise ValueError("blocks_per_level must contain three entries")

        c1, c2, c3 = [int(value) for value in channels]
        b1, b2, b3 = [int(value) for value in blocks_per_level]
        self.channels = [c1, c2, c3]
        self.blocks_per_level = [b1, b2, b3]
        self.padding_mode = str(padding_mode)
        if self.padding_mode == "circular":
            conv_layer = CircularConv2d
        elif self.padding_mode == "zero":
            conv_layer = ZeroPaddedConv2d
        else:
            raise ValueError("padding_mode must be 'circular' or 'zero'")

        self.input_proj = conv_layer(in_channels, c1, kernel_size=3)
        self.level1 = nn.Sequential(*[LensTableResidualBlock(c1, conv_layer=conv_layer) for _ in range(b1)])

        self.down1 = conv_layer(c1, c2, kernel_size=3, stride=2)
        self.level2 = nn.Sequential(*[LensTableResidualBlock(c2, conv_layer=conv_layer) for _ in range(b2)])

        self.down2 = conv_layer(c2, c3, kernel_size=3, stride=2)
        self.level3 = nn.Sequential(*[LensTableResidualBlock(c3, conv_layer=conv_layer) for _ in range(b3)])

    def forward(self, pred_table: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行 lens-table 编码并返回多尺度特征。"""

        if pred_table.ndim != 4 or pred_table.shape[1:] != (64, 48, 67):
            raise ValueError(
                "Expected pred_table shape [B, 64, 48, 67], "
                f"got {tuple(pred_table.shape)}"
            )

        # [B, r, theta, c] -> [B, c, theta, r]。
        x = pred_table.permute(0, 3, 2, 1).contiguous()
        f1 = self.level1(self.input_proj(x))
        f2 = self.level2(self.down1(f1))
        f3 = self.level3(self.down2(f2))
        return {"F_1": f1, "F_2": f2, "F_3": f3}
