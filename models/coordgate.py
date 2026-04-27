"""坐标门控（CoordGate）相关模块。

该文件提供两层能力：
1) `build_polar_coords`：根据特征图尺寸生成归一化极坐标 `(r, theta)`；
2) `CoordGateNAFBlock`：在 NAFBlock 的深度卷积主支中注入坐标门控，
    让特征对“图像位置”具备显式条件化能力。

与传统仅依赖卷积感受野的位置编码不同，这里的门控直接从坐标映射到通道
权重，适合镜头像差/离焦这类与视场位置强相关的退化建模。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .nafblock import LayerNorm2d, SimpleGate, SimplifiedChannelAttention


def build_polar_coords(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构建归一化极坐标图。

    输出形状为 `[B, 2, H, W]`，两个通道依次为：
    - `r`：相对半径，归一化到 [0, 1]；
    - `theta`：角度，映射到 [0, 1]（由 atan2 线性换算）。

    该坐标图通常作为 CoordGate 的输入，用于生成位置相关的通道门控。
    """

    y = torch.linspace(0.0, float(height - 1), height, device=device, dtype=dtype)
    x = torch.linspace(0.0, float(width - 1), width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    dy = grid_y - center_y
    dx = grid_x - center_x
    # 以图像中心到角点的距离作为归一化半径上界。
    max_radius = math.sqrt(max(center_x, 1e-6) ** 2 + max(center_y, 1e-6) ** 2)
    radius = torch.sqrt(dx.square() + dy.square()) / max(max_radius, 1e-6)
    radius = radius.clamp(0.0, 1.0)
    theta = (torch.atan2(dy, dx) + math.pi) / (2.0 * math.pi)
    coords = torch.stack([radius, theta], dim=0).unsqueeze(0)
    return coords.expand(batch_size, -1, -1, -1).contiguous()


class CoordGate(nn.Module):
    """坐标到通道门控权重的映射器。

    本质是一个 1x1 Conv MLP：`coords -> gate`，并通过 Sigmoid 约束到 (0, 1)。
    """

    def __init__(self, channels: int, coord_dim: int = 2, mlp_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(coord_dim, mlp_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def compute_gate(self, coords: torch.Tensor) -> torch.Tensor:
        """根据坐标图计算门控图，输出形状 `[B, C, H, W]`。"""

        return self.net(coords)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """按元素应用门控：`x * gate(coords)`。"""

        return x * self.compute_gate(coords)


class CoordGateNAFBlock(nn.Module):
    """带坐标门控的 NAFBlock 变体。

    结构与标准 NAFBlock 接近，但在深度卷积分支后增加了 CoordGate，
    使特征变换可显式依赖空间位置（极坐标）。

    与 NAFBlock 一样，使用两个可学习残差缩放参数 `beta/gamma` 来稳定训练初期。
    """

    def __init__(
        self,
        channels: int,
        coordgate_mlp_hidden: int = 32,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        dw_channels = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.coord_gate = CoordGate(dw_channels, coord_dim=2, mlp_hidden=coordgate_mlp_hidden)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def _forward_impl(self, inp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """前向主逻辑。

        分两段：
        1) 局部混合段：Norm -> 1x1 -> DWConv -> CoordGate -> SimpleGate -> SCA -> 1x1；
        2) FFN 段：Norm -> 1x1 -> SimpleGate -> 1x1。
        """

        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.coord_gate(x, coords)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.ffn1(self.norm2(y))
        x = self.sg(x)
        x = self.ffn2(x)
        x = self.dropout2(x)
        return y + x * self.gamma

    def forward(self, inp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """支持可选激活检查点（checkpoint）以节省显存。"""

        if self.use_checkpoint and self.training and inp.requires_grad:
            return checkpoint(self._forward_impl, inp, coords, use_reentrant=False) # type: ignore
        return self._forward_impl(inp, coords)
