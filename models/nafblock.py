"""NAF 系列基础模块。

该文件实现 NAFNet 风格的基础积木：
- `LayerNorm2d`：适配 NCHW 的 LayerNorm；
- `SimpleGate`：通道二分后逐元素相乘；
- `SimplifiedChannelAttention`：极简通道注意力；
- `NAFBlock`：主干局部混合 + FFN 的双残差结构。

其中 `beta/gamma` 初始化为 0，能够让网络在训练初期更接近恒等映射，
提升深层堆叠时的稳定性。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    """NCHW 张量上的 LayerNorm。

    实现方式：临时转换为 NHWC 调用 `F.layer_norm`，再转回 NCHW。
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """NAF 的门控算子：`chunk(x,2)` 后做 Hadamard 乘积。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    # 在 第 1 维度（通道 / 特征维度） 上，把张量 x 切成大小相等的 2 份
    # 得到两个形状完全一样的张量：x1 和 x2
    # 不会改变张量的长宽，只切通道 / 特征

class SimplifiedChannelAttention(nn.Module):
    """极简通道注意力。

    通过全局平均池化提取每个通道的全局响应，再经 1x1 Conv 生成缩放因子。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class NAFBlock(nn.Module):
    """NAFBlock 主体实现。

    计算路径：
    1) 局部混合分支：LN -> 1x1 -> DWConv -> SimpleGate -> SCA -> 1x1；
    2) FFN 分支：LN -> 1x1 -> SimpleGate -> 1x1；
    3) 两次残差注入，并分别由 `beta/gamma` 缩放。
    """

    def __init__(
        self,
        channels: int,
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

    def _forward_impl(self, inp: torch.Tensor) -> torch.Tensor:
        """不含 checkpoint 分支的核心前向。"""

        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dwconv(x)
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

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """根据配置决定是否启用激活检查点。"""

        if self.use_checkpoint and self.training and inp.requires_grad:
            return checkpoint(self._forward_impl, inp, use_reentrant=False) # type: ignore
        return self._forward_impl(inp)
