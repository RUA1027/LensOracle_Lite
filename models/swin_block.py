"""Swin Transformer 基础组件（轻量实现）。

该文件提供 PriorEstimator bottleneck 所需的窗口注意力模块：
- WindowAttention：窗口内多头自注意力 + 相对位置偏置；
- SwinTransformerBlock：支持 shift window 的局部 Transformer block；
- RSTB：Residual Swin Transformer Block，外包一个卷积形成残差单元。

实现目标：在保持局部建模效率的同时，通过移位窗口让跨窗口信息逐层交换。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    """把标量统一为二维 tuple。"""

    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    """Stochastic Depth（按样本随机丢弃残差分支）。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Transformer 常用两层前馈网络。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """把 `[B, H, W, C]` 切分为窗口块。

    返回 `[num_windows*B, window, window, C]`。
    """

    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, channels
    )


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """把窗口块还原回 `[B, H, W, C]`。"""

    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    """窗口内多头自注意力（含相对位置偏置）。"""

    def __init__(
        self,
        dim: int,
        window_size: int | Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = to_2tuple(window_size)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )

        # 1. 生成窗口内在高度和宽度方向的绝对坐标
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # 构建二维网格坐标
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        
        # 2. 展平以便计算两两相对距离，形状变为 [2, N] (N = window_size_h * window_size_w)
        coords_flatten = torch.flatten(coords, 1)
        
        # 3. 广播相减，计算任意两点间的相对坐标差，得到 [2, N, N]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        
        # 4. 调整维度顺序为 [N, N, 2]，更符合直觉：(查询点, 键点, [Δh, Δw])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # 5. 将相对坐标平移至非负区间，避免作为索引时出现负数
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        
        # 6. 为高度维度的相对位移乘上步长权重，将二维坐标唯一编码为一维索引
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        
        # 7. 注册为 buffer，随模型保存和设备转移，但不更新梯度
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行窗口注意力。

        参数：
        - `x`: `[B*nW, N, C]`，N 为窗口 token 数；
        - `mask`: shift window 时的注意力掩码。
        """

        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """单个 Swin Block（支持 shift window）。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.dim)
        hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(self.dim, hidden_features=hidden_dim, drop=drop)

    def _build_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """构建 shift window 的 attention mask。

        不同窗口来源的 token 之间赋予大负值，阻止错误跨窗口注意力连接。
        """

        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """执行一层 Swin block 前向。"""

        height, width = x_size
        batch, tokens, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1)
        padded_h, padded_w = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._build_mask(padded_h, padded_w, x.device)
        else:
            shifted = x
            attn_mask = None

        x_windows = window_partition(shifted, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attn_windows, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        if pad_h > 0 or pad_w > 0:
            x = x[:, :height, :width, :]
        x = x.reshape(batch, height * width, channels)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block。

    结构：若干 SwinTransformerBlock 串联 + 3x3 Conv，最后与输入做残差相加。
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 2,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        blocks = []
        for idx in range(int(num_blocks)):
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if idx % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入输出均为 `[B, C, H, W]`。"""

        batch, channels, height, width = x.shape
        residual = x
        seq = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            seq = block(seq, (height, width))
        seq = seq.transpose(1, 2).reshape(batch, channels, height, width)
        seq = self.conv(seq)
        return residual + seq
