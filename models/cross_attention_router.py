"""基于物理坐标的 lens-table cross-attention 路由模块。

该模块负责把 lens-table encoder 的 token 按 `(r, theta)` 位置路由到
restoration 特征图的对应位置。默认只返回融合后的特征；训练诊断需要时
可通过 `return_attn=True` 暴露 attention 权重。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


def fourier_encode_coords(coords: torch.Tensor, num_freqs: int = 8) -> torch.Tensor:
    """对 `[... , 2]` 坐标做 Fourier feature 编码。

    输入最后一维固定为 `(r, theta)`，输出最后一维为 `4*num_freqs`：
    - `r` 的 sin/cos 多频率特征；
    - `theta` 的 sin/cos 多频率特征。

    这样可以把低维连续坐标映射到更适合注意力线性层建模的高维周期空间。
    """

    if coords.shape[-1] != 2:
        raise ValueError(f"Expected coord last dim=2, got {tuple(coords.shape)}")
    freqs = torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)
    freqs = (2.0 ** freqs) * math.pi
    expanded = coords.unsqueeze(-1) * freqs
    encoded = torch.cat([torch.sin(expanded), torch.cos(expanded)], dim=-1)
    return encoded.flatten(start_dim=-2)


def build_lens_token_coords(
    theta_size: int,
    radial_size: int,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """生成 lens-table token 的 `[B, theta*r, 2]` 物理坐标。"""

    theta = torch.arange(theta_size, device=device, dtype=dtype) / max(theta_size, 1)
    radius = torch.linspace(0.0, 1.0, radial_size, device=device, dtype=dtype)
    grid_theta, grid_radius = torch.meshgrid(theta, radius, indexing="ij")
    coords = torch.stack([grid_radius, grid_theta], dim=-1).reshape(1, theta_size * radial_size, 2)
    return coords.expand(batch_size, -1, -1).contiguous()


class CrossAttentionRouter(nn.Module):
    """将多尺度 lens-table 特征注入 restoration 特征图。"""

    def __init__(
        self,
        feat_channels: int,
        prior_channels: int,
        num_heads: int = 4,
        head_dim: int = 64,
        fourier_feat_num_freqs: int = 8,
    ):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.prior_channels = int(prior_channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.fourier_feat_num_freqs = int(fourier_feat_num_freqs)
        self.pos_dim = 4 * self.fourier_feat_num_freqs
        inner_dim = self.num_heads * self.head_dim

        self.to_q = nn.Linear(self.feat_channels + self.pos_dim, inner_dim)
        self.to_k = nn.Linear(self.prior_channels + self.pos_dim, inner_dim)
        self.to_v = nn.Linear(self.prior_channels + self.pos_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, self.feat_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 `[B,N,H*D]` 重排为 `[B,H,N,D]`，便于多头注意力计算。"""

        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        coord_map: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """执行跨注意力路由。

        参数约定：
        - `x`: 查询特征（通常来自 restoration/ODN 主干），形状 `[B,C,H,W]`；
        - `prior_feat`: 键值特征（来自 lens-table 编码器），形状 `[B,Cp,Hp,Wp]`；
        - `coord_map`: 查询位置的 `(r,theta)` 坐标图，形状 `[B,2,H,W]`。

        返回：
        - 默认返回融合后特征 `x + routed`；
        - `return_attn=True` 时额外返回 attention 权重，便于可视化诊断。
        """

        if x.ndim != 4:
            raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
        if prior_feat.ndim != 4:
            raise ValueError(f"Expected prior_feat shape [B,C,H,W], got {tuple(prior_feat.shape)}")
        if coord_map.ndim != 4 or coord_map.shape[1] != 2:
            raise ValueError(f"Expected coord_map shape [B,2,H,W], got {tuple(coord_map.shape)}")

        b, c, h, w = x.shape
        _, c_prior, hp, wp = prior_feat.shape
        if c != self.feat_channels:
            raise ValueError(f"Router feat_channels={self.feat_channels}, got {c}")
        if c_prior != self.prior_channels:
            raise ValueError(f"Router prior_channels={self.prior_channels}, got {c_prior}")

        # 1) 先把先验特征 token 化，再拼接先验 token 自身的物理坐标编码。
        
        # 把 4D 特征图 [B,C,H,W] 展平成 token 序列 [B,N,C]
        # 给每个 token 加上它的物理坐标编码（傅里叶）

        # 把图像特征图变成Transformer 能吃的序列 tokens
        # flatten(2)：把 [B, C, H, W] 特征图 → 展平空间维度 → [B, C, H×W]
        # transpose(1, 2)：交换通道和序列维度 → [B, H×W, C]（Transformer 标准输入格式：批量 × 序列长度 × 特征维度）
        # contiguous()：让张量内存连续，避免后续运算报错
        prior_tokens = prior_feat.flatten(2).transpose(1, 2).contiguous()
        
        # 给每个 token 生成原始坐标位置，为上面 H×W 个 token 生成对应的 (x, y) 坐标
        prior_coords = build_lens_token_coords(hp, wp, b, prior_feat.device, prior_feat.dtype)
        
        # 对坐标做傅里叶位置编码，得到位置嵌入 prior_pos，维度为 [B, H×W, pos_dim]，其中 pos_dim = 4 * num_freqs
        prior_pos = fourier_encode_coords(prior_coords, self.fourier_feat_num_freqs)
        # 特征 + 位置编码 → 让注意力知道像素在哪
        prior_tokens = torch.cat([prior_tokens, prior_pos], dim=-1)

        # 2) 同理处理查询特征 token 与查询坐标编码。
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()
        query_coords = coord_map.flatten(2).transpose(1, 2).contiguous().to(dtype=x.dtype)
        query_pos = fourier_encode_coords(query_coords, self.fourier_feat_num_freqs)
        # 特征 + 位置编码 → 让注意力知道像素在哪
        query_tokens = torch.cat([x_tokens, query_pos], dim=-1)

        # 3) 线性映射到 Q/K/V，并切分为多头。
        q = self._reshape_heads(self.to_q(query_tokens))
        k = self._reshape_heads(self.to_k(prior_tokens))
        v = self._reshape_heads(self.to_v(prior_tokens))

        # 4) 标准缩放点积注意力。
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(attn, dim=-1)

        # 5) 注意力加权取值并恢复回特征图布局。
        routed = torch.matmul(attn, v)
        routed = routed.permute(0, 2, 1, 3).contiguous().view(b, h * w, self.num_heads * self.head_dim)
        routed = self.out_proj(routed).transpose(1, 2).contiguous().view(b, c, h, w)

        # 6) 采用残差融合，保留主干原始表示并叠加先验注入结果。
        output = x + routed
        if return_attn:
            return output, attn
        return output
