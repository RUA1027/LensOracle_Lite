"""Optical Degradation Network，用于 Stage 2 的物理闭环验证。

ODN 接收 sharp 图像和原始 `pred_table`，尝试重建真实 blur。它故意保持
较小容量，避免只靠图像卷积主干绕过 lens-table 信息。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.cross_attention_router import CrossAttentionRouter
from models.nafblock import NAFBlock
from utils.coord_utils import compute_polar_coord_map


class _ODNStage(nn.Module):
    """ODN 编码器中的 stride-2 降采样块。

    结构：`Conv(stride=2) + NAFBlock x N`。
    作用是在压缩空间分辨率的同时保留局部结构表达。
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(out_channels) for _ in range(max(1, int(num_blocks)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class _ODNUpStage(nn.Module):
    """ODN 解码器中的上采样块。

    结构：`ConvTranspose2d + NAFBlock x N`，用于逐步恢复空间分辨率。
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(out_channels) for _ in range(max(1, int(num_blocks)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.up(x))


class OpticalDegradationNetwork(nn.Module):
    """轻量 encoder-modulator-decoder 退化模拟器。

    设计意图：
    - 输入 sharp 与 `pred_table`，预测由该光学先验诱导的 blur；
    - 通过容量受限主干 + 一次关键 cross-attention，抑制“绕开先验”的捷径。
    """

    def __init__(
        self,
        base_channels: int = 32,
        bottleneck_channels: int = 96,
        num_heads: int = 4,
        head_dim: int = 32,
        num_blocks: int = 1,
        fourier_feat_num_freqs: int = 8,
    ):
        super().__init__()
        base_channels = int(base_channels)
        bottleneck_channels = int(bottleneck_channels)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = bottleneck_channels

        self.input_proj = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.enc1 = _ODNStage(c1, c1, num_blocks)
        self.enc2 = _ODNStage(c1, c2, num_blocks)
        self.enc3 = _ODNStage(c2, c3, num_blocks)
        self.table_router = CrossAttentionRouter(
            feat_channels=c3,
            prior_channels=67,
            num_heads=num_heads,
            head_dim=head_dim,
            fourier_feat_num_freqs=fourier_feat_num_freqs,
        )
        self.dec3 = _ODNUpStage(c3, c2, num_blocks)
        self.dec2 = _ODNUpStage(c2, c1, num_blocks)
        self.dec1 = _ODNUpStage(c1, c1, num_blocks)
        self.out = nn.Conv2d(c1, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias) # type: ignore

    def forward(self, sharp: torch.Tensor, pred_table: torch.Tensor) -> torch.Tensor:
        """执行退化模拟前向。

        关键步骤：
        1) 编码 sharp 到低分辨率瓶颈特征；
        2) 将 `pred_table` 转成 token 特征并按极坐标路由注入；
        3) 解码回图像分辨率并预测 residual blur；
        4) 输出 `sharp + residual` 作为模拟 blur。
        """

        if pred_table.ndim != 4 or pred_table.shape[1:] != (64, 48, 67):
            raise ValueError(f"Expected pred_table shape [B,64,48,67], got {tuple(pred_table.shape)}")

        h_img, w_img = sharp.shape[-2:]
        x = self.input_proj(sharp)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        # [B, r, theta, c] -> [B, c, theta, r]，对齐 CrossAttentionRouter 的 prior 输入约定。
        prior_feat = pred_table.permute(0, 3, 2, 1).contiguous()
        coord_map = compute_polar_coord_map(
            x.shape[-2],
            x.shape[-1],
            h_img,
            w_img,
            crop_info=None,
            device=x.device,
            dtype=x.dtype,
        ).expand(x.shape[0], -1, -1, -1)
        # 在瓶颈尺度注入镜头先验，提升“sharp->blur”映射的物理一致性。
        x = self.table_router(x, prior_feat, coord_map)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        if x.shape[-2:] != sharp.shape[-2:]:
            x = F.interpolate(x, size=sharp.shape[-2:], mode="bilinear", align_corners=False)
        # 输出 residual 叠加到 sharp，形成最终模拟 blur。
        return sharp + self.out(x)
