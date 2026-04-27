"""BlindPriorEstimator：从 blur 图像估计原生 lens-table。

新版 prior estimator 不再输出像素级 prior map，而是直接预测
`[B, 64, 48, 67]` 的 PSF-SFR lens-table。监督信号因此保持在 ray-tracing
产生的原生物理空间里，避免像素展开和反投影带来的插值伪迹。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .lens_table_encoder import CircularConv2d
from .nafblock import NAFBlock
from .swin_block import RSTB


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """把输入 padding 到指定倍数。

    目的：保证多次 stride=2 下采样后不会出现奇偶尺寸不对齐问题。
    返回 `(padded_x, (pad_h, pad_w))`，便于必要时回裁。
    """

    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)


class _LensTableDecoderBlock(nn.Module):
    """lens-table 解码器上采样块。

    由反卷积放大分辨率，再用 CircularConv2d 精炼特征，保证 theta 维的
    周期边界处理与 lens-table 物理语义一致。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            CircularConv2d(out_channels, out_channels, kernel_size=3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BlindPriorEstimator(nn.Module):
    """图像 encoder + lens-table decoder 的盲先验估计器。

    输入：blur 图像 `[B,3,H,W]`。
    输出：原生 lens-table `[B,64,48,67]`。

    架构分为三段：
    1) 图像编码器（NAF blocks + 下采样）提取高层退化表征；
    2) RSTB bottleneck 增强全局建模；
    3) 以 latent 为条件的解码器重建固定分辨率 lens-table。
    """

    def __init__(
        self,
        encoder_channels: list[int],
        blocks_per_level: int = 2,
        rstb_num_blocks: int = 2,
        rstb_window_size: int = 8,
        rstb_num_heads: int = 8,
        d_latent: int = 512,
        decoder_seed_channels: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain 4 stages")

        self.encoder_channels = list(encoder_channels)
        self.blocks_per_level = int(blocks_per_level)
        self.d_latent = int(d_latent)
        self.decoder_seed_channels = int(decoder_seed_channels)
        self.use_checkpoint = bool(use_checkpoint)

        self.input_proj = nn.Conv2d(3, self.encoder_channels[0], kernel_size=3, padding=1)
        self.encoder_stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        NAFBlock(channels=channels, use_checkpoint=self.use_checkpoint)
                        for _ in range(self.blocks_per_level)
                    ]
                )
                for channels in self.encoder_channels
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(
                    self.encoder_channels[idx],
                    self.encoder_channels[idx + 1],
                    kernel_size=2,
                    stride=2,
                )
                for idx in range(len(self.encoder_channels) - 1)
            ]
        )
        # Bottleneck 用 RSTB 做局部窗口注意力与跨窗口信息交换。
        self.bottleneck = nn.Sequential(
            *[
                RSTB(
                    dim=self.encoder_channels[-1],
                    num_blocks=2,
                    num_heads=rstb_num_heads,
                    window_size=rstb_window_size,
                )
                for _ in range(int(rstb_num_blocks))
            ]
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.latent_mlp = nn.Sequential(
            nn.Linear(self.encoder_channels[-1], self.d_latent),
            nn.GELU(),
            nn.Linear(self.d_latent, self.d_latent),
            nn.GELU(),
        )
        self.seed_mlp = nn.Sequential(
            nn.Linear(self.d_latent, self.decoder_seed_channels * 3 * 4),
            nn.GELU(),
        )

        # 通过 4 次上采样把种子分辨率 3x4 逐步恢复到 48x64。
        decoder_channels = [
            self.decoder_seed_channels,
            max(self.decoder_seed_channels // 2, 64),
            max(self.decoder_seed_channels // 4, 64),
            max(self.decoder_seed_channels // 8, 64),
        ]
        self.decoder = nn.Sequential(
            _LensTableDecoderBlock(self.decoder_seed_channels, decoder_channels[0]),
            _LensTableDecoderBlock(decoder_channels[0], decoder_channels[1]),
            _LensTableDecoderBlock(decoder_channels[1], decoder_channels[2]),
            _LensTableDecoderBlock(decoder_channels[2], decoder_channels[3]),
        )
        self.output_head = nn.Conv2d(decoder_channels[3], 67, kernel_size=1)

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        """执行 prior 估计前向。

        关键流程：
        1) 编码 blur 到深层特征；
        2) 全局池化得到 latent；
        3) latent 生成解码种子并重建 table；
        4) 输出转换为 `[B,64,48,67]`。
        """

        x, _ = _pad_to_multiple(blur, multiple=2 ** (len(self.encoder_channels) - 1))
        x = self.input_proj(x)

        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.bottleneck(x)
        # 空间信息压缩到全局向量，用于驱动固定拓扑的 table 解码。
        pooled = self.global_pool(x).flatten(1)
        latent = self.latent_mlp(pooled)
        seed = self.seed_mlp(latent).view(blur.shape[0], self.decoder_seed_channels, 3, 4)
        table_chw = self.output_head(self.decoder(seed))
        if table_chw.shape[-2:] != (48, 64):
            table_chw = F.interpolate(table_chw, size=(48, 64), mode="bilinear", align_corners=False)
        return table_chw.permute(0, 3, 2, 1).contiguous()
