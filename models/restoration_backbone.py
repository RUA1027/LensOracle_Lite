"""CoordGateNAFNet restoration 主干与 lens-table cross-attention 注入。

本文件保留原有 CoordGateNAFBlock 的位置感知能力，但完全移除旧式像素先验注入链路。外部物理先验通过 LensTableEncoder 产生
的 F1/F2/F3 多尺度特征，以 CrossAttentionRouter 形式注入。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .coordgate import CoordGateNAFBlock
from .cross_attention_router import CrossAttentionRouter
from utils.coord_utils import compute_polar_coord_map


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """把输入图像 padding 到 encoder 下采样所需倍数。"""

    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)


class CoordGateNAFNetRestoration(nn.Module):
    """通过 cross-attention 使用 lens-table 多尺度特征的恢复网络。

    设计核心：
    - 主干沿用 CoordGateNAF 结构，保持位置条件建模能力；
    - 通过 F1/F2/F3 三尺度先验，在编码与解码关键位置注入光学信息；
    - 可选浅层注入（use_shallow_attention）用于更强细节调制。
    """

    def __init__(
        self,
        encoder_channels: list[int],
        encoder_blocks: list[int],
        decoder_blocks: list[int],
        coordgate_mlp_hidden: int = 32,
        cross_attention_num_heads: int = 4,
        cross_attention_head_dim: int = 64,
        cross_attention_fourier_freqs: int = 8,
        lens_table_channels: list[int] | None = None,
        use_shallow_attention: bool = True,
        use_lens_attention: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain 4 stages")
        if len(encoder_blocks) != 4:
            raise ValueError("encoder_blocks must contain 4 entries")
        if len(decoder_blocks) != 3:
            raise ValueError("decoder_blocks must contain 3 entries")

        self.encoder_channels = list(encoder_channels)
        self.encoder_blocks = list(encoder_blocks)
        self.decoder_blocks = list(decoder_blocks)
        self.coordgate_mlp_hidden = int(coordgate_mlp_hidden)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_lens_attention = bool(use_lens_attention)
        self.use_shallow_attention = bool(use_shallow_attention)
        self.lens_table_channels = list(lens_table_channels or [128, 192, 256])

        self.input_proj = nn.Conv2d(3, self.encoder_channels[0], kernel_size=3, padding=1)
        self.encoder_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CoordGateNAFBlock(
                            channels=channels,
                            coordgate_mlp_hidden=coordgate_mlp_hidden,
                            use_checkpoint=self.use_checkpoint,
                        )
                        for _ in range(blocks)
                    ]
                )
                for channels, blocks in zip(self.encoder_channels, self.encoder_blocks)
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

        # Cross-attention 路由器在编码中后段、瓶颈以及解码阶段分布式注入。
        router_kwargs = {
            "num_heads": cross_attention_num_heads,
            "head_dim": cross_attention_head_dim,
            "fourier_feat_num_freqs": cross_attention_fourier_freqs,
        }
        self.encoder_router_f1 = (
            CrossAttentionRouter(self.encoder_channels[0], self.lens_table_channels[0], **router_kwargs)
            if self.use_lens_attention and self.use_shallow_attention
            else None
        )
        self.encoder_router_f2 = (
            CrossAttentionRouter(
                self.encoder_channels[2],
                self.lens_table_channels[1],
                **router_kwargs,
            )
            if self.use_lens_attention
            else None
        )
        self.bottleneck_router_f3 = (
            CrossAttentionRouter(
                self.encoder_channels[3],
                self.lens_table_channels[2],
                **router_kwargs,
            )
            if self.use_lens_attention
            else None
        )

        skip_channels = self.encoder_channels[:-1][::-1]
        decoder_in_channels = [self.encoder_channels[-1]] + skip_channels[:-1]
        decoder_out_channels = skip_channels
        self.up_projs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                for in_channels, out_channels in zip(decoder_in_channels, decoder_out_channels)
            ]
        )
        self.fuse_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
                for out_channels in decoder_out_channels
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CoordGateNAFBlock(
                            channels=channels,
                            coordgate_mlp_hidden=coordgate_mlp_hidden,
                            use_checkpoint=self.use_checkpoint,
                        )
                        for _ in range(blocks)
                    ]
                )
                for channels, blocks in zip(decoder_out_channels, self.decoder_blocks)
            ]
        )
        self.decoder_router_f3 = (
            CrossAttentionRouter(
                decoder_out_channels[0],
                self.lens_table_channels[2],
                **router_kwargs,
            )
            if self.use_lens_attention
            else None
        )
        self.decoder_router_f2 = (
            CrossAttentionRouter(
                decoder_out_channels[1],
                self.lens_table_channels[1],
                **router_kwargs,
            )
            if self.use_lens_attention
            else None
        )
        self.decoder_router_f1 = (
            CrossAttentionRouter(decoder_out_channels[2], self.lens_table_channels[0], **router_kwargs)
            if self.use_lens_attention and self.use_shallow_attention
            else None
        )
        self.output_head = nn.Conv2d(decoder_out_channels[-1], 3, kernel_size=3, padding=1)

    def _coords(
        self,
        x: torch.Tensor,
        image_hw: tuple[int, int],
        crop_info: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """为当前特征尺度生成 `(r,theta)` 查询坐标图。"""

        coord = compute_polar_coord_map(
            x.shape[-2],
            x.shape[-1],
            image_hw[0],
            image_hw[1],
            crop_info=crop_info,
            device=x.device,
            dtype=x.dtype,
        )
        if coord.shape[0] == 1 and x.shape[0] != 1:
            coord = coord.expand(x.shape[0], -1, -1, -1)
        return coord

    def _run_stage(self, blocks: nn.ModuleList, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """按 CoordGateNAFBlock 顺序处理一个 encoder/decoder stage。"""

        for block in blocks:
            x = block(x, coords)
        return x

    def _route(
        self,
        router: Optional[CrossAttentionRouter],
        x: torch.Tensor,
        prior_feat: Optional[torch.Tensor],
        coords: torch.Tensor,
        attn_name: str,
        attn_cache: Dict[str, torch.Tensor],
        return_attn: bool,
    ) -> torch.Tensor:
        """执行一次可选路由。

        当 `router is None` 时直接透传；否则根据 `return_attn` 决定是否缓存权重。
        """

        if router is None or prior_feat is None:
            return x
        if return_attn:
            x, attn = router(x, prior_feat, coords, return_attn=True)
            attn_cache[attn_name] = attn.detach()
            return x
        return router(x, prior_feat, coords, return_attn=False)

    def forward(
        self,
        blur: torch.Tensor,
        lens_features: Optional[Dict[str, torch.Tensor]] = None,
        crop_info: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """恢复网络前向。

        输入：
        - `blur`: `[B,3,H,W]`；
        - `lens_features`: 包含 `F_1/F_2/F_3` 的字典；
        - `crop_info`: 可选裁剪信息，用于把查询坐标准确映射回原图物理坐标。

        输出：
        - 默认返回 restored 图像；
        - 若 `return_attn=True`，额外返回多处路由权重缓存。
        """

        original_height, original_width = blur.shape[-2:]
        x, _ = _pad_to_multiple(blur, multiple=2 ** (len(self.encoder_channels) - 1))
        image_hw = (original_height, original_width)
        if crop_info is not None:
            crop_info = crop_info.to(device=x.device, dtype=x.dtype)
        if not self.use_lens_attention or lens_features is None:
            lens_features = {}

        attn_cache: Dict[str, torch.Tensor] = {}
        x = self.input_proj(x)
        skips: list[torch.Tensor] = []

        # Encoder：逐层提特征，并在指定层位注入 F1/F2/F3 先验。
        for idx, stage in enumerate(self.encoder_stages):
            coords = self._coords(x, image_hw=image_hw, crop_info=crop_info)
            x = self._run_stage(stage, x, coords)
            if idx == 0 and self.use_shallow_attention:
                x = self._route(
                    self.encoder_router_f1,
                    x,
                    lens_features.get("F_1"),
                    coords,
                    "encoder_F1",
                    attn_cache,
                    return_attn,
                )
            if idx == 2:
                x = self._route(
                    self.encoder_router_f2,
                    x,
                    lens_features.get("F_2"),
                    coords,
                    "encoder_F2",
                    attn_cache,
                    return_attn,
                )
            if idx == 3:
                x = self._route(
                    self.bottleneck_router_f3,
                    x,
                    lens_features.get("F_3"),
                    coords,
                    "bottleneck_F3",
                    attn_cache,
                    return_attn,
                )
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        # Decoder：逐层上采样，融合 skip，并在对应层注入 F3/F2/(可选 F1)。
        decoder_skips = skips[:-1][::-1]
        for idx, (up_proj, fuse, stage, skip) in enumerate(
            zip(self.up_projs, self.fuse_convs, self.decoder_stages, decoder_skips)
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up_proj(x)
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            coords = self._coords(x, image_hw=image_hw, crop_info=crop_info)
            if idx == 0:
                x = self._route(
                    self.decoder_router_f3,
                    x,
                    lens_features.get("F_3"),
                    coords,
                    "decoder_F3",
                    attn_cache,
                    return_attn,
                )
            elif idx == 1:
                x = self._route(
                    self.decoder_router_f2,
                    x,
                    lens_features.get("F_2"),
                    coords,
                    "decoder_F2",
                    attn_cache,
                    return_attn,
                )
            elif idx == 2 and self.use_shallow_attention:
                x = self._route(
                    self.decoder_router_f1,
                    x,
                    lens_features.get("F_1"),
                    coords,
                    "decoder_F1",
                    attn_cache,
                    return_attn,
                )
            x = self._run_stage(stage, x, coords)

        # 输出采用 residual 形式，与输入 blur 相加后再裁回原始分辨率。
        restored = self.output_head(x) + blur[:, :, : x.shape[-2], : x.shape[-1]]
        restored = restored[:, :, :original_height, :original_width]
        if return_attn:
            return restored, attn_cache
        return restored
