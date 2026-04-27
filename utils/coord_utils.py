"""图像特征位置到光学极坐标 `(r, theta)` 的映射工具。

该模块用于把 restoration 特征图上的网格位置，映射回原图坐标系后再
转换到光学极坐标系，为 cross-attention 查询提供物理位置信息。
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _normalize_original_size(value: Any, batch_size: int) -> list[tuple[int, int]]:
    """把 DataLoader collate 后的 original_size 规范为 `(H, W)` 列表。

兼容输入形式：
1) tensor 形态 `[B,2]`；
2) `(heights_tensor, widths_tensor)`；
3) Python list/tuple 的逐样本 `(H,W)`；
4) 单个 `(H,W)`（广播到整个 batch）。
"""

    if value is None:
        return []
    if torch.is_tensor(value):
        return [(int(item[0]), int(item[1])) for item in value.tolist()]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(torch.is_tensor(item) for item in value):
        heights = value[0].tolist()
        widths = value[1].tolist()
        return [(int(h), int(w)) for h, w in zip(heights, widths)]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return [(int(item[0]), int(item[1])) for item in value]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [(int(value[0]), int(value[1])) for _ in range(batch_size)]
    raise ValueError("Unsupported original_size format")


def compute_polar_coord_map(
    H_feat: int,
    W_feat: int,
    H_img: int,
    W_img: int,
    crop_info: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """计算 restoration 特征图每个位置对应的归一化极坐标。

    `crop_info` 采用数据集输出的归一化格式 `[top, left, crop_h, crop_w]`：
    - `top/left`: 裁剪框左上角在原图中的归一化位置；
    - `crop_h/crop_w`: 裁剪框尺寸占原图比例。

    返回形状：`[B, 2, H_feat, W_feat]`，其中通道 0 为 `r`，通道 1 为 `theta`。
    """

    device = device if device is not None else (crop_info.device if crop_info is not None else "cpu")
    dtype = dtype if dtype is not None else (crop_info.dtype if crop_info is not None else torch.float32)

    # 缺省为“整图无裁剪”场景。
    if crop_info is None:
        batch_size = 1
        crop_info = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device, dtype=dtype)
    else:
        if crop_info.ndim == 1:
            crop_info = crop_info.unsqueeze(0)
        if crop_info.ndim != 2 or crop_info.shape[1] != 4:
            raise ValueError(f"Expected crop_info shape [B,4], got {tuple(crop_info.shape)}")
        crop_info = crop_info.to(device=device, dtype=dtype)
        batch_size = int(crop_info.shape[0])

    # 以像素中心采样，减少边界偏差。
    y = torch.arange(H_feat, device=device, dtype=dtype) + 0.5
    x = torch.arange(W_feat, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)

    top = crop_info[:, 0].view(batch_size, 1, 1) * float(H_img)
    left = crop_info[:, 1].view(batch_size, 1, 1) * float(W_img)
    crop_h = crop_info[:, 2].view(batch_size, 1, 1) * float(H_img)
    crop_w = crop_info[:, 3].view(batch_size, 1, 1) * float(W_img)

    # 先把特征图坐标映射回原图坐标系。
    img_y = top + grid_y * (crop_h / max(H_feat, 1))
    img_x = left + grid_x * (crop_w / max(W_feat, 1))

    cy = float(H_img) / 2.0
    cx = float(W_img) / 2.0
    r_max = max(math.sqrt(cx * cx + cy * cy), 1.0e-6)
    # 原图笛卡尔坐标 -> 极坐标，并做 [0,1] 归一化。
    radius = torch.sqrt((img_x - cx).square() + (img_y - cy).square()) / r_max
    theta = (torch.atan2(img_y - cy, img_x - cx) + math.pi) / (2.0 * math.pi)

    coord = torch.stack([radius.clamp(0.0, 1.0), theta.clamp(0.0, 1.0)], dim=1)
    return coord.contiguous()


def normalize_original_size_batch(original_size: Any, batch_size: int) -> list[tuple[int, int]]:
    """公开给 trainer/metrics 使用的 original_size 规范化函数。

当输入为空时返回 `(0,0)` 占位，保证下游调用总能拿到等长列表。
"""

    sizes = _normalize_original_size(original_size, batch_size)
    if sizes:
        return sizes
    return [(0, 0) for _ in range(batch_size)]
