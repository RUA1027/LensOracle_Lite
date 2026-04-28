"""BPFR-Net 可视化工具函数集合。

覆盖内容包括：
1) lens-table 通道对比图与 SFR 曲线；
2) attention 权重热力图；
3) ODN 重建对比图；
4) 训练阶段关键指标仪表盘；
5) restoration 结果局部放大图；
6) 中心+边缘视场复原对比图；
7) 残差热力图；
8) 全画幅复原图保存。

所有函数均以“可离线保存 PNG”为目标，采用无界面后端 Agg。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch


def _ensure_parent(filename: str | Path) -> None:
    """确保输出文件父目录存在。"""

    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def _to_numpy_image(tensor: torch.Tensor):
    """把常见张量形状转换为可 imshow 的 numpy 图像。"""

    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image.numpy()


def _load_pyplot():
    """延迟导入 matplotlib，并切换到无 GUI 后端。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_lens_table_comparison(
    pred_table: torch.Tensor,
    gt_psf_sfr: torch.Tensor,
    filename: str,
    channels_to_show: Sequence[int] = (0, 10, 20, 30, 40, 50, 60),
) -> None:
    """绘制指定通道的 lens-table GT/Pred 对比热图。

输入 table 约定可为 `[B,64,48,67]` 或 `[64,48,67]`，函数内部会自动取
首样本并对每个通道使用统一色域范围，便于直观比较偏差。
    """

    plt = _load_pyplot()
    pred = pred_table.detach().float().cpu()
    gt = gt_psf_sfr.detach().float().cpu()
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]
    channels = [int(c) for c in channels_to_show if 0 <= int(c) < pred.shape[-1]]
    if not channels:
        channels = [0]

    fig, axes = plt.subplots(2, len(channels), figsize=(3.2 * len(channels), 6.0), squeeze=False)
    for col, channel in enumerate(channels):
        # 统一转成 (theta, r) 视图，便于观测角向与径向结构。
        gt_img = gt[:, :, channel].transpose(0, 1)
        pred_img = pred[:, :, channel].transpose(0, 1)
        vmin = float(torch.minimum(gt_img.min(), pred_img.min()))
        vmax = float(torch.maximum(gt_img.max(), pred_img.max()))
        im0 = axes[0, col].imshow(gt_img, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(f"GT c{channel}")
        im1 = axes[1, col].imshow(pred_img, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, col].set_title(f"Pred c{channel}")
        fig.colorbar(im0, ax=[axes[0, col], axes[1, col]], fraction=0.046, pad=0.04)
    for ax in axes.ravel():
        ax.set_xlabel("r")
        ax.set_ylabel("theta")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_sfr_curves(
    pred_table: torch.Tensor,
    gt_psf_sfr: torch.Tensor,
    filename: str,
    sample_positions: Sequence[tuple[int, int]] = ((0, 0), (32, 24), (63, 47)),
) -> None:
    """在若干 `(r, theta)` 位置绘制 67 维 SFR 曲线。

用于检查不同视场点位上，预测曲线与 GT 曲线的形状一致性。
    """

    plt = _load_pyplot()
    pred = pred_table.detach().float().cpu()
    gt = gt_psf_sfr.detach().float().cpu()
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]

    fig, axes = plt.subplots(len(sample_positions), 1, figsize=(8, 2.6 * len(sample_positions)), squeeze=False)
    x = torch.arange(pred.shape[-1]).numpy()
    for row, (r_idx, theta_idx) in enumerate(sample_positions):
        r = max(0, min(int(r_idx), pred.shape[0] - 1))
        theta = max(0, min(int(theta_idx), pred.shape[1] - 1))
        ax = axes[row, 0]
        ax.plot(x, gt[r, theta].numpy(), label="GT", linewidth=1.8)
        ax.plot(x, pred[r, theta].numpy(), label="Pred", linewidth=1.4)
        ax.set_title(f"r={r}, theta={theta}")
        ax.set_xlabel("SFR channel")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_attention_weights(
    attn_weights: torch.Tensor,
    coord_map: torch.Tensor,
    filename: str,
    query_positions: Sequence[tuple[int, int]] = ((10, 10), (50, 50), (100, 100)),
) -> None:
    """可视化 cross-attention 对选定查询点的 token 权重分布。"""

    plt = _load_pyplot()
    attn = attn_weights.detach().float().cpu()
    if attn.ndim == 4:
        attn = attn[0]
    if attn.ndim != 3:
        raise ValueError(f"Expected attention shape [heads, query, token], got {tuple(attn.shape)}")
    coord = coord_map.detach().float().cpu()
    if coord.ndim == 4:
        coord = coord[0]

    num_heads, query_count, token_count = attn.shape
    # 尝试把 token 数还原为二维网格 (theta_token, r_token)。
    hp = int(round(token_count ** 0.5))
    while hp > 1 and token_count % hp != 0:
        hp -= 1
    wp = token_count // hp
    hq = int(coord.shape[-2]) if coord.ndim == 3 else int(round(query_count ** 0.5))
    wq = int(coord.shape[-1]) if coord.ndim == 3 else max(1, query_count // max(1, hq))

    positions = list(query_positions) or [(0, 0)]
    fig, axes = plt.subplots(len(positions), 1, figsize=(6, 4 * len(positions)), squeeze=False)
    # 多头平均后展示，减少单头随机波动。
    mean_attn = attn.mean(dim=0)
    for row, (qy, qx) in enumerate(positions):
        y = max(0, min(int(qy), hq - 1))
        x = max(0, min(int(qx), wq - 1))
        q_index = max(0, min(y * wq + x, query_count - 1))
        heatmap = mean_attn[q_index].reshape(hp, wp)
        ax = axes[row, 0]
        im = ax.imshow(heatmap.numpy(), cmap="magma", aspect="auto")
        if coord.ndim == 3 and coord.shape[0] == 2:
            r = float(coord[0, y, x])
            theta = float(coord[1, y, x])
            ax.scatter([r * (wp - 1)], [theta * (hp - 1)], c="cyan", s=30, marker="x")
            ax.set_title(f"query=({y},{x}) r={r:.2f}, theta={theta:.2f}")
        else:
            ax.set_title(f"query=({y},{x})")
        ax.set_xlabel("r token")
        ax.set_ylabel("theta token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-8) -> float:
    """轻量 PSNR 计算（用于图题展示）。"""

    mse = torch.mean((x.detach().float().cpu() - y.detach().float().cpu()).square())
    return float(10.0 * torch.log10(torch.tensor(1.0) / (mse + eps)))


def plot_odn_reconstruction(sharp: torch.Tensor, blur: torch.Tensor, simulated_blur: torch.Tensor, filename: str) -> None:
    """绘制 ODN 闭环对比图（sharp / blur / simulated_blur）。"""

    plt = _load_pyplot()
    images = [
        ("Sharp", sharp),
        ("Blur", blur),
        (f"ODN ({_psnr(simulated_blur, blur):.2f} dB)", simulated_blur),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for ax, (title, tensor) in zip(axes[0], images):
        ax.imshow(_to_numpy_image(tensor))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_training_dashboard(metrics_history: Iterable[dict], filename: str) -> None:
    """绘制三阶段训练关键指标仪表盘。"""

    plt = _load_pyplot()
    history = list(metrics_history)
    panels = [
        ("Stage 1 L1", "Val_PSF_SFR_L1"),
        ("Lens ID", "Val_LensIdentifiability"),
        ("Stage 2 ODN L1", "Val_ODN_L1"),
        ("Stage 3 PSNR", "PSNR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for ax, (title, key) in zip(axes.ravel(), panels):
        xs = [float(item.get("step", idx)) for idx, item in enumerate(history) if key in item]
        ys = [float(item[key]) for item in history if key in item]
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_restoration_with_zoom(
    blur: torch.Tensor,
    restored: torch.Tensor,
    sharp_gt: torch.Tensor | None = None,
    filename: str = "restoration_zoom.png",
    zoom_box: tuple[int, int, int, int] | None = None,
) -> None:
    """绘制 blur/restored/(可选)GT 及局部放大图。"""

    plt = _load_pyplot()
    tensors = [("Blur", blur), ("Restored", restored)]
    if sharp_gt is not None:
        tensors.append(("GT", sharp_gt))

    h, w = blur.shape[-2:]
    # 默认放大框：取中心区域。
    if zoom_box is None:
        box_h = max(1, h // 3)
        box_w = max(1, w // 3)
        top = max(0, (h - box_h) // 2)
        left = max(0, (w - box_w) // 2)
        zoom_box = (top, left, box_h, box_w)
    top, left, box_h, box_w = zoom_box

    fig, axes = plt.subplots(2, len(tensors), figsize=(4 * len(tensors), 7), squeeze=False)
    for col, (title, tensor) in enumerate(tensors):
        axes[0, col].imshow(_to_numpy_image(tensor))
        axes[0, col].set_title(title)
        axes[0, col].axis("off")
        crop = tensor[..., top : top + box_h, left : left + box_w]
        axes[1, col].imshow(_to_numpy_image(crop))
        axes[1, col].set_title(f"{title} zoom")
        axes[1, col].axis("off")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_center_edge_comparison(
    blur: torch.Tensor,
    restored: torch.Tensor,
    gt: torch.Tensor,
    filename: str | Path,
    mode: str = "test",
    crop_size: int = 256,
) -> None:
    """绘制中心与边缘视场的复原对比图。

根据 mode 设定不同的绘图行数：
如果 mode == 'test'，分别展示 [中心, 左上, 右上, 左下, 右下] 5行的 Blur / Restored / GT 图像块；
如果 mode == 'train'，则分别展示 [中心, 某一随机角点] 2行的 Blur / Restored / GT 图像块。
    """
    plt = _load_pyplot()
    
    h, w = blur.shape[-2:]
    c_size = min(crop_size, h, w)
    
    # 定义 5 个裁剪位置 (Top, Left, Label)
    crops_info = [
        ("Center", max(0, (h - c_size) // 2), max(0, (w - c_size) // 2)),
        ("Top-Left", 0, 0),
        ("Top-Right", 0, w - c_size),
        ("Bottom-Left", h - c_size, 0),
        ("Bottom-Right", h - c_size, w - c_size)
    ]
    
    if mode == "train":
        chosen_corner = random.choice(crops_info[1:])
        selected_crops = [crops_info[0], chosen_corner]
    else:
        selected_crops = crops_info

    n_rows = len(selected_crops)
    n_cols = 3  # Blur, Restored, GT
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    
    tensors = [("Blur", blur), ("Restored", restored), ("GT", gt)]
    
    for row, (label, top, left) in enumerate(selected_crops):
        for col, (title, tensor) in enumerate(tensors):
            crop = tensor[..., top : top + c_size, left : left + c_size]
            ax = axes[row, col]
            ax.imshow(_to_numpy_image(crop))
            ax.set_title(f"{label} - {title}")
            ax.axis("off")
            
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_residual_heatmap(
    gt: torch.Tensor, 
    restored: torch.Tensor, 
    filename: str | Path
) -> None:
    """绘制真实图与复原结果之间的残差热力图，用于观测伪影与恢复遗漏结构。"""
    plt = _load_pyplot()
    
    # 绝对残差并进行通道均值池化，转换为 2D 差异图
    residual = torch.abs(gt.detach().float().cpu() - restored.detach().float().cpu())
    if residual.ndim == 4:
        residual = residual[0]
    
    # [C, H, W] -> [H, W]
    if residual.ndim == 3:
        residual = residual.mean(dim=0)
        
    residual = residual.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # 为防止个别噪点影响，基于 percentile 截断极值
    vmax = float(np.percentile(residual, 99.5)) if np.any(residual) else 1.0
    
    im = ax.imshow(residual, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title("GT vs Restored Residual")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_full_frame(
    restored: torch.Tensor, 
    filename: str | Path
) -> None:
    """将全画幅复原图以一比一原尺寸直接保存到本地无白连白边图像，弃用 mpl 渲染。"""
    _ensure_parent(filename)
    
    img_np = _to_numpy_image(restored)
    # _to_numpy_image 得到的 shape 为 [H, W, C] (或单通道 [H, W]) 值域 [0, 1]
    
    img_np = (img_np * 255.0).round().clip(0, 255).astype(np.uint8)
    
    if img_np.ndim == 2:
        img_pil = Image.fromarray(img_np, mode="L")
    else:
        # 当通道数为单通道时，被保留为 [H, W, 1]，需要转回 L 模式或去除最后一维
        if img_np.shape[-1] == 1:
            img_pil = Image.fromarray(img_np[..., 0], mode="L")
        else:
            img_pil = Image.fromarray(img_np, mode="RGB")
            
    img_pil.save(filename)

