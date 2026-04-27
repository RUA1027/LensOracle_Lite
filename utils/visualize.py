"""BPFR-Net 可视化工具函数集合。

覆盖内容包括：
1) lens-table 通道对比图与 SFR 曲线；
2) attention 权重热力图；
3) ODN 重建对比图；
4) 训练阶段关键指标仪表盘；
5) restoration 结果局部放大图。

所有函数均以“可离线保存 PNG”为目标，采用无界面后端 Agg。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

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


def _ordered_variant_items(
    restored_by_variant: Mapping[str, torch.Tensor],
    order: Sequence[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    keys = list(order) if order is not None else list(restored_by_variant.keys())
    return [(key, restored_by_variant[key]) for key in keys if key in restored_by_variant]


def plot_ablation_recovery_comparison(
    blur: torch.Tensor,
    restored_by_variant: Mapping[str, torch.Tensor],
    sharp_gt: torch.Tensor | None,
    filename: str,
    order: Sequence[str] | None = None,
) -> None:
    """Plot full-size Blur / variants / GT comparison for ablation reports."""

    plt = _load_pyplot()
    panels: list[tuple[str, torch.Tensor]] = [("Blur", blur)]
    panels.extend(_ordered_variant_items(restored_by_variant, order))
    if sharp_gt is not None:
        panels.append(("GT", sharp_gt))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2), squeeze=False)
    for ax, (title, tensor) in zip(axes[0], panels):
        ax.imshow(_to_numpy_image(tensor))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _crop_center_and_edge(tensor: torch.Tensor, crop_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = tensor.shape[-2:]
    size = int(crop_size or max(1, min(h, w) // 4))
    size = max(1, min(size, h, w))
    center_top = max(0, (h - size) // 2)
    center_left = max(0, (w - size) // 2)
    center = tensor[..., center_top : center_top + size, center_left : center_left + size]
    edge = tensor[..., 0:size, 0:size]
    return center, edge


def plot_center_edge_ablation_crops(
    blur: torch.Tensor,
    restored_by_variant: Mapping[str, torch.Tensor],
    sharp_gt: torch.Tensor | None,
    filename: str,
    order: Sequence[str] | None = None,
    crop_size: int | None = None,
) -> None:
    """Plot same-size center and edge crops for each ablation output."""

    plt = _load_pyplot()
    panels: list[tuple[str, torch.Tensor]] = [("Blur", blur)]
    panels.extend(_ordered_variant_items(restored_by_variant, order))
    if sharp_gt is not None:
        panels.append(("GT", sharp_gt))

    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.4), squeeze=False)
    for col, (title, tensor) in enumerate(panels):
        center, edge = _crop_center_and_edge(tensor, crop_size=crop_size)
        axes[0, col].imshow(_to_numpy_image(center))
        axes[0, col].set_title(f"{title} center")
        axes[0, col].axis("off")
        axes[1, col].imshow(_to_numpy_image(edge))
        axes[1, col].set_title(f"{title} edge")
        axes[1, col].axis("off")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_ablation_error_maps(
    restored_by_variant: Mapping[str, torch.Tensor],
    sharp_gt: torch.Tensor,
    filename: str,
    order: Sequence[str] | None = None,
) -> None:
    """Plot mean absolute error maps with one shared color scale."""

    plt = _load_pyplot()
    items = _ordered_variant_items(restored_by_variant, order)
    errors = []
    for title, tensor in items:
        error = (tensor.detach().float().cpu() - sharp_gt.detach().float().cpu()).abs()
        if error.ndim == 4:
            error = error[0]
        if error.ndim == 3:
            error = error.mean(dim=0)
        errors.append((title, error))
    vmax = max([float(err.max()) for _, err in errors] + [1.0e-6])

    fig, axes = plt.subplots(1, len(errors), figsize=(4.0 * len(errors), 4.0), squeeze=False)
    last_im = None
    for ax, (title, error) in zip(axes[0], errors):
        last_im = ax.imshow(error.numpy(), cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes[0]), fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_incorrect_prior_injection(
    correct_restored: torch.Tensor,
    incorrect_restored: torch.Tensor,
    sharp_gt: torch.Tensor | None,
    filename: str,
    true_lens: str,
    injected_lens: str,
) -> None:
    """Plot CorrectPrior vs IncorrectPrior and label the true/injected lenses."""

    plt = _load_pyplot()
    panels: list[tuple[str, torch.Tensor]] = [
        ("CorrectPrior", correct_restored),
        (f"IncorrectPrior\ntrue={true_lens}\ninjected={injected_lens}", incorrect_restored),
    ]
    if sharp_gt is not None:
        panels.append(("GT", sharp_gt))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), squeeze=False)
    for ax, (title, tensor) in zip(axes[0], panels):
        ax.imshow(_to_numpy_image(tensor))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    _ensure_parent(filename)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
