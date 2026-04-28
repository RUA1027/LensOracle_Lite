"""BPFR-Net 推理/测试入口脚本。

本脚本负责把训练好的 checkpoint 应用于指定数据集，并完成：
1) 恢复结果推理；
2) 指标统计（有 GT 时）；
3) 可选图像导出与可视化诊断；
4) JSON/CSV 结果落盘。

数据流与训练阶段一致：
blur -> prior_estimator -> lens_encoder -> restoration_net -> restored。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import load_config
from utils.checkpoint_sanitizer import sanitize_legacy_checkpoint, summarize_removed_keys
from utils.coord_utils import compute_polar_coord_map
from utils.metrics import PerformanceEvaluator
from utils.model_builder import build_models_from_config, build_test_dataloader_by_type, get_supported_dataset_types
from utils.visualize import (
    plot_attention_weights,
    plot_lens_table_comparison,
    plot_restoration_with_zoom,
    plot_sfr_curves,
    plot_center_edge_comparison,
    plot_residual_heatmap,
    save_full_frame,
)


def _normalize_thread_env(default_threads: int = 8) -> Tuple[str, Dict[str, str]]:
    """补齐常见 BLAS/OpenMP 线程环境变量。

    目的：在不同机器上获得更稳定的 CPU 线程行为，避免某些环境变量缺失。
    """

    cpu_count = os.cpu_count() or 1
    fallback = str(max(1, min(default_threads, cpu_count)))
    overrides: Dict[str, str] = {}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key, "").strip() == "":
            os.environ[key] = fallback
            overrides[key] = fallback
    return fallback, overrides


def _set_torch_threads(default_threads: str) -> None:
    """设置 PyTorch 计算线程与 inter-op 线程。"""

    torch_threads = int(os.environ.get("OMP_NUM_THREADS", default_threads))
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, torch_threads)))
        except RuntimeError:
            pass


def _sanitize_for_json(obj: Any) -> Any:
    """把对象递归转换为 JSON 安全结构。

    对 NaN/Inf 浮点值转为 None，避免 `json.dump(..., allow_nan=False)` 失败。
    """

    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """把 `[C,H,W]` 或 `[H,W]` 张量转换为 PIL 图像。"""

    array = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    array = (array * 255.0).round().astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def save_comparison_image(blur: torch.Tensor, sharp_gt: torch.Tensor, restored: torch.Tensor, save_path: str) -> None:
    """保存三联对比图（blur | restored | sharp_gt）。"""

    blur_pil = _tensor_to_pil(blur)
    sharp_pil = _tensor_to_pil(sharp_gt)
    restored_pil = _tensor_to_pil(restored)
    width, height = blur_pil.size
    canvas = Image.new("RGB", (width * 3, height))
    canvas.paste(blur_pil, (0, 0))
    canvas.paste(restored_pil, (width, 0))
    canvas.paste(sharp_pil, (width * 2, 0))
    canvas.save(save_path)


def save_single_result(restored: torch.Tensor, save_path: str) -> None:
    """保存单张恢复结果图。"""

    _tensor_to_pil(restored).save(save_path)


def _resolve_filename(raw_value: Any, index: int) -> str:
    """从 batch 字段中解析样本文件名。"""

    if isinstance(raw_value, (list, tuple)):
        return str(raw_value[index]) if raw_value else f"sample_{index:04d}.png"
    return str(raw_value)


def _build_output_dir(config, output_override: str | None, dataset_type: str) -> str:
    """解析测试输出目录。

    - 若用户显式传入 `--output`，直接使用；
    - 否则按 `experiment.output_dir/test_<dataset>_<timestamp>` 自动生成。
    """

    if output_override:
        output_dir = Path(output_override)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.experiment.output_dir) / f"test_{dataset_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _load_checkpoint(prior_estimator, lens_encoder, odn, restoration_net, checkpoint_path: str, device: str) -> Dict[str, Any]:
    """加载并分发 checkpoint 权重到四个子模块。

    同时执行 legacy key 清理，过滤掉 `total_ops/total_params` 等非参数字段。
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "restoration_net" not in checkpoint:
        raise ValueError("Expected a Stage3 restoration checkpoint containing restoration_net.")
    checkpoint, sanitization_report = sanitize_legacy_checkpoint(checkpoint)
    # 四模块按 key 分别加载，strict=False 便于兼容轻微结构差异。
    if "lens_table_encoder" in checkpoint:
        lens_encoder.load_state_dict(checkpoint.get("lens_table_encoder", {}), strict=False)
    restoration_state = {
        key: value
        for key, value in dict(checkpoint.get("restoration_net", {}) or {}).items()
        if "total_ops" not in key and "total_params" not in key
    }
    restoration_net.load_state_dict(restoration_state, strict=False)
    lens_encoder.eval()
    restoration_net.eval()
    checkpoint["sanitization_report"] = sanitization_report
    return checkpoint


def _write_results(output_dir: str, payload: Dict[str, Any], has_gt: bool) -> None:
    """写出测试结果 JSON 和 CSV。"""

    results_path = Path(output_dir) / "test_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    csv_path = Path(output_dir) / "test_results.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        if has_gt:
            handle.write("filename,PSNR,SSIM,MAE,LPIPS\n")
            for row in payload["per_image_results"]:
                handle.write(
                    f"{row['filename']},{row['PSNR']},{row['SSIM']},{row['MAE']},{row['LPIPS']}\n"
                )
        else:
            handle.write("filename\n")
            for row in payload["per_image_results"]:
                handle.write(f"{row['filename']}\n")


def _resolve_prior_table(batch: Dict[str, Any], mode: str, device: str) -> torch.Tensor | None:
    """Resolve a Stage3 prior tensor from batch fields without image-to-prior inference."""

    if mode == "none":
        return None
    key_by_mode = {"correct_gt": "gt_psf_sfr", "incorrect_gt": "incorrect_gt_psf_sfr"}
    key = key_by_mode.get(str(mode))
    if key is None:
        raise ValueError(f"Unsupported prior mode: {mode}")
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Batch is missing required prior tensor '{key}'.")
    return value.to(device)


def _print_metric_table(title: str, metrics: Dict[str, Any]) -> None:
    """以统一格式打印指标字典。"""

    print(title)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            print(f"  {key}: {float(value):.6f}")
        else:
            print(f"  {key}: {value}")


def _export_visuals(
    sample_dir: Path,
    blur: torch.Tensor,
    restored: torch.Tensor,
    pred_table: torch.Tensor | None,
    lens_features,
    restoration_net,
    crop_info,
    sharp: torch.Tensor | None,
    gt_table: torch.Tensor | None,
    channels: List[int],
) -> None:
    """导出单样本可视化产物。

    包括：
    - lens table 对比图 / SFR 曲线（有 gt_table 时）；
    - restoration 细节放大图；
    - 一组 attention 权重图（若模型返回注意力缓存）。
    """

    sample_dir.mkdir(parents=True, exist_ok=True)
    if pred_table is not None and gt_table is not None:
        plot_lens_table_comparison(pred_table.cpu(), gt_table.cpu(), filename=str(sample_dir / "lens_table_comparison.png"), channels_to_show=channels)
        plot_sfr_curves(pred_table.cpu(), gt_table.cpu(), filename=str(sample_dir / "sfr_curves.png"))
    plot_restoration_with_zoom(blur.cpu(), restored.cpu(), sharp_gt=None if sharp is None else sharp.cpu(), filename=str(sample_dir / "restoration_zoom.png"))
    if not lens_features:
        return
    # 额外跑一次 return_attn=True 的前向，仅用于可视化注意力。
    with torch.no_grad():
        batch_blur = blur.unsqueeze(0)
        _, attn = restoration_net(batch_blur, {key: value[:1] for key, value in lens_features.items()}, crop_info=crop_info, return_attn=True)
    if attn:
        name, weights = next(iter(attn.items()))
        query_count = int(weights.shape[-2])
        hq = max(1, int(math.sqrt(query_count)))
        while hq > 1 and query_count % hq != 0:
            hq -= 1
        wq = max(1, query_count // hq)
        coord = compute_polar_coord_map(hq, wq, blur.shape[-2], blur.shape[-1], crop_info=crop_info, device=weights.device, dtype=weights.dtype)
        plot_attention_weights(weights[0].cpu(), coord[0].cpu(), filename=str(sample_dir / f"attention_{name}.png"))


def main() -> None:
    """命令行主入口。"""

    thread_default, thread_overrides = _normalize_thread_env()
    _set_torch_threads(thread_default)
    parser = argparse.ArgumentParser(description="BPFR-Net lens-table fusion testing")
    parser.add_argument("--checkpoint", "-ckpt", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default="config/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dataset-type", "-dt", type=str, default="omnilens_mixlib", choices=get_supported_dataset_types())
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-restored", action="store_true")
    parser.add_argument("--export-visuals", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=8)
    args = parser.parse_args()

    # 1) 加载配置并解析设备。
    config = load_config(args.config, args.override if args.override else None)
    prior_mode = str(getattr(config.ablation, "eval_prior_mode", "correct_gt"))
    lens_encoder_enabled = bool(getattr(config.ablation, "lens_encoder_enabled", True))
    device = str(config.experiment.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead.")
        device = "cpu"
    config.experiment.device = device
    if thread_overrides:
        print("Normalized thread env vars: " + ", ".join(f"{key}={value}" for key, value in thread_overrides.items()))

    # 2) 准备输出目录与可选导出子目录。
    output_dir = _build_output_dir(config, args.output, args.dataset_type)
    if args.save_images:
        (Path(output_dir) / "comparisons").mkdir(parents=True, exist_ok=True)
    if args.save_restored:
        (Path(output_dir) / "restored").mkdir(parents=True, exist_ok=True)

    # 3) 构建模型并加载 checkpoint。
    prior_estimator, lens_encoder, odn, restoration_net = build_models_from_config(config, device)
    checkpoint = _load_checkpoint(prior_estimator, lens_encoder, odn, restoration_net, args.checkpoint, device)
    report = checkpoint.get("sanitization_report", {})
    if report and (report.get("removed_prior_keys") or report.get("removed_restoration_keys")):
        print(f"Checkpoint stripped legacy keys: {summarize_removed_keys(report)}")

    # 4) 构建 evaluator 与测试 dataloader。
    evaluator = PerformanceEvaluator(device=device)
    test_loader, has_gt = build_test_dataloader_by_type(
        args.dataset_type,
        config=config,
        data_root_override=args.data_root,
        require_psf_sfr=prior_mode in {"correct_gt", "incorrect_gt"},
        require_incorrect_psf_sfr=prior_mode == "incorrect_gt",
    )
    if not has_gt and not args.save_restored:
        args.save_restored = True
        (Path(output_dir) / "restored").mkdir(parents=True, exist_ok=True)

    visual_cfg = getattr(getattr(config, "visualization", None), "export", None)
    visual_enabled = bool(args.export_visuals or (visual_cfg is not None and getattr(visual_cfg, "enabled", False)))
    channels = list(getattr(visual_cfg, "prior_channels", [0, 10, 20, 30, 40, 50, 60])) if visual_cfg else [0, 10, 20]
    visual_root = Path(output_dir) / "visualizations"
    # 强制创建 visualizations 目录，因为即使其它可视化被拦截，也要输出3种独立诊断图
    visual_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BPFR-Net lens-table fusion testing")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Has ground truth: {has_gt}")

    results: List[Dict[str, float | str]] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    mae_values: List[float] = []
    lpips_values: List[float] = []
    visual_count = 0

    # 5) 主测试循环：推理 -> 指标聚合 -> 可选导出。
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test", leave=False):
            blur = batch["blur"].to(device)
            sharp_raw = batch.get("sharp")
            sharp = sharp_raw.to(device) if torch.is_tensor(sharp_raw) else None
            crop_info = batch.get("crop_info")
            crop_info = crop_info.to(device) if torch.is_tensor(crop_info) else None
            gt_table = batch.get("gt_psf_sfr")
            gt_table = gt_table.to(device) if torch.is_tensor(gt_table) else None
            # 与训练 stage3 一致的推理链路。
            prior_table = _resolve_prior_table(batch, prior_mode, device)
            lens_features = (
                lens_encoder(prior_table)
                if lens_encoder_enabled and prior_table is not None
                else None
            )
            restored = restoration_net(blur, lens_features, crop_info=crop_info)

            for index in range(int(blur.shape[0])):
                filename = _resolve_filename(batch.get("filename", "sample.png"), index)
                # 逐图计算指标，方便同时输出 per-image 明细。
                metrics = evaluator.compute_image_metrics(restored[index : index + 1], None if sharp is None else sharp[index : index + 1])
                results.append({"filename": filename, **metrics})
                psnr_values.append(metrics["PSNR"])
                ssim_values.append(metrics["SSIM"])
                mae_values.append(metrics["MAE"])
                lpips_values.append(metrics["LPIPS"])
                if args.save_images and sharp is not None:
                    save_comparison_image(blur[index].cpu(), sharp[index].cpu(), restored[index].cpu(), str(Path(output_dir) / "comparisons" / filename))
                if args.save_restored:
                    save_single_result(restored[index].cpu(), str(Path(output_dir) / "restored" / filename))
                
                # 强制输出 3 种独立诊断图 (无视 visual_limit 拦截)
                diag_dir = visual_root / f"{index:03d}_{Path(filename).stem}"
                
                # 全画幅复原图
                save_full_frame(restored[index].cpu(), str(diag_dir / "restored_full.png"))
                
                # 中心与边缘对比、残差热力图 (需要存在 GT 才能比对残差)
                if sharp is not None:
                    plot_center_edge_comparison(blur[index].cpu(), restored[index].cpu(), sharp[index].cpu(), str(diag_dir / "center_edge_comparison.png"), mode="test")
                    plot_residual_heatmap(sharp[index].cpu(), restored[index].cpu(), str(diag_dir / "residual_heatmap.png"))

                if visual_enabled and visual_count < max(0, int(args.visual_limit)):
                    single_crop = crop_info[index : index + 1] if torch.is_tensor(crop_info) else None
                    _export_visuals(
                        sample_dir=visual_root / f"{visual_count:03d}_{Path(filename).stem}",
                        blur=blur[index],
                        restored=restored[index],
                        pred_table=None if prior_table is None else prior_table[index],
                        lens_features=None if lens_features is None else {key: value[index : index + 1] for key, value in lens_features.items()},
                        restoration_net=restoration_net,
                        crop_info=single_crop,
                        sharp=None if sharp is None else sharp[index],
                        gt_table=None if gt_table is None else gt_table[index],
                        channels=channels,
                    )
                    visual_count += 1

    # 6) 汇总指标与模型参数规模，写出最终报告。
    average_metrics = {
        "PSNR": evaluator.aggregate_metric_list(psnr_values),
        "SSIM": evaluator.aggregate_metric_list(ssim_values),
        "MAE": evaluator.aggregate_metric_list(mae_values),
        "LPIPS": evaluator.aggregate_metric_list(lpips_values),
        "Num_Images": len(results),
    }
    model_stats = {
        "restoration_params": evaluator._count_parameters(restoration_net),
        "lens_encoder_params": evaluator._count_parameters(lens_encoder) if lens_encoder_enabled else 0.0,
        "total_active_params": evaluator._count_parameters(
            restoration_net,
            lens_encoder if lens_encoder_enabled else None,
        ),
    }
    payload = {
        "dataset_type": args.dataset_type,
        "has_gt": has_gt,
        "average_metrics": average_metrics,
        "model_stats": model_stats,
        "per_image_results": results,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "ablation": dict(getattr(config.ablation, "__dict__", {})),
        "prior_mode": prior_mode,
    }
    _write_results(output_dir, payload, has_gt=has_gt)
    _print_metric_table("Average metrics:", average_metrics)
    _print_metric_table("Model stats:", {"Total active params (M)": model_stats["total_active_params"]})
    if visual_enabled:
        print(f"Visualizations exported: {visual_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
