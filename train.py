"""BPFR-Net 三阶段训练入口脚本。

该脚本负责把 `ThreeStageTrainer` 驱动起来，并处理训练运行期的工程化职责：
1) 解析配置与运行参数；
2) 自动恢复/热启动 checkpoint；
3) 按阶段迭代训练与验证；
4) 管理 best/latest/final checkpoint；
5) 导出阶段可视化与毕业报告。

与常规“按 epoch 固定循环”不同，本脚本以 global_step（优化器步数）为主轴，
并允许在不同阶段使用不同 batch size 与迭代窗口。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from tqdm import tqdm

from config import load_config
from trainer import STAGE1, STAGE2, STAGE3
from utils.checkpoint_sanitizer import sanitize_legacy_checkpoint, summarize_removed_keys
from utils.coord_utils import compute_polar_coord_map
from utils.metrics import PerformanceEvaluator, extract_stage_score, resolve_stage_metric_spec
from utils.model_builder import build_mixlib_dataloader, build_models_from_config, build_trainer_from_config
from utils.visualize import (
    plot_attention_weights,
    plot_lens_table_comparison,
    plot_odn_reconstruction,
    plot_restoration_with_zoom,
    plot_sfr_curves,
)


STAGE_TAGS = {STAGE1: "stage1", STAGE2: "stage2", STAGE3: "stage3"}
BEST_PERFORMANCE_FILENAME = "best_performance.json"


def _stage_tag(stage: str) -> str:
    """把内部 stage 名映射为更短的文件名后缀。"""

    return STAGE_TAGS.get(stage, stage)


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    """兼容 dict/object 两种配置访问方式。"""

    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _dedupe_paths(paths: Iterable[Path]) -> Tuple[Path, ...]:
    """按字符串路径去重并保持原顺序。"""

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def _write_json_payload(path: Path, payload: Dict[str, Any]) -> None:
    """原子写 JSON（临时文件写入后 replace）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    tmp_path.replace(path)


def _coalesce_metric(metrics: Dict[str, Any], *keys: str) -> float:
    """按候选 key 顺序提取第一个有限数值指标。"""

    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return float("nan")


def _legacy_stage1_report_unused(val_metrics: Dict[str, Any], graduation_cfg: Any) -> Dict[str, Any]:
    """根据 Stage 1 验证指标生成毕业报告。

    报告包含：
    - 规范化综合分数（加权）；
    - 通过/告警/失败状态；
    - 阈值与原始指标快照。
    """

    prior_l1 = _coalesce_metric(val_metrics, "Val_PSF_SFR_L1", "val_psf_sfr_l1", "val_loss")
    prior_msssim = _coalesce_metric(val_metrics, "Val_PSF_SFR_MSSSIM", "val_psf_sfr_msssim")
    identifiability = _coalesce_metric(val_metrics, "Val_LensIdentifiability")
    batch_std = _coalesce_metric(val_metrics, "Val_BatchStd", "batch_std")
    weights = {
        "prior_l1": float(_cfg_get(graduation_cfg, "prior_l1_weight", 0.45)),
        "prior_msssim": float(_cfg_get(graduation_cfg, "prior_msssim_weight", 0.35)),
        "lens_identifiability": float(_cfg_get(graduation_cfg, "lens_identifiability_weight", 0.20)),
    }
    score_components = {
        "prior_l1": 0.0 if not math.isfinite(prior_l1) else 1.0 / (1.0 + max(prior_l1, 0.0)),
        "prior_msssim": 0.0 if not math.isfinite(prior_msssim) else min(1.0, max(0.0, prior_msssim)),
        "lens_identifiability": 0.0 if not math.isfinite(identifiability) else min(1.0, max(0.0, identifiability)),
    }
    aggregate_score = sum(weights[key] * score_components[key] for key in score_components)
    batch_std_warn_min = float(_cfg_get(graduation_cfg, "batch_std_warn_min", 0.010))
    batch_std_fail_min = float(_cfg_get(graduation_cfg, "batch_std_fail_min", 0.004))
    prior_l1_fail_max = float(_cfg_get(graduation_cfg, "prior_l1_fail_max", 0.22))
    prior_msssim_fail_min = float(_cfg_get(graduation_cfg, "prior_msssim_fail_min", 0.55))
    warnings: list[str] = []
    if math.isfinite(batch_std) and batch_std < batch_std_warn_min:
        warnings.append("batch_std_low")
    if math.isfinite(batch_std) and batch_std < batch_std_fail_min:
        status = "FAIL_COLLAPSE"
    elif (math.isfinite(prior_l1) and prior_l1 > prior_l1_fail_max) or (
        math.isfinite(prior_msssim) and prior_msssim < prior_msssim_fail_min
    ):
        status = "FAIL_PHYSICS_STRUCTURE"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return {
        "mode": str(_cfg_get(graduation_cfg, "mode", "soft")),
        "status": status,
        "aggregate_score": float(aggregate_score),
        "weights": weights,
        "metrics": {
            "Val_PSF_SFR_L1": prior_l1,
            "Val_PSF_SFR_MSSSIM": prior_msssim,
            "Val_LensIdentifiability": identifiability,
            "Val_BatchStd": batch_std,
        },
        "warnings": warnings,
        "thresholds": {
            "batch_std_warn_min": batch_std_warn_min,
            "batch_std_fail_min": batch_std_fail_min,
            "prior_l1_fail_max": prior_l1_fail_max,
            "prior_msssim_fail_min": prior_msssim_fail_min,
        },
    }


def _build_fixed_checkpoint_dir(config) -> Path:
    """构造“固定命名” checkpoint 目录（不含 run 时间戳）。"""

    return Path(config.experiment.output_dir) / str(config.experiment.name) / str(config.experiment.checkpoints_subdir or "checkpoints")


def _resolve_stage_metric_name(config, stage: str) -> str:
    """读取某阶段用于 best 判定的配置指标名。"""

    if stage == STAGE1:
        return str(getattr(config.checkpoint, "stage1_metric", "val_psf_sfr_l1"))
    if stage == STAGE2:
        return str(getattr(config.checkpoint, "stage2_metric", "val_odn_l1"))
    return str(getattr(config.checkpoint, "stage3_metric", "psnr"))


def _best_performance_template() -> Dict[str, Dict[str, Any]]:
    """best_performance.json 的标准骨架。"""

    return {STAGE3: {}, "latest": {}}


def _load_best_performance(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取 best_performance 状态文件；异常时返回模板。"""

    template = _best_performance_template()
    if not path.exists():
        return template
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return template
    if isinstance(payload, dict):
        for key in template:
            if isinstance(payload.get(key), dict):
                template[key] = dict(payload[key])
    return template


def _persist_best_performance(paths: Tuple[Path, ...], state: Dict[str, Dict[str, Any]]) -> None:
    """把 best/latest 状态同步写到多个目标路径。"""

    for path in _dedupe_paths(paths):
        _write_json_payload(path, state)


def _build_best_tracking_state(config, best_performance_state: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """基于配置和历史状态构建运行期 best 跟踪器。"""

    tracking: Dict[str, Dict[str, Any]] = {}
    for stage in (STAGE3,):
        metric_name = _resolve_stage_metric_name(config, stage)
        _, _, maximize = resolve_stage_metric_spec(metric_name, stage)
        fallback = float("-inf") if maximize else float("inf")
        stored = best_performance_state.get(stage, {}).get("metric_value")
        score = float(stored) if isinstance(stored, (int, float)) and math.isfinite(float(stored)) else fallback
        tracking[stage] = {"metric_name": metric_name, "maximize": maximize, "score": score}
    return tracking


def _save_checkpoint_bundle(trainer, checkpoint_dirs: Tuple[Path, ...], filenames: Tuple[str, ...], epoch: int, stage: str, global_step: int, val_metrics: Dict[str, Any]) -> None:
    """把同一 checkpoint 内容写入多个目录与多个文件名。"""

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            trainer.save_checkpoint(str(checkpoint_dir / filename), epoch=epoch, stage=stage, global_step=global_step, val_metrics=val_metrics)


def _resolve_auto_start_checkpoint(requested_stage: str, fixed_checkpoints_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """在未显式传参时自动推断 resume/init-from 路径。

    策略优先级：
    - 同阶段 latest；
    - 上一阶段 best（用于 warm start）；
    - 通用 best/latest 兜底。
    """

    latest = fixed_checkpoints_dir / "latest.pt"
    latest_stage3 = fixed_checkpoints_dir / "latest_stage3.pt"
    best = fixed_checkpoints_dir / "best.pt"
    if requested_stage in {"3", "all"}:
        if latest_stage3.exists():
            return str(latest_stage3), None, "auto_resume_stage3_latest"
        if latest.exists():
            return str(latest), None, "auto_resume_latest"
        return (None, str(best), "auto_init_from_best") if best.exists() else (None, None, None)
    return None, None, None


def _apply_stage_warm_start(modules: Dict[str, torch.nn.Module], checkpoint_path: str, device: str) -> Dict[str, list[str]]:
    """把 checkpoint 中可匹配权重注入给指定模块集合。"""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Warm-start checkpoint must be a dict.")
    checkpoint, report = sanitize_legacy_checkpoint(checkpoint)
    for name, module in modules.items():
        state = checkpoint.get(name)
        if not isinstance(state, dict):
            continue
        state = {
            key: value
            for key, value in state.items()
            if isinstance(key, str) and "total_ops" not in key and "total_params" not in key
        }
        module.load_state_dict(state, strict=False)
    return report


def _normalize_thread_env(default_threads: int = 8) -> Tuple[str, Dict[str, str]]:
    """补齐多种线程环境变量，减少不同平台线程行为差异。"""

    cpu_count = os.cpu_count() or 1
    fallback = str(max(1, min(default_threads, cpu_count)))
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    overrides: Dict[str, str] = {}
    for key in keys:
        if os.environ.get(key, "").strip() == "":
            os.environ[key] = fallback
            overrides[key] = fallback
    return fallback, overrides


def _set_torch_threads(default_threads: str) -> None:
    """设置 PyTorch 计算线程配置。"""

    torch_threads = int(os.environ.get("OMP_NUM_THREADS", default_threads))
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, torch_threads)))
        except RuntimeError:
            pass


def _set_seed(seed: int) -> None:
    """设置 Python / PyTorch / NumPy 随机种子。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _stage_window(config, requested_stage: str) -> Tuple[int, int]:
    """返回请求阶段对应的 global_step 窗口 `[start, end)`。"""

    schedule = config.training.stage_schedule
    stage3 = int(schedule.stage3_iterations)
    return {
        "3": (0, stage3),
        "all": (0, stage3),
    }[requested_stage]


def _stage_display_name(stage: str) -> str:
    """把内部 stage 常量映射为可读文本。"""

    return {
        STAGE1: "Legacy Stage 1: Prior Estimator",
        STAGE2: "Legacy Stage 2: Optical Degradation Network",
        STAGE3: "Stage 3: Restoration",
    }.get(stage, stage)


def _build_output_dir(config) -> str:
    """构造本次训练 run 的输出目录（可带时间戳）。"""

    base_dir = Path(config.experiment.output_dir)
    run_name = config.experiment.run_name or config.experiment.name
    if bool(getattr(config.experiment, "use_timestamp", True)):
        run_name = f"{run_name}_{datetime.now().strftime(getattr(config.experiment, 'timestamp_format', '%m%d_%H%M'))}"
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _next_interval_boundary(current_step: int, interval: int, upper_bound: int) -> int:
    """计算下一个触发边界（validation/save）。"""

    if interval <= 0:
        return int(upper_bound)
    boundary = ((int(current_step) // int(interval)) + 1) * int(interval)
    return min(boundary, int(upper_bound))


def _is_interval_trigger(step: int, interval: int) -> bool:
    """判断当前 step 是否命中间隔触发点。"""

    return interval > 0 and step > 0 and int(step) % int(interval) == 0


def _extract_validation_loss(metrics: Dict[str, Any], stage: str) -> Optional[float]:
    """按阶段偏好顺序提取验证损失近似值。"""

    preferred = {
        STAGE1: ("Val_PSF_SFR_L1", "val_psf_sfr_l1", "val_loss", "loss"),
        STAGE2: ("Val_ODN_L1", "val_odn_l1", "val_loss", "loss"),
        STAGE3: ("val_loss", "MAE", "mae", "loss"),
    }
    for key in preferred.get(stage, ("val_loss", "loss")):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _save_periodic_checkpoint(trainer, checkpoints_dir: Path, epoch: int, stage: str, global_step: int, val_metrics: Dict[str, Any], save_interval: int) -> bool:
    """按 save_interval 保存 latest checkpoint。"""

    if not _is_interval_trigger(global_step, save_interval):
        return False
    _save_checkpoint_bundle(
        trainer=trainer,
        checkpoint_dirs=(checkpoints_dir,),
        filenames=("latest.pt", f"latest_{_stage_tag(stage)}.pt"),
        epoch=epoch,
        stage=stage,
        global_step=global_step,
        val_metrics=val_metrics,
    )
    return True


def _maybe_save_best_checkpoint(trainer, checkpoints_dir: Path, epoch: int, stage: str, global_step: int, val_metrics: Dict[str, Any], best_tracking_state: Dict[str, Dict[str, Any]]) -> bool:
    """若当前验证结果优于历史 best，则保存 best checkpoint。"""

    stage_state = best_tracking_state.get(stage)
    if not isinstance(stage_state, dict):
        return False
    score_info = extract_stage_score(val_metrics, str(stage_state.get("metric_name", "val_loss")), stage)
    if score_info is None:
        return False
    _, score, maximize = score_info
    fallback = float("-inf") if maximize else float("inf")
    previous_best = stage_state.get("score", fallback)
    if not isinstance(previous_best, (int, float)) or not math.isfinite(float(previous_best)):
        previous_best = fallback
    improved = float(score) > float(previous_best) if maximize else float(score) < float(previous_best)
    if not improved:
        return False
    stage_state["score"] = float(score)
    _save_checkpoint_bundle(
        trainer=trainer,
        checkpoint_dirs=(checkpoints_dir,),
        filenames=("best.pt", f"best_{_stage_tag(stage)}.pt"),
        epoch=epoch,
        stage=stage,
        global_step=global_step,
        val_metrics=val_metrics,
    )
    return True


def _average_metrics(totals: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    """把累计和转为平均值；无样本键返回 NaN。"""

    return {key: totals[key] / counts[key] if counts.get(key, 0) > 0 else float("nan") for key in totals}


def _train_one_cycle(trainer, loader, stage: str, global_step: int, target_step: int) -> Tuple[Dict[str, float], int, Optional[Dict[str, Any]]]:
    """执行一个训练小周期，直到达到 `target_step`。

    该函数内部兼容梯度累积逻辑：
    - 仅当 `optimizer_step` 发生时，global_step 才递增；
    - 循环结束后若有未 flush 的累积梯度，会尝试补一次 step。
    """

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    last_batch = None
    progress = tqdm(loader, desc=f"{stage} @step {global_step}", leave=False)
    for batch in progress:
        metrics = trainer.train_step(batch=batch, stage=stage)
        last_batch = batch
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        if float(metrics.get("optimizer_step", 0.0)) > 0.0:
            global_step += 1
            trainer.log_to_tensorboard(metrics, global_step, prefix="train_step")
        progress.set_postfix(loss=f"{metrics.get('loss', float('nan')):.4f}")
        if global_step >= target_step:
            break
    # 循环结束后清理残余累积梯度，避免状态泄漏到下一周期。
    if trainer.pending_accumulation_steps > 0:
        if global_step < target_step and trainer.flush_pending_gradients(stage=stage):
            global_step += 1
        else:
            trainer.reset_after_oom()
    return _average_metrics(totals, counts), global_step, last_batch


def _validate_one_epoch(trainer, loader, stage: str, device: str, evaluator: PerformanceEvaluator) -> Dict[str, float]:
    """按当前阶段调用对应 evaluator 接口。"""

    return evaluator.evaluate(
        restoration_net=trainer.restoration_net,
        psf_net=None,
        lens_table_encoder=trainer.lens_table_encoder if trainer.lens_encoder_enabled else None,
        val_loader=loader,
        device=device,
        prior_mode=trainer.eval_prior_mode,
    )


def _should_export_visuals(config, epoch: int) -> bool:
    """判断当前 epoch 是否应导出训练可视化。"""

    export_cfg = getattr(getattr(config, "visualization", None), "export", None)
    if export_cfg is None or not bool(getattr(export_cfg, "enabled", False)):
        return False
    interval = int(getattr(export_cfg, "interval", 0))
    return interval > 0 and ((epoch + 1) % interval == 0)


def _export_epoch_visuals(trainer, config, batch: Optional[Dict[str, Any]], stage: str, epoch: int, output_dir: str, device: str) -> None:
    """导出某个 epoch 的代表性可视化样本。"""

    if batch is None:
        return
    vis_dir = Path(output_dir) / "visualizations" / f"epoch_{epoch + 1:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = getattr(config.visualization, "export", None)
    channels = list(getattr(export_cfg, "prior_channels", [0, 10, 20, 30, 40, 50, 60]))
    blur = batch["blur"][:1].to(device)
    sharp = batch.get("sharp")
    sharp_tensor = sharp[:1].to(device) if torch.is_tensor(sharp) else None
    crop_info = batch.get("crop_info")
    crop_info = crop_info[:1].to(device) if torch.is_tensor(crop_info) else None
    gt_table = batch.get("gt_psf_sfr")
    gt_table = gt_table[:1].to(device) if torch.is_tensor(gt_table) else None
    # 暂存训练模式，导出完成后恢复，避免影响后续训练状态。
    states = [trainer.lens_table_encoder.training, trainer.restoration_net.training]
    trainer.lens_table_encoder.eval()
    trainer.restoration_net.eval()
    try:
        with torch.no_grad():
            prior_table = trainer._resolve_prior_table(batch, trainer.train_prior_mode)
            if prior_table is not None:
                prior_table = prior_table[:1]
            features = trainer.lens_table_encoder(prior_table) if trainer.lens_encoder_enabled and prior_table is not None else None
            restored, attn = trainer.restoration_net(blur, features, crop_info=crop_info, return_attn=True)
            plot_restoration_with_zoom(blur[0].cpu(), restored[0].cpu(), sharp_gt=None if sharp_tensor is None else sharp_tensor[0].cpu(), filename=str(vis_dir / "restoration_zoom.png"))
            if prior_table is not None and gt_table is not None:
                plot_lens_table_comparison(prior_table[0].cpu(), gt_table[0].cpu(), filename=str(vis_dir / "lens_table_comparison.png"), channels_to_show=channels)
                plot_sfr_curves(prior_table[0].cpu(), gt_table[0].cpu(), filename=str(vis_dir / "sfr_curves.png"))
            if bool(getattr(export_cfg, "export_attention_weights", True)) and attn:
                name, weights = next(iter(attn.items()))
                query_count = int(weights.shape[-2])
                hq = max(1, int(math.sqrt(query_count)))
                while hq > 1 and query_count % hq != 0:
                    hq -= 1
                wq = max(1, query_count // hq)
                coord = compute_polar_coord_map(hq, wq, blur.shape[-2], blur.shape[-1], crop_info=crop_info, device=weights.device, dtype=weights.dtype)
                plot_attention_weights(weights[0].cpu(), coord[0].cpu(), filename=str(vis_dir / f"attention_{name}.png"))
    finally:
        trainer.lens_table_encoder.train(states[0])
        trainer.restoration_net.train(states[1])


def _print_metric_table(title: str, metrics: Dict[str, Any]) -> None:
    """格式化打印指标字典。"""

    print(title)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            print(f"  {key}: {float(value):.6f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    """命令行训练入口。"""

    # 0) 线程环境初始化（优先在导入后、训练前完成）。
    thread_default, thread_overrides = _normalize_thread_env()
    _set_torch_threads(thread_default)
    parser = argparse.ArgumentParser(description="BPFR-Net lens-table fusion training")
    parser.add_argument("--config", "-c", default="config/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--stage", type=str, default="all", choices=["3", "all"])
    args = parser.parse_args()

    # 1) 加载配置并设置设备/随机种子。
    config = load_config(args.config, args.override if args.override else None)
    device = str(config.experiment.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead.")
        device = "cpu"
    config.experiment.device = device
    _set_seed(int(config.experiment.seed))
    if thread_overrides:
        print("Normalized thread env vars: " + ", ".join(f"{k}={v}" for k, v in thread_overrides.items()))

    # 2) 准备输出目录与 checkpoint 目录。
    output_dir = _build_output_dir(config)
    checkpoints_dir = Path(output_dir) / str(config.experiment.checkpoints_subdir or "checkpoints")
    fixed_checkpoints_dir = _build_fixed_checkpoint_dir(config)
    checkpoint_save_dirs = _dedupe_paths((checkpoints_dir, fixed_checkpoints_dir))
    for path in checkpoint_save_dirs:
        path.mkdir(parents=True, exist_ok=True)
    config.save(str(Path(output_dir) / "resolved_config.yaml"))

    tensorboard_dir = str(Path(output_dir) / "tensorboard") if bool(getattr(config.experiment.tensorboard, "enabled", False)) else None
    # 3) 构建模型与 trainer。
    prior_estimator, lens_encoder, odn, restoration_net = build_models_from_config(config, device)
    trainer = build_trainer_from_config(
        config=config,
        prior_estimator=prior_estimator,
        lens_table_encoder=lens_encoder,
        odn=odn,
        restoration_net=restoration_net,
        device=device,
        tensorboard_dir=tensorboard_dir,
    )
    # 4) 解析启动方式：resume 与 init-from 互斥。
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from cannot be used together.")
    run_start_step, run_end_step = _stage_window(config, args.stage)
    resume_path = args.resume
    init_from_path = args.init_from
    # 若用户未显式指定，尝试自动推断恢复路径。
    if resume_path is None and init_from_path is None:
        auto_resume, auto_init, reason = _resolve_auto_start_checkpoint(args.stage, fixed_checkpoints_dir)
        if auto_resume:
            resume_path = auto_resume
            print(f"Using {reason}: {resume_path}")
        elif auto_init:
            init_from_path = auto_init
            print(f"Using {reason}: {init_from_path}")
    start_step = run_start_step
    virtual_epoch = 0
    # 5) 先做 warm-start（仅权重），再做 resume（权重+优化器状态）。
    if init_from_path:
        info = trainer.load_checkpoint(init_from_path, load_optimizer=False)
        report = info.get("sanitization_report", {})
        if report and (report.get("removed_prior_keys") or report.get("removed_restoration_keys")):
            print(f"Warm-start stripped legacy keys: {summarize_removed_keys(report)}")
    if resume_path:
        info = trainer.load_checkpoint(resume_path, load_optimizer=True)
        report = info.get("sanitization_report", {})
        if report and (report.get("removed_prior_keys") or report.get("removed_restoration_keys")):
            print(f"Resume stripped legacy keys: {summarize_removed_keys(report)}")
        if isinstance(info.get("global_step"), int):
            start_step = max(run_start_step, int(info["global_step"]))
        if isinstance(info.get("epoch"), int):
            virtual_epoch = int(info["epoch"]) + 1

    # 6) 为三个阶段分别构建 train/val dataloader。
    schedule = config.training.stage_schedule
    loaders = {
        STAGE3: {
            "train": build_mixlib_dataloader(config, mode="train", batch_size_override=int(schedule.stage3_batch_size), stage_name="stage3"),
            "val": build_mixlib_dataloader(config, mode="val", batch_size_override=int(schedule.stage3_batch_size), stage_name="stage3"),
        },
    }
    evaluator = PerformanceEvaluator(device=device)
    # 7) 初始化 best/latest 状态跟踪。
    best_paths = _dedupe_paths((checkpoints_dir / BEST_PERFORMANCE_FILENAME, fixed_checkpoints_dir / BEST_PERFORMANCE_FILENAME))
    best_state = _load_best_performance(fixed_checkpoints_dir / BEST_PERFORMANCE_FILENAME)
    best_tracking = _build_best_tracking_state(config, best_state)
    _persist_best_performance(best_paths, best_state)
    save_interval = int(config.experiment.save_interval)
    validation_interval = int(getattr(config.experiment, "validation_interval", 0))
    global_step = int(start_step)
    last_stage = STAGE3
    last_val_metrics: Dict[str, Any] = {}

    print("=" * 60)
    print("BPFR-Net lens-table fusion training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Iteration window: {run_start_step} -> {run_end_step}")

    # 8) 主训练循环：按 global_step 驱动，跨阶段推进。
    while global_step < run_end_step:
        stage = STAGE3
        stage_end = run_end_step
        cycle_target = min(stage_end, run_end_step)
        # 当前小周期的截止点由“阶段结束/验证边界/保存边界”共同决定。
        if validation_interval > 0:
            cycle_target = min(cycle_target, _next_interval_boundary(global_step, validation_interval, cycle_target))
        if save_interval > 0:
            cycle_target = min(cycle_target, _next_interval_boundary(global_step, save_interval, cycle_target))
        print(f"\n{_stage_display_name(stage)}: step {global_step} -> {cycle_target}")
        # 8.1 训练一个小周期。
        train_metrics, global_step, last_batch = _train_one_cycle(trainer, loaders[stage]["train"], stage, global_step, cycle_target)
        trainer.log_to_tensorboard(train_metrics, global_step, prefix="train")
        _print_metric_table("Train metrics:", train_metrics)
        # 8.2 到达验证点时运行验证、更新 best，并按需保存 best checkpoint。
        should_validate = _is_interval_trigger(global_step, validation_interval) or global_step >= cycle_target or global_step >= run_end_step
        if should_validate:
            val_metrics = _validate_one_epoch(trainer, loaders[stage]["val"], stage, device, evaluator)
            trainer.log_to_tensorboard(val_metrics, global_step, prefix="val")
            last_val_metrics = val_metrics
            last_stage = stage
            _print_metric_table("Validation metrics:", val_metrics)
            trainer.update_best_metrics(val_metrics, stage)
            if _maybe_save_best_checkpoint(trainer, checkpoints_dir, virtual_epoch, stage, global_step, val_metrics, best_tracking):
                score_info = extract_stage_score(val_metrics, str(best_tracking[stage]["metric_name"]), stage)
                if score_info is not None:
                    display, value, maximize = score_info
                    best_state[stage] = {
                        "metric_name": display,
                        "metric_value": float(value),
                        "maximize": bool(maximize),
                        "epoch": int(virtual_epoch),
                        "global_step": int(global_step),
                        "checkpoint": str(fixed_checkpoints_dir / f"best_{_stage_tag(stage)}.pt"),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                _save_checkpoint_bundle(trainer, tuple(p for p in checkpoint_save_dirs if str(p) != str(checkpoints_dir)), ("best.pt", f"best_{_stage_tag(stage)}.pt"), virtual_epoch, stage, global_step, val_metrics)
        # 8.3 周期性保存 latest checkpoint。
        if _save_periodic_checkpoint(trainer, checkpoints_dir, virtual_epoch, stage, global_step, last_val_metrics, save_interval):
            _save_checkpoint_bundle(trainer, tuple(p for p in checkpoint_save_dirs if str(p) != str(checkpoints_dir)), ("latest.pt", f"latest_{_stage_tag(stage)}.pt"), virtual_epoch, stage, global_step, last_val_metrics)
        best_state["latest"] = {
            "stage": stage,
            "epoch": int(virtual_epoch),
            "global_step": int(global_step),
            "checkpoint": str(fixed_checkpoints_dir / f"latest_{_stage_tag(stage)}.pt"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _persist_best_performance(best_paths, best_state)
        # 8.4 可视化导出（按 epoch 触发）。
        if _should_export_visuals(config, virtual_epoch):
            _export_epoch_visuals(trainer, config, last_batch, stage, virtual_epoch, output_dir, device)
        virtual_epoch += 1

    # 9) 训练结束后保存 final/latest，并关闭 tensorboard writer。
    final_stage = STAGE3
    _save_checkpoint_bundle(trainer, checkpoint_save_dirs, ("final_model.pt", "latest.pt", f"latest_{_stage_tag(final_stage)}.pt"), max(0, virtual_epoch - 1), final_stage, global_step, last_val_metrics)
    trainer.close_tensorboard()
    print("\nTraining finished.")
    print(f"Final checkpoint: {checkpoints_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
