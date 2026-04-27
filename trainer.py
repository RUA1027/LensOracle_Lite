"""三阶段 BPFR-Net 训练器。

ThreeStageTrainer 将新版 lens-table fusion 流程拆成三个互斥训练阶段：
1. Stage 1 只训练 PriorEstimator；
2. Stage 2 冻结 PriorEstimator，训练 ODN；
3. Stage 3 冻结 PriorEstimator/ODN，训练 LensTableEncoder 和 Restoration。

所有 loss 都在各自阶段内部计算，避免旧版像素级 prior map 与新 lens-table
监督混用。
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.checkpoint_sanitizer import sanitize_legacy_checkpoint


STAGE1 = "stage1_prior"
STAGE2 = "stage2_odn"
STAGE3 = "stage3_restoration"


def lens_table_tv_loss(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 lens-table 在 r 和 theta 维的 TV loss。

    table 形状为 `[B, 64, 48, 67]`。r 方向是普通相邻差分；theta 方向是
    circular 差分，最后一个角度会和第一个角度相连。
    """

    tv_r = torch.abs(table[:, 1:, :, :] - table[:, :-1, :, :]).mean()
    tv_theta = torch.abs(torch.roll(table, shifts=-1, dims=2) - table).mean()
    return tv_r + tv_theta, tv_r, tv_theta


def _get_stage_schedule_value(stage_schedule: Any, key: str, default: int) -> int:
    """从 stage_schedule 里读取整数配置并带默认值回退。"""

    return int(getattr(stage_schedule, key, default))


class ThreeStageTrainer:
    """新版三阶段训练器。

    职责边界：
    1) 管理三阶段参数可训练性与 loss 计算；
    2) 管理四路优化器/调度器与梯度累积；
    3) 管理 AMP、梯度裁剪与 non-finite 防护；
    4) 维护 checkpoint 与 tensorboard 相关状态。
    """

    def __init__(
        self,
        prior_estimator: nn.Module,
        lens_table_encoder: nn.Module,
        odn: nn.Module,
        restoration_net: nn.Module,
        lr_prior: float,
        lr_lens_encoder: float,
        lr_odn: float,
        lr_restoration: float,
        optimizer_type: str,
        weight_decay: float,
        grad_clip_prior: float,
        grad_clip_lens_encoder: float,
        grad_clip_odn: float,
        grad_clip_restoration: float,
        stage_schedule: Any,
        use_amp: bool,
        amp_dtype: str,
        accumulation_steps: int,
        device: str,
        tensorboard_dir: Optional[str],
        perceptual_weight: float = 0.0,
        perceptual_warmup_iterations: int = 0,
        perceptual_enabled: bool = True,
        perceptual_loss: Optional[nn.Module] = None,
        perceptual_loss_builder: Optional[Callable[[], nn.Module]] = None,
        charbonnier_loss: Optional[nn.Module] = None,
        charbonnier_enabled: bool = True,
        ms_ssim_loss: Optional[nn.Module] = None,
        ms_ssim_loss_builder: Optional[Callable[[], nn.Module]] = None,
        ms_ssim_weight: float = 0.0,
        ms_ssim_enabled: bool = False,
        tv_weight: float = 0.01,
        odn_loss_weight: float = 0.5,
        train_prior_mode: str = "correct_gt",
        eval_prior_mode: str = "correct_gt",
        lens_encoder_enabled: bool = True,
        ablation: Optional[Dict[str, Any]] = None,
        nonfinite_patience: int = 3,
        nonfinite_backoff_factor: float = 0.5,
        nonfinite_min_lr: float = 1.0e-6,
    ):
        self.prior_estimator = prior_estimator
        self.lens_table_encoder = lens_table_encoder
        self.odn = odn
        self.restoration_net = restoration_net
        self.device = device
        self.stage_schedule = stage_schedule
        self.stage1_iterations = _get_stage_schedule_value(stage_schedule, "stage1_iterations", 0)
        self.stage2_iterations = _get_stage_schedule_value(stage_schedule, "stage2_iterations", 0)
        self.stage3_iterations = _get_stage_schedule_value(stage_schedule, "stage3_iterations", 0)
        self.total_iterations = self.stage1_iterations + self.stage2_iterations + self.stage3_iterations
        self.accumulation_steps = max(1, int(accumulation_steps))

        self.grad_clip_prior = float(grad_clip_prior)
        self.grad_clip_lens_encoder = float(grad_clip_lens_encoder)
        self.grad_clip_odn = float(grad_clip_odn)
        self.grad_clip_restoration = float(grad_clip_restoration)
        self.tv_weight = float(tv_weight)
        self.odn_loss_weight = float(odn_loss_weight)
        self.train_prior_mode = str(train_prior_mode)
        self.eval_prior_mode = str(eval_prior_mode)
        self.lens_encoder_enabled = bool(lens_encoder_enabled)
        self.ablation = dict(ablation or {})
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_warmup_iterations = max(0, int(perceptual_warmup_iterations))
        self.perceptual_enabled = bool(perceptual_enabled)
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss_builder = perceptual_loss_builder
        self.charbonnier_loss = charbonnier_loss
        self.charbonnier_enabled = bool(charbonnier_enabled)
        self.ms_ssim_loss = ms_ssim_loss
        self.ms_ssim_loss_builder = ms_ssim_loss_builder
        self.ms_ssim_weight = float(ms_ssim_weight)
        self.ms_ssim_enabled = bool(ms_ssim_enabled)
        self.nonfinite_patience = max(1, int(nonfinite_patience))
        self.nonfinite_backoff_factor = float(nonfinite_backoff_factor)
        self.nonfinite_min_lr = float(nonfinite_min_lr)

        self.use_amp = bool(use_amp and device.startswith("cuda") and torch.cuda.is_available())
        self.amp_dtype = torch.float16 if str(amp_dtype).lower() == "float16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.optimizer_prior = (
            self._build_optimizer(self.prior_estimator.parameters(), optimizer_type, lr_prior, weight_decay)
            if self.stage1_iterations > 0
            else None
        )
        self.optimizer_lens_encoder = (
            self._build_optimizer(
                self.lens_table_encoder.parameters(),
                optimizer_type,
                lr_lens_encoder,
                weight_decay,
            )
            if self.lens_encoder_enabled
            else None
        )
        self.optimizer_odn = (
            self._build_optimizer(self.odn.parameters(), optimizer_type, lr_odn, weight_decay)
            if self.stage2_iterations > 0
            else None
        )
        self.optimizer_restoration = self._build_optimizer(
            self.restoration_net.parameters(), optimizer_type, lr_restoration, weight_decay
        )
        self.scheduler_prior = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_prior, T_max=max(1, self.stage1_iterations), eta_min=self.nonfinite_min_lr
            )
            if self.optimizer_prior is not None
            else None
        )
        self.scheduler_lens_encoder = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_lens_encoder,
                T_max=max(1, self.stage3_iterations),
                eta_min=self.nonfinite_min_lr,
            )
            if self.optimizer_lens_encoder is not None
            else None
        )
        self.scheduler_odn = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_odn, T_max=max(1, self.stage2_iterations), eta_min=self.nonfinite_min_lr
            )
            if self.optimizer_odn is not None
            else None
        )
        self.scheduler_restoration = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_restoration, T_max=max(1, self.stage3_iterations), eta_min=self.nonfinite_min_lr
            )
            if self.optimizer_restoration is not None
            else None
        )

        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=tensorboard_dir)
            except Exception:
                self.writer = None

        self.best_metrics: Dict[str, Dict[str, float]] = {
            STAGE1: {"val_psf_sfr_l1": float("inf"), "val_loss": float("inf")},
            STAGE2: {"val_odn_l1": float("inf"), "val_loss": float("inf")},
            STAGE3: {"psnr": float("-inf"), "val_loss": float("inf")},
        }
        self.prior_optimizer_steps = 0
        self.lens_encoder_optimizer_steps = 0
        self.odn_optimizer_steps = 0
        self.restoration_optimizer_steps = 0
        self._accum_step = 0
        self._last_step_nonfinite = False
        self.nonfinite_streak = 0
        self.lr_backoff_events = 0

    @property
    def pending_accumulation_steps(self) -> int:
        """当前尚未 flush 的累积步数。"""

        return int(self._accum_step)

    @staticmethod
    def _build_optimizer(params, optimizer_type: str, lr: float, weight_decay: float):
        """按配置创建优化器（Adam 或 AdamW）。"""

        params = list(params)
        if not params:
            return None
        if str(optimizer_type).lower() == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        """批量设置模块参数的 requires_grad。"""

        for param in module.parameters():
            param.requires_grad = enabled

    def _set_stage_trainability(self, stage: str) -> None:
        """根据阶段切换模块训练开关。

        - Stage1：仅 prior 可训练；
        - Stage2：仅 ODN 可训练；
        - Stage3：仅 lens_encoder + restoration 可训练。
        """

        self._set_requires_grad(self.prior_estimator, stage == STAGE1)
        self._set_requires_grad(self.lens_table_encoder, stage == STAGE3 and self.lens_encoder_enabled)
        self._set_requires_grad(self.odn, stage == STAGE2)
        self._set_requires_grad(self.restoration_net, stage == STAGE3)

        self.prior_estimator.train(stage == STAGE1)
        self.lens_table_encoder.train(stage == STAGE3 and self.lens_encoder_enabled)
        self.odn.train(stage == STAGE2)
        self.restoration_net.train(stage == STAGE3)

    def _zero_all_grad(self) -> None:
        """清空四路优化器梯度。"""

        for optimizer in (
            self.optimizer_prior,
            self.optimizer_lens_encoder,
            self.optimizer_odn,
            self.optimizer_restoration,
        ):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

    def _maybe_zero_grad(self) -> None:
        """仅在累积起点清梯度，兼容 gradient accumulation。"""

        if self._accum_step == 0:
            self._zero_all_grad()

    def get_stage(self, progress_step: int) -> str:
        """按全局进度步数返回所属阶段。"""

        if progress_step < self.stage1_iterations:
            return STAGE1
        if progress_step < self.stage1_iterations + self.stage2_iterations:
            return STAGE2
        return STAGE3

    def get_stage_by_iteration(self, optimizer_step: int) -> str:
        """语义别名：根据优化器步数推断阶段。"""

        return self.get_stage(int(optimizer_step))

    def _current_perceptual_weight(self, stage: str) -> float:
        """计算当前 step 的感知损失权重（含 warmup）。"""

        if stage != STAGE3 or not self.perceptual_enabled or self.perceptual_weight <= 0.0:
            return 0.0
        if self.perceptual_warmup_iterations <= 0:
            return self.perceptual_weight
        progress = min(1.0, self.restoration_optimizer_steps / float(self.perceptual_warmup_iterations))
        return self.perceptual_weight * progress

    def _get_perceptual_loss(self) -> Optional[nn.Module]:
        """延迟构建 perceptual loss（首次需要时实例化）。"""

        if not self.perceptual_enabled or self.perceptual_weight <= 0.0:
            return None
        if self.perceptual_loss is not None:
            return self.perceptual_loss
        if self.perceptual_loss_builder is None:
            return None
        self.perceptual_loss = self.perceptual_loss_builder()
        self.perceptual_loss_builder = None
        return self.perceptual_loss

    def _get_ms_ssim_loss(self) -> Optional[nn.Module]:
        """延迟构建 MS-SSIM loss。"""

        if not self.ms_ssim_enabled or self.ms_ssim_weight <= 0.0:
            return None
        if self.ms_ssim_loss is not None:
            return self.ms_ssim_loss
        if self.ms_ssim_loss_builder is None:
            return None
        self.ms_ssim_loss = self.ms_ssim_loss_builder()
        self.ms_ssim_loss_builder = None
        return self.ms_ssim_loss

    @staticmethod
    def _is_finite_scalar(value: Any) -> bool:
        """判断标量或标量张量是否有限。"""

        if torch.is_tensor(value):
            return bool(value.numel() > 0 and torch.isfinite(value).all().item())
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _stage1_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Stage 1 损失：prior L1 + TV 正则。"""

        blur = batch["blur"].to(self.device, non_blocking=True)
        gt = batch["gt_psf_sfr"].to(self.device, non_blocking=True)
        pred = self.prior_estimator(blur)
        loss_l1 = F.l1_loss(pred, gt)
        tv, tv_r, tv_theta = lens_table_tv_loss(pred)
        total = loss_l1 + self.tv_weight * tv
        zero = total.new_zeros(())
        return {
            "loss": total,
            "loss_table_l1": loss_l1,
            "loss_tv": tv,
            "loss_tv_r": tv_r,
            "loss_tv_theta": tv_theta,
            "loss_table_recon": zero,
            "loss_odn": zero,
            "loss_odn_weighted": zero,
            "loss_restoration": zero,
            "loss_perceptual": zero,
            "loss_ms_ssim": zero,
            "perceptual_weight_current": zero,
            "ms_ssim_weight_current": zero,
        }

    def _stage2_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Stage 2 损失：ODN 重建 blur 的 L1。

        prior 在该阶段冻结，仅作为条件输入；同时记录 pred 的诊断指标（不反传）。
        """

        blur = batch["blur"].to(self.device, non_blocking=True)
        sharp = batch["sharp"].to(self.device, non_blocking=True)
        gt = batch["gt_psf_sfr"].to(self.device, non_blocking=True)
        with torch.no_grad():
            pred = self.prior_estimator(blur)
        simulated = self.odn(sharp, pred)

        tv, tv_r, tv_theta = lens_table_tv_loss(pred)
        odn_loss = F.l1_loss(simulated, blur)
        pred_l1 = F.l1_loss(pred, gt)
        total = self.odn_loss_weight * odn_loss
        zero = total.new_zeros(())
        return {
            "loss": total,
            "loss_table_l1": pred_l1.detach(),
            "loss_tv": tv.detach(),
            "loss_tv_r": tv_r,
            "loss_tv_theta": tv_theta,
            "loss_table_recon": zero,
            "loss_odn": odn_loss,
            "loss_odn_weighted": total,
            "loss_restoration": zero,
            "loss_perceptual": zero,
            "loss_ms_ssim": zero,
            "perceptual_weight_current": zero,
            "ms_ssim_weight_current": zero,
        }

    def _resolve_prior_table(self, batch: Dict[str, Any], mode: str) -> Optional[torch.Tensor]:
        """Resolve the Stage3 prior table from batch data according to ablation mode."""

        if mode == "none":
            return None
        key_by_mode = {
            "correct_gt": "gt_psf_sfr",
            "incorrect_gt": "incorrect_gt_psf_sfr",
        }
        key = key_by_mode.get(str(mode))
        if key is None:
            raise ValueError(f"Unsupported prior mode: {mode}")
        table = batch.get(key)
        if not torch.is_tensor(table):
            raise KeyError(f"Batch is missing required prior tensor '{key}'.")
        return table.to(self.device, non_blocking=True)

    def _stage3_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Stage 3 损失：restoration 主损失 + 可选 perceptual/MS-SSIM。"""

        blur = batch["blur"].to(self.device, non_blocking=True)
        sharp = batch["sharp"].to(self.device, non_blocking=True)
        crop_info = batch.get("crop_info")
        if torch.is_tensor(crop_info):
            crop_info = crop_info.to(self.device, non_blocking=True)
        # Stage3 中 prior 固定，只提供条件先验，不参与梯度更新。
        prior_table = self._resolve_prior_table(batch, self.train_prior_mode)
        features = self.lens_table_encoder(prior_table) if self.lens_encoder_enabled and prior_table is not None else None
        restored = self.restoration_net(blur, features, crop_info=crop_info)
        # base restoration loss（Charbonnier 优先，否则退化为 L1）。
        if self.charbonnier_enabled and self.charbonnier_loss is not None:
            restoration_loss = self.charbonnier_loss(restored, sharp)
        else:
            restoration_loss = F.l1_loss(restored, sharp)
        total = restoration_loss
        zero = total.new_zeros(())

        perc_weight = self._current_perceptual_weight(STAGE3)
        loss_perceptual = zero
        perceptual_module = self._get_perceptual_loss()
        if perceptual_module is not None and perc_weight > 0.0:
            loss_perceptual = perceptual_module(restored, sharp)
            total = total + perc_weight * loss_perceptual

        loss_ms_ssim = zero
        ms_ssim_module = self._get_ms_ssim_loss()
        if ms_ssim_module is not None and self.ms_ssim_weight > 0.0:
            loss_ms_ssim = ms_ssim_module(restored, sharp)
            total = total + self.ms_ssim_weight * loss_ms_ssim

        return {
            "loss": total,
            "loss_table_l1": zero,
            "loss_tv": zero,
            "loss_tv_r": zero,
            "loss_tv_theta": zero,
            "loss_table_recon": zero,
            "loss_odn": zero,
            "loss_odn_weighted": zero,
            "loss_restoration": restoration_loss,
            "loss_perceptual": loss_perceptual,
            "loss_ms_ssim": loss_ms_ssim,
            "perceptual_weight_current": torch.full_like(zero, perc_weight),
            "ms_ssim_weight_current": torch.full_like(zero, self.ms_ssim_weight if self.ms_ssim_enabled else 0.0),
        }

    def _forward_and_losses(self, batch: Dict[str, Any], stage: str) -> Dict[str, torch.Tensor]:
        """按阶段路由到对应 loss 计算函数。"""

        if stage == STAGE1:
            return self._stage1_losses(batch)
        if stage == STAGE2:
            return self._stage2_losses(batch)
        if stage == STAGE3:
            return self._stage3_losses(batch)
        raise ValueError(f"Unknown training stage: {stage}")

    def _active_optimizers(self, stage: str):
        """返回当前阶段应参与 step 的优化器集合。"""

        if stage == STAGE1:
            if self.optimizer_prior is None or self.scheduler_prior is None:
                return []
            return [(self.optimizer_prior, self.scheduler_prior, self.prior_estimator.parameters(), self.grad_clip_prior, "prior")]
        if stage == STAGE2:
            if self.optimizer_odn is None or self.scheduler_odn is None:
                return []
            return [(self.optimizer_odn, self.scheduler_odn, self.odn.parameters(), self.grad_clip_odn, "odn")]
        if stage == STAGE3:
            active = []
            if self.lens_encoder_enabled and self.optimizer_lens_encoder is not None and self.scheduler_lens_encoder is not None:
                active.append(
                    (
                        self.optimizer_lens_encoder,
                        self.scheduler_lens_encoder,
                        self.lens_table_encoder.parameters(),
                        self.grad_clip_lens_encoder,
                        "lens_encoder",
                    )
                )
            if self.optimizer_restoration is not None and self.scheduler_restoration is not None:
                active.append(
                    (
                        self.optimizer_restoration,
                        self.scheduler_restoration,
                        self.restoration_net.parameters(),
                        self.grad_clip_restoration,
                        "restoration",
                    )
                )
            return active
        return []

    def _increment_step_counter(self, name: str) -> None:
        """更新各子模块优化器 step 计数器。"""

        if name == "prior":
            self.prior_optimizer_steps += 1
        elif name == "lens_encoder":
            self.lens_encoder_optimizer_steps += 1
        elif name == "odn":
            self.odn_optimizer_steps += 1
        elif name == "restoration":
            self.restoration_optimizer_steps += 1

    def _clip_and_step(self, stage: str) -> bool:
        """执行一次“裁剪 + optimizer.step + scheduler.step”。

        返回值：
        - True：本次成功完成参数更新；
        - False：因 non-finite 等原因跳过更新。
        """

        self._last_step_nonfinite = False
        active = self._active_optimizers(stage)
        if not active:
            return False

        if self.use_amp:
            # AMP 路径：先 unscale，再裁剪，再 step。
            for optimizer, _, _, _, _ in active:
                self.scaler.unscale_(optimizer)
            finite = True
            for _, _, params, clip_value, _ in active:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_value)
                finite = finite and self._is_finite_scalar(grad_norm)
            if not finite:
                self._last_step_nonfinite = True
                self._zero_all_grad()
                return False
            scale_before = float(self.scaler.get_scale())
            for optimizer, _, _, _, _ in active:
                self.scaler.step(optimizer)
            self.scaler.update()
            if float(self.scaler.get_scale()) < scale_before:
                self._last_step_nonfinite = True
                self._zero_all_grad()
                return False
        else:
            # 非 AMP 路径：直接裁剪 + step。
            for _, _, params, clip_value, _ in active:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_value)
                if not self._is_finite_scalar(grad_norm):
                    self._last_step_nonfinite = True
                    self._zero_all_grad()
                    return False
            for optimizer, _, _, _, _ in active:
                optimizer.step()

        for _, scheduler, _, _, name in active:
            scheduler.step()
            self._increment_step_counter(name)
        return True

    def _apply_lr_backoff(self, stage: str) -> bool:
        """在连续 non-finite 后按比例衰减当前阶段学习率。"""

        if self.nonfinite_backoff_factor <= 0.0 or self.nonfinite_backoff_factor >= 1.0:
            return False
        changed = False
        for optimizer, scheduler, _, _, _ in self._active_optimizers(stage):
            for group in optimizer.param_groups:
                current = float(group.get("lr", 0.0))
                target = max(self.nonfinite_min_lr, current * self.nonfinite_backoff_factor)
                if target + 1.0e-15 < current:
                    group["lr"] = target
                    changed = True
            base_lrs = getattr(scheduler, "base_lrs", None)
            if isinstance(base_lrs, list):
                scheduler.base_lrs = [max(self.nonfinite_min_lr, lr * self.nonfinite_backoff_factor) for lr in base_lrs]
        if changed:
            self.lr_backoff_events += 1
        return changed

    def _handle_nonfinite_event(self, stage: str) -> float:
        """处理 non-finite 事件并按 patience 决定是否触发 lr backoff。"""

        self.nonfinite_streak += 1
        if self.nonfinite_streak < self.nonfinite_patience:
            return 0.0
        self.nonfinite_streak = 0
        return 1.0 if self._apply_lr_backoff(stage) else 0.0

    def train_step(
        self,
        batch: Dict[str, Any],
        epoch: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, float]:
        """执行单个训练 step（支持梯度累积）。

        返回字典除 loss 外还包含：
        - `optimizer_step`：本次是否发生参数更新；
        - `skipped_nonfinite`：是否因 non-finite 被跳过；
        - `lr_backoff_event`：是否触发学习率回退。
        """

        if stage is None:
            stage = self.get_stage(0 if epoch is None else int(epoch))
        self._set_stage_trainability(stage)
        self._maybe_zero_grad()

        # 先走 AMP 前向；若 loss 非有限，再用全精度重算一次做兜底。
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            losses = self._forward_and_losses(batch, stage)

        if self.use_amp and not self._is_finite_scalar(losses["loss"]):
            with torch.cuda.amp.autocast(enabled=False):
                losses = self._forward_and_losses(batch, stage)

        if not self._is_finite_scalar(losses["loss"]):
            self._accum_step = 0
            self._zero_all_grad()
            output = {key: float(value.detach().item()) for key, value in losses.items()}
            output["optimizer_step"] = 0.0
            output["skipped_nonfinite"] = 1.0
            output["lr_backoff_event"] = self._handle_nonfinite_event(stage)
            output["nonfinite_streak"] = float(self.nonfinite_streak)
            return output

        loss_for_backward = losses["loss"] / float(self.accumulation_steps)
        if self.use_amp:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        optimizer_step = False
        self._accum_step += 1
        if self._accum_step >= self.accumulation_steps:
            optimizer_step = self._clip_and_step(stage)
            self._accum_step = 0

        skipped_nonfinite = 0.0
        lr_backoff_event = 0.0
        if self._last_step_nonfinite:
            skipped_nonfinite = 1.0
            lr_backoff_event = self._handle_nonfinite_event(stage)
        elif optimizer_step:
            self.nonfinite_streak = 0

        output = {key: float(value.detach().item()) for key, value in losses.items()}
        output["optimizer_step"] = 1.0 if optimizer_step else 0.0
        output["skipped_nonfinite"] = skipped_nonfinite
        output["lr_backoff_event"] = lr_backoff_event
        output["nonfinite_streak"] = float(self.nonfinite_streak)
        return output

    def flush_pending_gradients(self, epoch: Optional[int] = None, stage: Optional[str] = None) -> bool:
        """强制消费残余累积梯度。"""

        if self._accum_step <= 0:
            return False
        if stage is None:
            stage = self.get_stage(0 if epoch is None else int(epoch))
        stepped = self._clip_and_step(stage)
        self._accum_step = 0
        return stepped

    def reset_after_oom(self) -> bool:
        """在 OOM/异常后重置累积状态并清梯度。"""

        had_pending = self._accum_step > 0
        self._accum_step = 0
        self._zero_all_grad()
        return had_pending

    def get_current_lr(self) -> Dict[str, float]:
        """返回四路优化器当前学习率快照。"""

        lrs: Dict[str, float] = {}
        for name, optimizer in (
            ("prior", self.optimizer_prior),
            ("lens_encoder", self.optimizer_lens_encoder),
            ("odn", self.optimizer_odn),
            ("restoration", self.optimizer_restoration),
        ):
            if optimizer is not None:
                lrs[name] = float(optimizer.param_groups[0]["lr"])
        return lrs

    def update_best_metrics(self, val_metrics: Dict[str, Any], stage: str) -> Dict[str, bool]:
        """更新内存中的 best 指标缓存并返回命中标记。"""

        flags = {"val_psf_sfr_l1": False, "val_odn_l1": False, "psnr": False, "val_loss": False}
        if stage == STAGE1:
            metric = float(val_metrics.get("Val_PSF_SFR_L1", float("nan")))
            if math.isfinite(metric) and metric < self.best_metrics[STAGE1]["val_psf_sfr_l1"]:
                self.best_metrics[STAGE1]["val_psf_sfr_l1"] = metric
                self.best_metrics[STAGE1]["val_loss"] = metric
                flags["val_psf_sfr_l1"] = True
        elif stage == STAGE2:
            metric = float(val_metrics.get("Val_ODN_L1", float("nan")))
            if math.isfinite(metric) and metric < self.best_metrics[STAGE2]["val_odn_l1"]:
                self.best_metrics[STAGE2]["val_odn_l1"] = metric
                self.best_metrics[STAGE2]["val_loss"] = metric
                flags["val_odn_l1"] = True
        elif stage == STAGE3:
            metric = float(val_metrics.get("PSNR", float("nan")))
            if math.isfinite(metric) and metric > self.best_metrics[STAGE3]["psnr"]:
                self.best_metrics[STAGE3]["psnr"] = metric
                flags["psnr"] = True
            mae = float(val_metrics.get("MAE", float("nan")))
            if math.isfinite(mae) and mae < self.best_metrics[STAGE3]["val_loss"]:
                self.best_metrics[STAGE3]["val_loss"] = mae
                flags["val_loss"] = True
        return flags

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        stage: str,
        val_metrics: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """序列化保存完整训练状态。"""

        checkpoint = {
            "epoch": int(epoch),
            "stage": stage,
            "ablation": self.ablation,
            "val_metrics": val_metrics if val_metrics is not None else {},
            "best_metrics": self.best_metrics,
            "accum_step": int(self._accum_step),
            "global_step": None if global_step is None else int(global_step),
            "prior_optimizer_steps": int(self.prior_optimizer_steps),
            "lens_encoder_optimizer_steps": int(self.lens_encoder_optimizer_steps),
            "odn_optimizer_steps": int(self.odn_optimizer_steps),
            "restoration_optimizer_steps": int(self.restoration_optimizer_steps),
            "lens_table_encoder": self.lens_table_encoder.state_dict(),
            "restoration_net": self.restoration_net.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
        }
        if self.optimizer_lens_encoder is not None:
            checkpoint["optimizer_lens_encoder"] = self.optimizer_lens_encoder.state_dict()
        if self.optimizer_restoration is not None:
            checkpoint["optimizer_restoration"] = self.optimizer_restoration.state_dict()
        if self.scheduler_lens_encoder is not None:
            checkpoint["scheduler_lens_encoder"] = self.scheduler_lens_encoder.state_dict()
        if self.scheduler_restoration is not None:
            checkpoint["scheduler_restoration"] = self.scheduler_restoration.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """加载 checkpoint 到当前 trainer。

        - `load_optimizer=False` 用于 warm-start；
        - `load_optimizer=True` 用于断点续训。
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "restoration_net" not in checkpoint:
            raise ValueError("Expected a Stage3 restoration checkpoint containing restoration_net.")
        checkpoint, sanitization_report = sanitize_legacy_checkpoint(checkpoint)

        if "lens_table_encoder" in checkpoint:
            self.lens_table_encoder.load_state_dict(checkpoint.get("lens_table_encoder", {}), strict=False)
        self.restoration_net.load_state_dict(checkpoint.get("restoration_net", {}), strict=False)
        self._accum_step = 0
        self._zero_all_grad()

        if load_optimizer:
            for name, optimizer in (
                ("optimizer_lens_encoder", self.optimizer_lens_encoder),
                ("optimizer_restoration", self.optimizer_restoration),
            ):
                if name in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint[name])
            for name, scheduler in (
                ("scheduler_lens_encoder", self.scheduler_lens_encoder),
                ("scheduler_restoration", self.scheduler_restoration),
            ):
                if name in checkpoint and scheduler is not None:
                    try:
                        scheduler.load_state_dict(checkpoint[name])
                    except Exception:
                        pass
            if self.use_amp and checkpoint.get("scaler") is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.prior_optimizer_steps = int(checkpoint.get("prior_optimizer_steps", 0))
        self.lens_encoder_optimizer_steps = int(checkpoint.get("lens_encoder_optimizer_steps", 0))
        self.odn_optimizer_steps = int(checkpoint.get("odn_optimizer_steps", 0))
        self.restoration_optimizer_steps = int(checkpoint.get("restoration_optimizer_steps", 0))
        if isinstance(checkpoint.get("best_metrics"), dict):
            for stage_name, metrics in checkpoint["best_metrics"].items():
                if stage_name in self.best_metrics and isinstance(metrics, dict):
                    self.best_metrics[stage_name].update(metrics)

        return {
            "epoch": checkpoint.get("epoch"),
            "stage": checkpoint.get("stage"),
            "global_step": checkpoint.get("global_step"),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "sanitization_report": sanitization_report,
        }

    def log_to_tensorboard(self, metrics: Dict[str, Any], step: int, prefix: str = "train") -> None:
        """把标量指标写入 TensorBoard。"""

        if self.writer is None:
            return
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if math.isnan(scalar) or math.isinf(scalar):
                continue
            self.writer.add_scalar(f"{prefix}/{key}", scalar, step)

    def close_tensorboard(self) -> None:
        """关闭 TensorBoard writer。"""

        if self.writer is not None:
            self.writer.close()
            self.writer = None
