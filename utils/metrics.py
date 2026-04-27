"""BPFR-Net 三阶段评估指标集合。

设计目标：
1) Stage 1 在 lens-table 空间评估 prior 预测质量；
2) Stage 2 在 ODN 闭环空间评估 sharp->blur 退化重建能力；
3) Stage 3 在图像空间评估恢复质量（PSNR/SSIM/MAE/LPIPS）。

本模块仅负责评估，不参与任何训练反向传播或参数更新。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from trainer import STAGE1, STAGE2, lens_table_tv_loss


def resolve_stage_metric_spec(metric_name: str, stage: str) -> Tuple[str, Tuple[str, ...], bool]:
    """Resolve a configured metric name to canonical keys and comparison direction."""

    normalized = str(metric_name).strip().lower().replace("-", "_")
    if normalized in {"val_psf_sfr_l1", "psf_sfr_l1", "val_prior_l1", "l1"}:
        return "Val_PSF_SFR_L1", ("Val_PSF_SFR_L1", "val_psf_sfr_l1", "val_loss", "loss"), False
    if normalized in {"val_psf_sfr_msssim", "msssim", "ms_ssim"}:
        return "Val_PSF_SFR_MSSSIM", ("Val_PSF_SFR_MSSSIM", "val_psf_sfr_msssim"), True
    if normalized in {"val_lensidentifiability", "val_lens_identifiability", "lens_identifiability"}:
        return "Val_LensIdentifiability", ("Val_LensIdentifiability",), True
    if normalized in {"val_odn_l1", "odn_l1"}:
        return "Val_ODN_L1", ("Val_ODN_L1", "val_odn_l1", "val_loss", "loss"), False
    if normalized in {"val_odn_psnr", "odn_psnr"}:
        return "Val_ODN_PSNR", ("Val_ODN_PSNR", "val_odn_psnr"), True
    if normalized == "psnr":
        return "PSNR", ("PSNR", "psnr"), True
    if normalized == "ssim":
        return "SSIM", ("SSIM", "ssim"), True
    if normalized == "lpips":
        return "LPIPS", ("LPIPS", "lpips"), False
    if normalized in {"mae", "val_loss"}:
        return "MAE" if normalized == "mae" else "val_loss", ("val_loss", "MAE", "mae", "loss"), False
    if stage == STAGE1:
        return "Val_PSF_SFR_L1", ("Val_PSF_SFR_L1", "val_loss"), False
    if stage == STAGE2:
        return "Val_ODN_L1", ("Val_ODN_L1", "val_loss"), False
    return "PSNR", ("PSNR", "psnr"), True


def get_numeric_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """Extract a finite numeric metric with case-insensitive key fallback."""

    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    lowered = key.lower()
    for metric_key, metric_value in metrics.items():
        if str(metric_key).lower() == lowered and isinstance(metric_value, (int, float)):
            scalar = float(metric_value)
            if math.isfinite(scalar):
                return scalar
    return None


def extract_stage_score(metrics: Dict[str, Any], metric_name: str, stage: str) -> Optional[Tuple[str, float, bool]]:
    """Extract the comparable score for a training stage."""

    display_name, candidates, maximize = resolve_stage_metric_spec(metric_name, stage)
    for candidate in candidates:
        value = get_numeric_metric(metrics, candidate)
        if value is not None:
            return display_name, value, maximize
    return None


class PerformanceEvaluator:
    """统一评估器。

该类封装了三阶段评估中可复用的底层指标函数、模型模式切换、
以及 full-resolution 逐样本统计逻辑。
    """

    def __init__(self, device: str = "cuda", ssim_window: int = 11, ssim_sigma: float = 1.5):
        """初始化评估器。

        参数：
        - device: 指标计算所在设备；
        - ssim_window: SSIM 高斯窗口大小上限；
        - ssim_sigma: SSIM 高斯核标准差。
        """

        self.device = device
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self._lpips = None
        self._lpips_available: bool | None = None

    @staticmethod
    def _psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1.0e-8) -> torch.Tensor:
        """计算 PSNR。

        使用 MSE 形式：PSNR = 10 * log10(MAX^2 / (MSE + eps))。
        """

        mse = F.mse_loss(x, y)
        return 10.0 * torch.log10(max_val**2 / (mse + eps))

    @staticmethod
    def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构造 SSIM 使用的二维高斯窗口核。

        输出 shape 为 `[C, 1, K, K]`，便于直接用于 depthwise conv2d。
        """

        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.expand(channels, 1, size, size).contiguous()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """计算结构相似度 SSIM。

        为了在小尺寸 patch 上保持健壮性：
        - 会动态收缩窗口；
        - 当窗口过小（<3）时退化为 `1 / (1 + L1)` 近似分数。
        """

        channels = int(x.shape[1])
        window_size = min(self.ssim_window, int(x.shape[-1]), int(x.shape[-2]))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype) / (1.0 + F.l1_loss(x, y))
        window = self._gaussian_window(window_size, self.ssim_sigma, channels, x.device, x.dtype)
        padding = window_size // 2
        mu_x = F.conv2d(x, window, padding=padding, groups=channels)
        mu_y = F.conv2d(y, window, padding=padding, groups=channels)
        mu_x2 = mu_x.square()
        mu_y2 = mu_y.square()
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
        sigma_y = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
        )
        return ssim_map.mean()

    def _ensure_lpips_loaded(self) -> None:
        """懒加载 LPIPS 模型。

        仅首次调用时尝试导入与构建；若不可用则缓存失败状态，
        后续直接返回，避免重复 import 开销。
        """

        if self._lpips_available is not None:
            return
        try:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_available = True
        except Exception:
            self._lpips = None
            self._lpips_available = False

    def _lpips_score(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        """计算 LPIPS 感知距离。

        约束：当输入空间尺寸过小（<64）时返回 None，避免模型不稳定。
        """

        if min(int(x.shape[-2]), int(x.shape[-1])) < 64:
            return None
        self._ensure_lpips_loaded()
        if not self._lpips_available or self._lpips is None:
            return None
        return self._lpips(x * 2.0 - 1.0, y * 2.0 - 1.0).mean()

    @staticmethod
    def _count_parameters(*models: Optional[nn.Module]) -> float:
        """统计一个或多个模型的总参数量（单位：M）。"""

        params = 0
        for model in models:
            if model is None:
                continue
            params += sum(int(p.numel()) for p in model.parameters())
        return float(params) / 1.0e6

    @staticmethod
    def _sanitize_image_tensor(x: torch.Tensor) -> torch.Tensor:
        """把图像张量裁剪为可评估范围并清理非有限值。"""

        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    @staticmethod
    def _compute_mae(x: torch.Tensor, y: torch.Tensor) -> float:
        """计算 MAE（L1 均值）。"""

        return float(F.l1_loss(x, y).item())

    @staticmethod
    def _accumulate_if_finite(total: float, count: int, value: float) -> Tuple[float, int]:
        """仅累积有限值，过滤 NaN/Inf。"""

        if math.isfinite(value):
            return total + value, count + 1
        return total, count

    def _aggregate_metric_list(self, values: Iterable[float]) -> float:
        """对标量序列做“忽略非有限值”的平均聚合。"""

        total = 0.0
        count = 0
        for value in values:
            total, count = self._accumulate_if_finite(total, count, float(value))
        return total / count if count > 0 else float("nan")

    def aggregate_metric_list(self, values: Iterable[float]) -> float:
        """Public metric aggregation entry point shared by validation and testing."""

        return self._aggregate_metric_list(values)

    def _set_eval_mode(self, *models: Optional[nn.Module]) -> List[Tuple[nn.Module, bool]]:
        """切换模型到 eval，并返回原始 training 状态快照。"""

        states: List[Tuple[nn.Module, bool]] = []
        for model in models:
            if model is None:
                continue
            states.append((model, bool(model.training)))
            model.eval()
        return states

    @staticmethod
    def _restore_train_mode(states: List[Tuple[nn.Module, bool]]) -> None:
        """按快照恢复模型 train/eval 状态。"""

        for model, was_training in states:
            model.train(was_training)

    @staticmethod
    def _extract_image_pair(batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """从 batch 中提取 blur/sharp 图像对。

        当 batch 不含 sharp（如纯推理集）时返回 `(blur, None)`。
        """

        blur = batch["blur"].to(device)
        sharp_raw = batch.get("sharp")
        if sharp_raw is None or not torch.is_tensor(sharp_raw):
            return blur, None
        return blur, sharp_raw.to(device)

    def _compute_image_metrics(self, restored: torch.Tensor, sharp: Optional[torch.Tensor]) -> Dict[str, float]:
        """计算单对图像的恢复指标集合。"""

        if sharp is None:
            return {"PSNR": float("nan"), "SSIM": float("nan"), "MAE": float("nan"), "LPIPS": float("nan")}
        restored_eval = self._sanitize_image_tensor(restored)
        sharp_eval = self._sanitize_image_tensor(sharp)
        lp = self._lpips_score(restored_eval, sharp_eval)
        return {
            "PSNR": float(self._psnr(restored_eval, sharp_eval).item()),
            "SSIM": float(self._ssim(restored_eval, sharp_eval).item()),
            "MAE": self._compute_mae(restored_eval, sharp_eval),
            "LPIPS": float(lp.item()) if lp is not None else float("nan"),
        }

    def compute_image_metrics(self, restored: torch.Tensor, sharp: Optional[torch.Tensor]) -> Dict[str, float]:
        """Public image metric entry point shared by validation and testing."""

        return self._compute_image_metrics(restored, sharp)

    @staticmethod
    def _table_as_image(table: torch.Tensor) -> torch.Tensor:
        """把 lens-table 视作图像张量布局，用于 SSIM 计算。"""

        # [B, r, theta, c] -> [B, c, theta, r]
        return table.permute(0, 3, 2, 1).contiguous()

    def evaluate_stage1(self, psf_net: nn.Module, val_loader, device: str, prior_estimator: Optional[nn.Module] = None) -> Dict[str, float]:
        """评估 Stage 1（PriorEstimator）。

        指标包括：
        - `Val_PSF_SFR_L1` / `Val_PSF_SFR_MSSSIM`：table 重建质量；
        - `Val_TV_r` / `Val_TV_theta`：平滑性统计；
        - `Val_LensIdentifiability`：基于 nearest-lens 匹配的可辨识性；
        - `Val_BatchStd`：batch 内预测分布标准差，监控塌缩风险。
        """

        model = prior_estimator if prior_estimator is not None else psf_net
        states = self._set_eval_mode(model)
        l1_values: List[float] = []
        ssim_values: List[float] = []
        batch_std_values: List[float] = []
        tv_r_values: List[float] = []
        tv_theta_values: List[float] = []
        pred_tables: List[tuple[str, torch.Tensor]] = []
        gt_tables: Dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for batch in val_loader:
                    # 无 gt 时无法进行 Stage1 指标计算，直接跳过该 batch。
                    if "gt_psf_sfr" not in batch:
                        continue
                    blur = batch["blur"].to(device)
                    gt = batch["gt_psf_sfr"].to(device)
                    pred = model(blur)

                    # 1) 基础重建误差。
                    l1_values.append(float(F.l1_loss(pred, gt).item()))

                    # 2) 结构相似度（在 table 视图布局上计算）。
                    pred_img = self._table_as_image(pred)
                    gt_img = self._table_as_image(gt)
                    ssim_values.append(float(self._ssim(pred_img, gt_img, max_val=1.0).item()))

                    # 3) batch 内分布离散度（检测先验塌缩），仅在 B>1 时有意义。
                    if int(pred.shape[0]) > 1:
                        # 沿 batch 维求 std 后对其余维度取平均，得到单个标量。
                        batch_std_values.append(float(pred.detach().float().std(dim=0, unbiased=False).mean().item()))

                    # 4) TV 平滑项统计（评估用途，不参与反向）。
                    _, tv_r, tv_theta = lens_table_tv_loss(pred)
                    tv_r_values.append(float(tv_r.item()))
                    tv_theta_values.append(float(tv_theta.item()))

                    # 5) 收集 lens 级匹配信息，用于可辨识性评估。
                    lens_names = batch.get("lens_name", [f"sample_{idx}" for idx in range(pred.shape[0])])
                    for index, lens_name in enumerate(lens_names):
                        key = str(lens_name)
                        pred_tables.append((key, pred[index].detach().cpu()))
                        gt_tables.setdefault(key, gt[index].detach().cpu())
        finally:
            self._restore_train_mode(states)

        top1_hits = 0
        if pred_tables and gt_tables:
            for lens_name, pred_table in pred_tables:
                # 与所有候选 lens 的 gt_table 做 L1 距离排序，统计 top1 命中。
                ranked = sorted(
                    (
                        (candidate, float(F.l1_loss(pred_table, gt_table).item()))
                        for candidate, gt_table in gt_tables.items()
                    ),
                    key=lambda item: item[1],
                )
                if ranked and ranked[0][0] == lens_name:
                    top1_hits += 1

        identifiability = top1_hits / float(len(pred_tables)) if pred_tables else float("nan")
        l1 = self._aggregate_metric_list(l1_values)
        return {
            "Val_PSF_SFR_L1": l1,
            "Val_PSF_SFR_MSSSIM": self._aggregate_metric_list(ssim_values),
            "Val_TV_r": self._aggregate_metric_list(tv_r_values),
            "Val_TV_theta": self._aggregate_metric_list(tv_theta_values),
            "Val_LensIdentifiability": identifiability,
            "Val_BatchStd": self._aggregate_metric_list(batch_std_values),
            "Params(M)": self._count_parameters(model),
        }

    def evaluate_stage2(
        self,
        prior_estimator: nn.Module,
        odn: nn.Module,
        val_loader,
        device: str,
    ) -> Dict[str, float]:
        """评估 Stage 2（ODN 闭环）。

        通过 sharp + pred_table -> simulated_blur，并与真实 blur 比较，
        输出 L1 与 PSNR 两个核心指标。
        """

        states = self._set_eval_mode(prior_estimator, odn)
        odn_l1_values: List[float] = []
        odn_psnr_values: List[float] = []
        try:
            with torch.no_grad():
                for batch in val_loader:
                    # Stage2 必需的输入对：blur/sharp。
                    blur = batch["blur"].to(device)
                    sharp = batch["sharp"].to(device)
                    pred = prior_estimator(blur)
                    simulated = odn(sharp, pred)
                    odn_l1_values.append(float(F.l1_loss(simulated, blur).item()))
                    odn_psnr_values.append(float(self._psnr(self._sanitize_image_tensor(simulated), self._sanitize_image_tensor(blur)).item()))
        finally:
            self._restore_train_mode(states)
        return {
            "Val_ODN_L1": self._aggregate_metric_list(odn_l1_values),
            "Val_ODN_PSNR": self._aggregate_metric_list(odn_psnr_values),
            "Params(M)": self._count_parameters(prior_estimator, odn),
        }

    def _run_pipeline(
        self,
        restoration_net: nn.Module,
        blur: torch.Tensor,
        prior_estimator: Optional[nn.Module] = None,
        lens_table_encoder: Optional[nn.Module] = None,
        crop_info: Optional[torch.Tensor] = None,
        prior_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """统一推理管线入口。

        - 若提供 prior+lens_encoder：走完整 BPFR 注入链路；
        - 否则退化为 restoration 直接前向。
        """

        if lens_table_encoder is None or prior_table is None:
            return restoration_net(blur, lens_features=None, crop_info=crop_info)
        lens_features = lens_table_encoder(prior_table)
        return restoration_net(blur, lens_features, crop_info=crop_info)

    @staticmethod
    def _resolve_batch_prior(batch: Dict[str, Any], mode: str, device: str) -> Optional[torch.Tensor]:
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

    def evaluate(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        val_loader,
        device: str,
        prior_estimator: Optional[nn.Module] = None,
        lens_table_encoder: Optional[nn.Module] = None,
        prior_mode: str = "correct_gt",
    ) -> Dict[str, float]:
        """评估 Stage 3（图像恢复指标）。"""

        prior_model = prior_estimator if prior_estimator is not None else psf_net
        active_lens_encoder = None if str(prior_mode) == "none" else lens_table_encoder
        states = self._set_eval_mode(restoration_net, active_lens_encoder)
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        mae_values: List[float] = []
        lpips_values: List[float] = []
        try:
            with torch.no_grad():
                for batch in val_loader:
                    blur, sharp = self._extract_image_pair(batch, device)
                    crop_info = batch.get("crop_info")
                    if torch.is_tensor(crop_info):
                        crop_info = crop_info.to(device)

                    # 统一走可注入/可退化的推理入口。
                    prior_table = self._resolve_batch_prior(batch, prior_mode, device)
                    restored = self._run_pipeline(
                        restoration_net,
                        blur,
                        prior_model,
                        active_lens_encoder,
                        crop_info,
                        prior_table=prior_table,
                    )
                    metrics = self._compute_image_metrics(restored, sharp)
                    psnr_values.append(metrics["PSNR"])
                    ssim_values.append(metrics["SSIM"])
                    mae_values.append(metrics["MAE"])
                    lpips_values.append(metrics["LPIPS"])
        finally:
            self._restore_train_mode(states)
        return {
            "PSNR": self._aggregate_metric_list(psnr_values),
            "SSIM": self._aggregate_metric_list(ssim_values),
            "MAE": self._aggregate_metric_list(mae_values),
            "LPIPS": self._aggregate_metric_list(lpips_values),
            "Params(M)": self._count_parameters(restoration_net, active_lens_encoder),
        }

    @staticmethod
    def evaluate_model(restoration_net: nn.Module, psf_net: Optional[nn.Module], val_loader, device: str) -> Dict[str, float]:
        """兼容旧调用的静态封装入口。"""

        evaluator = PerformanceEvaluator(device=device)
        return evaluator.evaluate(restoration_net=restoration_net, psf_net=psf_net, val_loader=val_loader, device=device)

    def evaluate_full_resolution(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        test_loader,
        device: str,
        prior_estimator: Optional[nn.Module] = None,
        lens_table_encoder: Optional[nn.Module] = None,
        prior_mode: str = "correct_gt",
    ) -> Tuple[Dict[str, float], List[Dict[str, float | str]]]:
        """全分辨率评估并返回逐图明细。

        返回：
        - average_metrics: 聚合指标；
        - results: 每张图的文件名与指标记录。
        """

        prior_model = prior_estimator if prior_estimator is not None else psf_net
        active_lens_encoder = None if str(prior_mode) == "none" else lens_table_encoder
        states = self._set_eval_mode(restoration_net, active_lens_encoder)
        results: List[Dict[str, float | str]] = []
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        mae_values: List[float] = []
        lpips_values: List[float] = []
        try:
            with torch.no_grad():
                for batch in test_loader:
                    blur, sharp = self._extract_image_pair(batch, device)
                    crop_info = batch.get("crop_info")
                    if torch.is_tensor(crop_info):
                        crop_info = crop_info.to(device)
                    prior_table = self._resolve_batch_prior(batch, prior_mode, device)
                    restored = self._run_pipeline(
                        restoration_net,
                        blur,
                        prior_model,
                        active_lens_encoder,
                        crop_info,
                        prior_table=prior_table,
                    )

                    # 对 batch 中每个样本单独记录指标，便于导出详细报表。
                    for index in range(int(blur.shape[0])):
                        sample_metrics = self._compute_image_metrics(
                            restored[index : index + 1],
                            None if sharp is None else sharp[index : index + 1],
                        )
                        raw_filename = batch.get("filename", "sample.png")
                        filename = str(raw_filename[index] if isinstance(raw_filename, (list, tuple)) else raw_filename)
                        results.append({"filename": filename, **sample_metrics})
                        psnr_values.append(sample_metrics["PSNR"])
                        ssim_values.append(sample_metrics["SSIM"])
                        mae_values.append(sample_metrics["MAE"])
                        lpips_values.append(sample_metrics["LPIPS"])
        finally:
            self._restore_train_mode(states)
        average_metrics = {
            "PSNR": self._aggregate_metric_list(psnr_values),
            "SSIM": self._aggregate_metric_list(ssim_values),
            "MAE": self._aggregate_metric_list(mae_values),
            "LPIPS": self._aggregate_metric_list(lpips_values),
            "Num_Images": len(results),
            "Params(M)": self._count_parameters(restoration_net, active_lens_encoder),
        }
        return average_metrics, results

    def _build_injection_aware_benchmark_model(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        device: str,
        lens_table_encoder: Optional[nn.Module] = None,
    ) -> nn.Module:
        """构建用于 FLOPs/速度测试的统一包装模型。

        该包装器把 prior+lens_encoder+restoration 串联为单一 `forward(blur)`，
        便于外部基准工具调用。
        """

        class _PipelineWrapper(nn.Module):
            """注入链路包装器。"""

            def __init__(self, restoration_model, prior_model, lens_encoder):
                super().__init__()
                self.restoration_model = restoration_model
                self.prior_model = prior_model
                self.lens_encoder = lens_encoder

            def forward(self, blur: torch.Tensor) -> torch.Tensor:
                """执行单输入前向，内部自动决定是否启用先验注入。"""

                if self.prior_model is None or self.lens_encoder is None:
                    return self.restoration_model(blur)
                pred = self.prior_model(blur)
                features = self.lens_encoder(pred)
                return self.restoration_model(blur, features)

        wrapper = _PipelineWrapper(restoration_net, psf_net, lens_table_encoder).to(device)
        wrapper.eval()
        return wrapper

    @staticmethod
    def _try_flops(model: nn.Module, device: str, input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> Optional[float]:
        """尝试使用 THOP 估算 FLOPs。

        若环境缺少 thop 或 profile 失败，返回 None，保证主流程不中断。
        """

        try:
            from thop import profile
        except Exception:
            return None
        try:
            dummy = torch.randn(*input_shape, device=device)
            flops, _ = profile(model, inputs=(dummy,), verbose=False) # type: ignore
        except Exception:
            return None
        return float(flops)
