"""BPFR-Net lens-table fusion 配置系统。

本模块负责把 YAML 配置加载为强类型 dataclass 对象，并提供三项核心能力：
1) 结构化配置建模：把训练/数据/可视化/评估配置分层组织；
2) 兼容旧字段：在加载时吸收部分历史字段命名，降低配置迁移成本；
3) 运行时覆盖：支持 `--override a.b=value` 动态覆盖任意层级字段。

整体服务于三阶段流程：
- Stage 1：训练 prior_estimator；
- Stage 2：冻结 prior，训练 ODN；
- Stage 3：训练 lens_table_encoder + restoration。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PriorEstimatorConfig:
    """PriorEstimator 模块的结构超参数。"""

    encoder_channels: List[int] = field(default_factory=lambda: [48, 96, 192, 256])
    blocks_per_level: int = 2
    rstb_num_blocks: int = 2
    rstb_window_size: int = 8
    rstb_num_heads: int = 8
    d_latent: int = 512
    decoder_seed_channels: int = 256


@dataclass
class LensTableEncoderConfig:
    """LensTableEncoder 的多尺度通道与层深配置。"""

    channels: List[int] = field(default_factory=lambda: [128, 192, 256])
    blocks_per_level: List[int] = field(default_factory=lambda: [2, 2, 3])


@dataclass
class CrossAttentionConfig:
    """Cross-attention 路由超参数（ODN 与 Restoration 共用）。"""

    num_heads: int = 4
    head_dim: int = 64
    use_shallow_attention: bool = False
    fourier_feat_num_freqs: int = 8


@dataclass
class ODNConfig:
    """Optical Degradation Network（Stage 2）配置。"""

    base_channels: int = 32
    bottleneck_channels: int = 96
    num_heads: int = 4
    head_dim: int = 32
    num_blocks: int = 1
    loss_weight: float = 0.5


@dataclass
class AblationConfig:
    """Stage3-only restoration ablation switches."""

    variant: str = "correctprior"
    train_prior_mode: str = "correct_gt"
    eval_prior_mode: str = "correct_gt"
    lens_encoder_enabled: bool = True
    lens_encoder_padding: str = "circular"
    incorrect_prior_policy: str = "same_split"


@dataclass
class LensSplitConfig:
    """Lens-level train/val/test split configuration."""

    split_manifest: str = "split_manifest.json"
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class Stage1GraduationConfig:
    """Stage 1 毕业评估配置（报告与阈值判定）。"""

    mode: str = "soft"
    prior_l1_weight: float = 0.45
    prior_msssim_weight: float = 0.35
    lens_identifiability_weight: float = 0.20
    batch_std_warn_min: float = 0.010
    batch_std_fail_min: float = 0.004
    prior_l1_fail_max: float = 0.22
    prior_msssim_fail_min: float = 0.55


@dataclass
class OmniLens2Config:
    """OmniLens++/MixLib 路径与切分参数。"""

    psf_sfr_dir: str = "./data/OmniLens++/AODLibpro_lens/psf_sfr"
    mixlib_ab_dir: str = "./data/OmniLens++/split_MixLib/MixLib_32401l40i/hybrid/ab"
    mixlib_gt_dir: str = "./data/OmniLens++/split_MixLib/MixLib_32401l40i/hybrid/gt"
    mixlib_label_dir: str = "./data/OmniLens++/split_MixLib/MixLib_32401l40i/hybrid/label"
    mixlib_val_split_ratio: float = 0.1
    mixlib_test_split_ratio: float = 0.1
    mixlib_split_seed: int = 42


@dataclass
class OODEvalConfig:
    """OOD 测试集默认路径配置。"""

    root: str = "/root/autodl-tmp"
    dpdd_canon_dir: str = "dd_dp_dataset_png"
    dpdd_pixel_dir: str = "dd_dp_dataset_pixel/test_c"
    realdof_dir: str = "RealDOF"


@dataclass
class RestorationConfig:
    """Stage 3 Restoration 主干配置。"""

    @dataclass
    class CharbonnierLossConfig:
        """Charbonnier 重建损失配置。"""

        enabled: bool = True
        epsilon: float = 1.0e-3

    @dataclass
    class PerceptualLossConfig:
        """感知损失（VGG）配置。"""

        enabled: bool = True
        weight: float = 0.05
        warmup_iterations: int = 20_000

    @dataclass
    class MSSSIMLossConfig:
        """MS-SSIM 附加损失配置。"""

        enabled: bool = True
        weight: float = 0.10

    @dataclass
    class LossesConfig:
        """Restoration 阶段损失聚合配置。"""

        charbonnier: "RestorationConfig.CharbonnierLossConfig" = field(
            default_factory=lambda: RestorationConfig.CharbonnierLossConfig()
        )
        perceptual: "RestorationConfig.PerceptualLossConfig" = field(
            default_factory=lambda: RestorationConfig.PerceptualLossConfig()
        )
        ms_ssim: "RestorationConfig.MSSSIMLossConfig" = field(
            default_factory=lambda: RestorationConfig.MSSSIMLossConfig()
        )

    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 256])
    encoder_blocks: List[int] = field(default_factory=lambda: [4, 6, 6, 8])
    decoder_blocks: List[int] = field(default_factory=lambda: [6, 4, 2])
    coordgate_mlp_hidden: int = 32
    losses: "RestorationConfig.LossesConfig" = field(
        default_factory=lambda: RestorationConfig.LossesConfig()
    )


@dataclass
class OptimizerConfig:
    """分模块学习率与优化器配置。"""

    type: str = "adamw"
    lr_prior: float = 2.0e-4
    lr_lens_encoder: float = 1.0e-4
    lr_odn: float = 1.0e-4
    lr_restoration: float = 1.0e-4
    weight_decay: float = 0.01


@dataclass
class GradientClipConfig:
    """不同子模块的梯度裁剪阈值。"""

    restoration: float = 2.5
    prior: float = 1.0
    lens_encoder: float = 1.0
    odn: float = 1.0


@dataclass
class StageScheduleConfig:
    """三阶段训练步数与批大小配置。"""

    stage1_iterations: int = 0
    stage2_iterations: int = 0
    stage3_iterations: int = 400_000
    stage1_batch_size: int = 4
    stage2_batch_size: int = 4
    stage3_batch_size: int = 4


@dataclass
class NonFiniteGuardConfig:
    """非有限数值防护策略配置。"""

    patience: int = 3
    backoff_factor: float = 0.5
    min_lr: float = 1.0e-6


@dataclass
class TrainingConfig:
    """训练器总配置（优化器、调度、AMP、梯度检查点等）。"""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    accumulation_steps: int = 1
    stage_schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)
    nonfinite_guard: NonFiniteGuardConfig = field(default_factory=NonFiniteGuardConfig)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    grad_checkpointing: bool = True
    tv_weight: float = 0.01


@dataclass
class AugmentationConfig:
    """数据增强开关。

    注意：当前 BPFR strict 模式下，随机翻转/旋转在数据集侧会被禁止。
    """

    random_flip: bool = False
    random_rotate90: bool = False


@dataclass
class DataConfig:
    """数据加载与裁剪配置。"""

    batch_size: int = 8
    crop_size: int = 256
    val_crop_size: int = 256
    num_workers: int = 8
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class VisualizationExportConfig:
    """可视化导出配置。"""

    enabled: bool = True
    interval: int = 10
    prior_channels: List[int] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60])
    export_attention_weights: bool = True


@dataclass
class VisualizationConfig:
    """可视化配置根节点。"""

    export: VisualizationExportConfig = field(default_factory=VisualizationExportConfig)


@dataclass
class TensorBoardConfig:
    """TensorBoard 日志配置。"""

    enabled: bool = True
    log_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """实验运行与输出目录配置。"""

    name: str = "bpfr_net_lens_table"
    seed: int = 42
    device: str = "cuda"
    save_interval: int = 10_000
    validation_interval: int = 2_000
    output_dir: str = "results"
    run_name: Optional[str] = None
    use_timestamp: bool = True
    timestamp_format: str = "%m%d_%H%M"
    checkpoints_subdir: str = "checkpoints"
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)


@dataclass
class CheckpointConfig:
    """最佳模型判定指标配置。"""

    stage1_metric: str = "val_psf_sfr_l1"
    stage2_metric: str = "val_odn_l1"
    stage3_metric: str = "psnr"


@dataclass
class Config:
    """顶层配置对象。

    该类聚合全部子配置，并提供：
    - to_dict: 递归转为可序列化字典；
    - save: 以 YAML 形式落盘，便于复现实验。
    """

    prior_estimator: PriorEstimatorConfig = field(default_factory=PriorEstimatorConfig)
    lens_table_encoder: LensTableEncoderConfig = field(default_factory=LensTableEncoderConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    odn: ODNConfig = field(default_factory=ODNConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    lens_split: LensSplitConfig = field(default_factory=LensSplitConfig)
    stage1_graduation: Stage1GraduationConfig = field(default_factory=Stage1GraduationConfig)
    omnilens2: OmniLens2Config = field(default_factory=OmniLens2Config)
    ood_eval: OODEvalConfig = field(default_factory=OODEvalConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=False, allow_unicode=True)


def _dataclass_to_dict(obj: Any) -> Any:
    """递归把 dataclass/list 转为基础 Python 结构。

    用于统一序列化入口，确保 Config.save() 输出可直接 YAML dump。
    """

    if hasattr(obj, "__dataclass_fields__"):
        return {key: _dataclass_to_dict(value) for key, value in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def _dict_to_dataclass(cls, data: Optional[Dict[str, Any]]):
    """递归把 dict 转为 dataclass。

    设计要点：
    - 缺失字段：使用 dataclass 默认值；
    - 多余字段：自动忽略（兼容历史配置文件中的废弃项）；
    - 嵌套字段：递归构造子 dataclass。
    """

    if data is None:
        return cls()
    kwargs = {}
    for field_info in cls.__dataclass_fields__.values():
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        default_value = getattr(cls(), name)
        if hasattr(default_value, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _dict_to_dataclass(type(default_value), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_overrides(config_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """处理命令行 `--override a.b=value`。

    规则：
    - 支持 bool/int/float/list 的自动解析；
    - 无法解析时按字符串保留；
    - 若中间层级不存在，会自动创建字典节点。
    """

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        try:
            if value_str.lower() in ("true", "false"):
                value: Any = value_str.lower() == "true"
            elif value_str.startswith("[") and value_str.endswith("]"):
                value = yaml.safe_load(value_str)
            elif "." in value_str or "e-" in value_str.lower() or "e+" in value_str.lower():
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str

        current = config_dict
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return config_dict


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge YAML dictionaries without mutating inputs."""

    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(path_obj: Path) -> Dict[str, Any]:
    """Load YAML and apply an optional base_config relative to the child file."""

    with open(path_obj, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path_obj}")
    base_config = data.get("base_config")
    if base_config is None:
        return data
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        base_path = path_obj.parent / base_path
    base_data = _load_yaml_with_base(base_path.resolve())
    return _deep_merge_dicts(base_data, data)


def _build_config_from_dict(data: Dict[str, Any]) -> Config:
    """从 YAML 字典构造 Config，并执行旧字段兼容转换。

    当前兼容策略包含：
    - prior_estimator.z_lens_dim -> d_latent；
    - 丢弃若干已废弃字段（如 output_channels、sft_kernel_size 等）。
    """

    # PriorEstimator 历史字段兼容。
    prior_data = dict(data.get("prior_estimator", {}) or {})
    prior_data.pop("output_channels", None)
    if "z_lens_dim" in prior_data and "d_latent" not in prior_data:
        prior_data["d_latent"] = prior_data.pop("z_lens_dim")

    # 训练调度历史字段兼容。
    training_data = dict(data.get("training", {}) or {})
    stage_schedule_data = dict(training_data.get("stage_schedule", {}) or {})
    if "stage2_warmup_iterations" in stage_schedule_data:
        stage_schedule_data.pop("stage2_warmup_iterations", None)

    # Restoration 历史字段兼容。
    restoration_data = dict(data.get("restoration", {}) or {})
    restoration_data.pop("sft_kernel_size", None)
    restoration_data.pop("perceptual_weight", None)
    restoration_data.pop("perceptual_warmup_iterations", None)

    # Legacy YAML files used the ambiguous key name "protocol".
    lens_split_data = dict(data.get("protocol", {}) or {})
    lens_split_data.update(dict(data.get("lens_split", {}) or {}))

    graduation_data = dict(data.get("stage1_graduation", {}) or {})
    if (
        "prototype_margin_weight" in graduation_data
        and "lens_identifiability_weight" not in graduation_data
    ):
        graduation_data["lens_identifiability_weight"] = graduation_data.pop("prototype_margin_weight")

    return Config(
        prior_estimator=_dict_to_dataclass(PriorEstimatorConfig, prior_data),
        lens_table_encoder=_dict_to_dataclass(
            LensTableEncoderConfig, data.get("lens_table_encoder", {})
        ),
        cross_attention=_dict_to_dataclass(CrossAttentionConfig, data.get("cross_attention", {})),
        odn=_dict_to_dataclass(ODNConfig, data.get("odn", {})),
        ablation=_dict_to_dataclass(AblationConfig, data.get("ablation", {})),
        lens_split=_dict_to_dataclass(LensSplitConfig, lens_split_data),
        stage1_graduation=_dict_to_dataclass(
            Stage1GraduationConfig,
            graduation_data,
        ),
        omnilens2=_dict_to_dataclass(OmniLens2Config, data.get("omnilens2", {})),
        ood_eval=_dict_to_dataclass(OODEvalConfig, data.get("ood_eval", {})),
        restoration=_dict_to_dataclass(RestorationConfig, restoration_data),
        training=TrainingConfig(
            optimizer=_dict_to_dataclass(
                OptimizerConfig, training_data.get("optimizer", {})
            ),
            gradient_clip=_dict_to_dataclass(
                GradientClipConfig, training_data.get("gradient_clip", {})
            ),
            accumulation_steps=training_data.get("accumulation_steps", 1),
            stage_schedule=_dict_to_dataclass(StageScheduleConfig, stage_schedule_data),
            nonfinite_guard=_dict_to_dataclass(
                NonFiniteGuardConfig, training_data.get("nonfinite_guard", {})
            ),
            use_amp=training_data.get("use_amp", True),
            amp_dtype=training_data.get("amp_dtype", "bfloat16"),
            grad_checkpointing=training_data.get("grad_checkpointing", True),
            tv_weight=training_data.get("tv_weight", 0.01),
        ),
        data=DataConfig(
            batch_size=data.get("data", {}).get("batch_size", 8),
            crop_size=data.get("data", {}).get("crop_size", 256),
            val_crop_size=data.get("data", {}).get("val_crop_size", 256),
            num_workers=data.get("data", {}).get("num_workers", 8),
            augmentation=_dict_to_dataclass(
                AugmentationConfig, data.get("data", {}).get("augmentation", {})
            ),
        ),
        visualization=VisualizationConfig(
            export=_dict_to_dataclass(
                VisualizationExportConfig, data.get("visualization", {}).get("export", {})
            )
        ),
        experiment=_dict_to_dataclass(ExperimentConfig, data.get("experiment", {})),
        checkpoint=_dict_to_dataclass(CheckpointConfig, data.get("checkpoint", {})),
    )


def load_config(path: str = "config/default.yaml", overrides: Optional[List[str]] = None) -> Config:
    """加载 YAML 配置并返回强类型 Config。

    执行顺序：
    1) 读取 YAML；
    2) 应用命令行 overrides（若提供）；
    3) 执行兼容转换并构造 dataclass。
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = _load_yaml_with_base(path_obj.resolve())
    if overrides:
        data = _apply_overrides(data, overrides)
    return _build_config_from_dict(data)


def get_default_config() -> Config:
    """返回默认配置对象（不读取外部文件）。"""

    return Config()
