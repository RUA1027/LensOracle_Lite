"""模型、Trainer 和 DataLoader 构建工具。

本文件是配置到运行时对象的唯一集中入口。新版返回完整的 lens-table
fusion pipeline：PriorEstimator、LensTableEncoder、ODN 和 Restoration。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from config import Config
from models.degradation_simulator import OpticalDegradationNetwork
from models.lens_table_encoder import LensTableEncoder
from models.losses import CharbonnierLoss, MSSSIMLoss, VGGPerceptualLoss
from models.prior_estimator import BlindPriorEstimator
from models.restoration_backbone import CoordGateNAFNetRestoration
from trainer import ThreeStageTrainer
from utils.evaluation_datasets import BlurOnlyTestDataset, DPDDTestDataset, GenericPairedTestDataset
from utils.omnilens_dataset import (
    LensGroupedBatchSampler,
    MixLibDataset,
    create_lens_split_manifest,
    load_lens_split_manifest,
)


def build_models_from_config(config: Config, device: str):
    """根据配置构建完整四模块 pipeline。

返回顺序固定为：
1) prior_estimator
2) lens_table_encoder
3) odn
4) restoration_net

该顺序被 train/test 入口与 checkpoint 加载逻辑共同依赖。
"""

    # grad_checkpointing 是全局开关，会传给支持 checkpoint 的模型模块。
    use_checkpoint = bool(getattr(config.training, "grad_checkpointing", False))
    prior_cfg = config.prior_estimator
    encoder_cfg = config.lens_table_encoder
    attn_cfg = config.cross_attention
    odn_cfg = config.odn
    restoration_cfg = config.restoration

    prior_estimator = BlindPriorEstimator(
        encoder_channels=prior_cfg.encoder_channels,
        blocks_per_level=prior_cfg.blocks_per_level,
        rstb_num_blocks=prior_cfg.rstb_num_blocks,
        rstb_window_size=prior_cfg.rstb_window_size,
        rstb_num_heads=prior_cfg.rstb_num_heads,
        d_latent=prior_cfg.d_latent,
        decoder_seed_channels=prior_cfg.decoder_seed_channels,
        use_checkpoint=use_checkpoint,
    ).to(device)
    lens_table_encoder = LensTableEncoder(
        in_channels=67,
        channels=encoder_cfg.channels,
        blocks_per_level=encoder_cfg.blocks_per_level,
        padding_mode=getattr(config.ablation, "lens_encoder_padding", "circular"),
    ).to(device)
    odn = OpticalDegradationNetwork(
        base_channels=odn_cfg.base_channels,
        bottleneck_channels=odn_cfg.bottleneck_channels,
        num_heads=odn_cfg.num_heads,
        head_dim=odn_cfg.head_dim,
        num_blocks=odn_cfg.num_blocks,
        fourier_feat_num_freqs=attn_cfg.fourier_feat_num_freqs,
    ).to(device)
    restoration_net = CoordGateNAFNetRestoration(
        encoder_channels=restoration_cfg.encoder_channels,
        encoder_blocks=restoration_cfg.encoder_blocks,
        decoder_blocks=restoration_cfg.decoder_blocks,
        coordgate_mlp_hidden=restoration_cfg.coordgate_mlp_hidden,
        cross_attention_num_heads=attn_cfg.num_heads,
        cross_attention_head_dim=attn_cfg.head_dim,
        cross_attention_fourier_freqs=attn_cfg.fourier_feat_num_freqs,
        lens_table_channels=encoder_cfg.channels,
        use_shallow_attention=attn_cfg.use_shallow_attention,
        use_lens_attention=bool(getattr(config.ablation, "lens_encoder_enabled", True)),
        use_checkpoint=use_checkpoint,
    ).to(device)
    return prior_estimator, lens_table_encoder, odn, restoration_net


def _get_loss_cfg(config: Config, key: str, default=None):
    """安全读取 restoration.losses 子项配置。"""

    losses_cfg = getattr(config.restoration, "losses", None)
    if losses_cfg is None:
        return default
    return getattr(losses_cfg, key, default)


def build_trainer_from_config(
    config: Config,
    prior_estimator,
    lens_table_encoder,
    odn,
    restoration_net,
    device: str,
    tensorboard_dir: Optional[str] = None,
):
    """构建 ThreeStageTrainer，并延迟初始化 VGG/MS-SSIM 等较重 loss。"""

    # 若用户未显式传入 tensorboard_dir，则按实验配置推导默认路径。
    if tensorboard_dir is None and getattr(config.experiment.tensorboard, "enabled", False):
        base_dir = config.experiment.tensorboard.log_dir
        if os.path.isabs(base_dir):
            tensorboard_dir = os.path.join(base_dir, config.experiment.name)
        else:
            tensorboard_dir = os.path.join(config.experiment.output_dir, base_dir, config.experiment.name)

    # 感知损失：按需懒加载，避免无用的 VGG 初始化开销。
    perc_cfg = _get_loss_cfg(config, "perceptual", None)
    perceptual_enabled = bool(getattr(perc_cfg, "enabled", True)) if perc_cfg is not None else True
    perceptual_weight = float(getattr(perc_cfg, "weight", 0.0)) if perc_cfg is not None else 0.0
    perceptual_warmup = int(getattr(perc_cfg, "warmup_iterations", 0)) if perc_cfg is not None else 0
    perceptual_loss_builder = (lambda: VGGPerceptualLoss().to(device)) if perceptual_enabled and perceptual_weight > 0 else None

    # Charbonnier 是轻量损失，通常可直接初始化。
    charb_cfg = _get_loss_cfg(config, "charbonnier", None)
    charbonnier_enabled = bool(getattr(charb_cfg, "enabled", True)) if charb_cfg is not None else True
    charbonnier_eps = float(getattr(charb_cfg, "epsilon", 1.0e-3)) if charb_cfg is not None else 1.0e-3
    charbonnier_loss = CharbonnierLoss(epsilon=charbonnier_eps).to(device) if charbonnier_enabled else None

    # MS-SSIM 同样采用按需懒加载。
    ms_cfg = _get_loss_cfg(config, "ms_ssim", None)
    ms_ssim_enabled = bool(getattr(ms_cfg, "enabled", False)) if ms_cfg is not None else False
    ms_ssim_weight = float(getattr(ms_cfg, "weight", 0.0)) if ms_cfg is not None else 0.0
    ms_ssim_loss_builder = (lambda: MSSSIMLoss().to(device)) if ms_ssim_enabled and ms_ssim_weight > 0 else None

    return ThreeStageTrainer(
        prior_estimator=prior_estimator,
        lens_table_encoder=lens_table_encoder,
        odn=odn,
        restoration_net=restoration_net,
        lr_prior=config.training.optimizer.lr_prior,
        lr_lens_encoder=config.training.optimizer.lr_lens_encoder,
        lr_odn=config.training.optimizer.lr_odn,
        lr_restoration=config.training.optimizer.lr_restoration,
        optimizer_type=config.training.optimizer.type,
        weight_decay=config.training.optimizer.weight_decay,
        grad_clip_prior=config.training.gradient_clip.prior,
        grad_clip_lens_encoder=config.training.gradient_clip.lens_encoder,
        grad_clip_odn=config.training.gradient_clip.odn,
        grad_clip_restoration=config.training.gradient_clip.restoration,
        stage_schedule=config.training.stage_schedule,
        use_amp=config.training.use_amp,
        amp_dtype=config.training.amp_dtype,
        accumulation_steps=config.training.accumulation_steps,
        device=device,
        tensorboard_dir=tensorboard_dir,
        perceptual_weight=perceptual_weight,
        perceptual_warmup_iterations=perceptual_warmup,
        perceptual_enabled=perceptual_enabled,
        perceptual_loss_builder=perceptual_loss_builder,
        charbonnier_loss=charbonnier_loss,
        charbonnier_enabled=charbonnier_enabled,
        ms_ssim_loss_builder=ms_ssim_loss_builder,
        ms_ssim_weight=ms_ssim_weight,
        ms_ssim_enabled=ms_ssim_enabled,
        tv_weight=float(getattr(config.training, "tv_weight", 0.01)),
        odn_loss_weight=float(getattr(config.odn, "loss_weight", 0.5)),
        train_prior_mode=str(getattr(config.ablation, "train_prior_mode", "correct_gt")),
        eval_prior_mode=str(getattr(config.ablation, "eval_prior_mode", "correct_gt")),
        lens_encoder_enabled=bool(getattr(config.ablation, "lens_encoder_enabled", True)),
        ablation=dict(getattr(config.ablation, "__dict__", {})),
        nonfinite_patience=getattr(config.training.nonfinite_guard, "patience", 3),
        nonfinite_backoff_factor=getattr(config.training.nonfinite_guard, "backoff_factor", 0.5),
        nonfinite_min_lr=getattr(config.training.nonfinite_guard, "min_lr", 1.0e-6),
    )


def _pin_memory(config: Config) -> bool:
    """仅在 CUDA 设备下启用 DataLoader pin_memory。"""

    return config.experiment.device == "cuda"


def _resolve_mixlib_batch_size(
    config: Config,
    mode: str,
    stage_name: Optional[str],
    batch_size_override: Optional[int],
) -> int:
    """按阶段/模式解析 MixLib 批大小优先级。

优先级：
1) 调用方显式 batch_size_override；
2) stage_schedule 对应阶段 batch size；
3) data.batch_size。
"""

    if batch_size_override is not None:
        return int(batch_size_override)
    stage_schedule = getattr(config.training, "stage_schedule", None)
    if stage_schedule is not None:
        if stage_name == "stage1":
            return int(getattr(stage_schedule, "stage1_batch_size", config.data.batch_size))
        if stage_name == "stage2":
            return int(getattr(stage_schedule, "stage2_batch_size", config.data.batch_size))
        if stage_name == "stage3":
            return int(getattr(stage_schedule, "stage3_batch_size", config.data.batch_size))
        if mode == "train":
            return int(getattr(stage_schedule, "stage3_batch_size", config.data.batch_size))
    return int(config.data.batch_size)


def _resolve_lens_manifest(config: Config) -> dict:
    """读取或生成镜头级划分清单。"""

    manifest_path = Path(str(config.lens_split.split_manifest))
    if manifest_path.exists():
        return load_lens_split_manifest(manifest_path)
    return create_lens_split_manifest(
        label_dir=config.omnilens2.mixlib_label_dir,
        output_path=manifest_path,
        train_ratio=config.lens_split.train_ratio,
        val_ratio=config.lens_split.val_ratio,
        test_ratio=config.lens_split.test_ratio,
        seed=config.lens_split.split_seed,
    )


def build_mixlib_dataloader(
    config: Config,
    mode: str = "train",
    batch_size_override: Optional[int] = None,
    stage_name: Optional[str] = None,
):
    """构建 MixLibDataset 的 DataLoader。

关键策略：
1) Stage1 train 使用 LensGroupedBatchSampler，强化同镜头样本对比；
2) Stage2/3 或 val/test 使用常规 batch 采样；
3) test 模式下可按场景选择是否强制 require_psf_sfr。
"""

    split_manifest = _resolve_lens_manifest(config)
    prior_mode = (
        str(getattr(config.ablation, "train_prior_mode", "correct_gt"))
        if mode == "train"
        else str(getattr(config.ablation, "eval_prior_mode", "correct_gt"))
    )
    require_incorrect_psf_sfr = prior_mode == "incorrect_gt"
    require_psf_sfr = prior_mode in {"correct_gt", "incorrect_gt"}
    dataset = MixLibDataset(
        ab_dir=config.omnilens2.mixlib_ab_dir,
        gt_dir=config.omnilens2.mixlib_gt_dir,
        label_dir=config.omnilens2.mixlib_label_dir,
        psf_sfr_dir=config.omnilens2.psf_sfr_dir,
        crop_size=config.data.crop_size if mode == "train" else config.data.val_crop_size,
        mode=mode,
        random_flip=config.data.augmentation.random_flip,
        random_rotate90=config.data.augmentation.random_rotate90,
        val_split_ratio=config.omnilens2.mixlib_val_split_ratio,
        test_split_ratio=getattr(config.omnilens2, "mixlib_test_split_ratio", 0.0),
        split_seed=config.omnilens2.mixlib_split_seed,
        split_manifest=split_manifest,
        require_psf_sfr=require_psf_sfr,
        require_incorrect_psf_sfr=require_incorrect_psf_sfr,
        incorrect_prior_policy=str(getattr(config.ablation, "incorrect_prior_policy", "same_split")),
    )
    effective_batch_size = _resolve_mixlib_batch_size(config, mode, stage_name, batch_size_override)
    batch_size = effective_batch_size if mode == "train" else max(1, effective_batch_size // 4)
    if mode == "train" and stage_name == "stage1":
        # Stage1 按 lens 分组采样：每个 batch 聚合多个镜头、每镜头固定样本数。
        sampler = LensGroupedBatchSampler(
            lens_names=[str(sample.get("lens_name", "unknown")) for sample in dataset.samples],
            samples_per_lens=2,
            lenses_per_batch=max(1, batch_size // 2),
            shuffle=True,
            seed=config.lens_split.split_seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=_pin_memory(config),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=mode == "train",
        num_workers=config.data.num_workers,
        pin_memory=_pin_memory(config),
        drop_last=mode == "train",
    )


DATASET_TYPE_REGISTRY = {
    "dpdd": {"cls": "dpdd_test", "has_gt": True},
    "dpdd_canon": {"cls": "dpdd_test", "has_gt": True},
    "dpdd_pixel": {"cls": "generic_paired", "has_gt": True},
    "realdof": {"cls": "generic_paired", "has_gt": True},
    "extreme": {"cls": "generic_paired", "has_gt": True},
    "cuhk": {"cls": "blur_only", "has_gt": False},
    "omnilens_mixlib": {"cls": "omnilens_mixlib", "has_gt": True},
}


def get_supported_dataset_types():
    """返回 CLI 支持的 dataset_type 列表。"""

    return list(DATASET_TYPE_REGISTRY.keys())


def _resolve_mixlib_test_dirs(config: Config, data_root_override: Optional[str]):
    """解析 MixLib 测试目录三元组 `(ab, gt, label)`。

支持两种 override 形式：
1) 目录下直接包含 ab/gt/label 子目录；
2) 传入 ab 目录路径（其同级包含 gt/label）。
"""

    if data_root_override is None:
        return (
            config.omnilens2.mixlib_ab_dir,
            config.omnilens2.mixlib_gt_dir,
            config.omnilens2.mixlib_label_dir,
        )
    root = Path(data_root_override)
    if (root / "ab").is_dir() and (root / "gt").is_dir() and (root / "label").is_dir():
        return (str(root / "ab"), str(root / "gt"), str(root / "label"))
    if root.name == "ab" and (root.parent / "gt").is_dir() and (root.parent / "label").is_dir():
        return (str(root), str(root.parent / "gt"), str(root.parent / "label"))
    raise FileNotFoundError("Invalid MixLib override path.")


def _resolve_ood_default_root(dataset_type: str, config: Config) -> Optional[str]:
    """根据 dataset_type 与 ood_eval 配置解析默认测试根路径。"""

    ood_cfg = getattr(config, "ood_eval", None)
    if ood_cfg is None:
        return None
    base_root = str(getattr(ood_cfg, "root", "") or "").strip()
    if not base_root:
        return None
    if dataset_type in {"dpdd", "dpdd_canon"}:
        sub_path = str(getattr(ood_cfg, "dpdd_canon_dir", "") or "").strip()
    elif dataset_type == "dpdd_pixel":
        sub_path = str(getattr(ood_cfg, "dpdd_pixel_dir", "") or "").strip()
    elif dataset_type == "realdof":
        sub_path = str(getattr(ood_cfg, "realdof_dir", "") or "").strip()
    else:
        return None
    if not sub_path:
        return None
    path_obj = Path(sub_path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(Path(base_root) / path_obj)


def build_test_dataloader_by_type(
    dataset_type: str,
    config: Config,
    data_root_override: str = None,
    require_psf_sfr: bool = False,
    require_incorrect_psf_sfr: bool = False,
):
    """按 dataset_type 构建测试 DataLoader。

返回 `(loader, has_gt)`，其中 has_gt 用于控制指标导出逻辑。
"""

    info = DATASET_TYPE_REGISTRY[dataset_type]
    cls_key = info["cls"]
    has_gt = info["has_gt"]
    if data_root_override is None:
        data_root_override = _resolve_ood_default_root(dataset_type, config)
    if cls_key != "omnilens_mixlib" and (require_psf_sfr or require_incorrect_psf_sfr):
        raise ValueError(
            "OOD datasets do not provide PSF-SFR; use the restoration_only ablation "
            "or dataset-type='omnilens_mixlib'."
        )

    if cls_key == "dpdd_test":
        # 标准 DPDD 测试目录结构。
        if data_root_override is None:
            raise ValueError("dataset-type='dpdd' requires --data-root.")
        dataset = DPDDTestDataset(root_dir=data_root_override, transform=None)
        collate_fn = None
    elif cls_key == "generic_paired":
        # 通用 source/target 配对结构。
        if data_root_override is None:
            raise ValueError(f"dataset-type='{dataset_type}' requires --data-root.")
        dataset = GenericPairedTestDataset(root_dir=data_root_override, transform=None)
        collate_fn = None
    elif cls_key == "blur_only":
        # 无 GT 数据集（例如 CUHK）。
        if data_root_override is None:
            raise ValueError("dataset-type='cuhk' requires --data-root.")
        dataset = BlurOnlyTestDataset(root_dir=data_root_override, transform=None)
        collate_fn = BlurOnlyTestDataset.collate_fn
    else:
        # omnilens_mixlib 分支：支持内置 split 或外部目录覆盖。
        ab_dir, gt_dir, label_dir = _resolve_mixlib_test_dirs(config, data_root_override)
        use_internal_lens_split = data_root_override is None
        split_manifest = _resolve_lens_manifest(config) if use_internal_lens_split else None
        dataset = MixLibDataset(
            ab_dir=ab_dir,
            gt_dir=gt_dir,
            label_dir=label_dir,
            psf_sfr_dir=config.omnilens2.psf_sfr_dir,
            crop_size=0,
            mode="test" if use_internal_lens_split else "val",
            random_flip=False,
            random_rotate90=False,
            val_split_ratio=config.omnilens2.mixlib_val_split_ratio if use_internal_lens_split else 0.0,
            test_split_ratio=getattr(config.omnilens2, "mixlib_test_split_ratio", 0.0)
            if use_internal_lens_split
            else 0.0,
            split_seed=config.omnilens2.mixlib_split_seed,
            require_psf_sfr=require_psf_sfr,
            require_incorrect_psf_sfr=require_incorrect_psf_sfr,
            incorrect_prior_policy=str(getattr(config.ablation, "incorrect_prior_policy", "same_split")),
            split_manifest=split_manifest,
        )
        collate_fn = None

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=_pin_memory(config),
        collate_fn=collate_fn,
    )
    return loader, has_gt


def build_test_dataloader_from_config(config: Config, data_root_override: str = None):
    """默认测试入口：使用 omnilens_mixlib 类型。"""

    prior_mode = str(getattr(config.ablation, "eval_prior_mode", "correct_gt"))
    return build_test_dataloader_by_type(
        "omnilens_mixlib",
        config=config,
        data_root_override=data_root_override,
        require_psf_sfr=prior_mode in {"correct_gt", "incorrect_gt"},
        require_incorrect_psf_sfr=prior_mode == "incorrect_gt",
    )
