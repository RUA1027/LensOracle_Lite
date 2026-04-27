from pathlib import Path

from config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_uses_lens_split_and_active_loss_weights():
    cfg = load_config("config/default.yaml")

    assert hasattr(cfg, "lens_split")
    assert hasattr(cfg, "ablation")
    assert not hasattr(cfg, "protocol")
    assert cfg.lens_split.split_manifest == "split_manifest.json"
    assert cfg.training.tv_weight == 0.01
    assert cfg.odn.loss_weight == 0.5
    assert cfg.stage1_graduation.lens_identifiability_weight == 0.20
    assert not hasattr(cfg.stage1_graduation, "prototype_margin_weight")
    assert cfg.ablation.variant == "correctprior"
    assert cfg.ablation.train_prior_mode == "correct_gt"
    assert cfg.ablation.eval_prior_mode == "correct_gt"
    assert cfg.ablation.lens_encoder_enabled is True
    assert cfg.ablation.lens_encoder_padding == "circular"
    assert cfg.ablation.incorrect_prior_policy == "same_split"
    assert cfg.training.stage_schedule.stage1_iterations == 0
    assert cfg.training.stage_schedule.stage2_iterations == 0
    assert cfg.training.stage_schedule.stage3_iterations == 400000


def test_removed_config_fields_are_absent_from_default_config():
    cfg = load_config("config/default.yaml")

    for name in ("data_root", "image_height", "image_width", "repeat_factor"):
        assert not hasattr(cfg.data, name)
    assert not hasattr(cfg.experiment, "epochs")
    assert not hasattr(cfg.experiment.tensorboard, "append_run_name")
    assert not hasattr(cfg.checkpoint, "save_best_per_stage")
    assert not hasattr(cfg.lens_split, "top_k_candidates")


def test_legacy_yaml_keys_migrate_to_active_config():
    cfg_dir = ROOT / "tests" / ".tmp_config"
    cfg_dir.mkdir(exist_ok=True)
    cfg_path = cfg_dir / "legacy.yaml"
    cfg_path.write_text(
        """
protocol:
  split_manifest: old_split.json
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
stage1_graduation:
  prototype_margin_weight: 0.33
training:
  tv_weight: 0.07
odn:
  loss_weight: 0.25
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.lens_split.split_manifest == "old_split.json"
    assert cfg.lens_split.train_ratio == 0.7
    assert cfg.stage1_graduation.lens_identifiability_weight == 0.33
    assert cfg.training.tv_weight == 0.07
    assert cfg.odn.loss_weight == 0.25


def test_sparse_ablation_config_inherits_default_config():
    cfg = load_config("config/ablations/restoration_only.yaml")

    assert cfg.experiment.output_dir == load_config("config/default.yaml").experiment.output_dir
    assert cfg.ablation.variant == "restoration_only"
    assert cfg.ablation.train_prior_mode == "none"
    assert cfg.ablation.eval_prior_mode == "none"
    assert cfg.ablation.lens_encoder_enabled is False
    assert cfg.training.stage_schedule.stage1_iterations == 0
    assert cfg.training.stage_schedule.stage2_iterations == 0
    assert cfg.training.stage_schedule.stage3_iterations == 400000


def test_all_ablation_configs_resolve_expected_variants():
    expected = {
        "config/default.yaml": ("correctprior", "correct_gt", "correct_gt", True, "circular"),
        "config/ablations/restoration_only.yaml": ("restoration_only", "none", "none", False, "circular"),
        "config/ablations/incorrectprior_eval.yaml": (
            "incorrectprior_eval",
            "correct_gt",
            "incorrect_gt",
            True,
            "circular",
        ),
        "config/ablations/zero_padding_encoder.yaml": (
            "zero_padding_encoder",
            "correct_gt",
            "correct_gt",
            True,
            "zero",
        ),
    }

    for path, values in expected.items():
        cfg = load_config(path)
        assert (
            cfg.ablation.variant,
            cfg.ablation.train_prior_mode,
            cfg.ablation.eval_prior_mode,
            cfg.ablation.lens_encoder_enabled,
            cfg.ablation.lens_encoder_padding,
        ) == values
        assert cfg.ablation.incorrect_prior_policy == "same_split"
