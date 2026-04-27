from pathlib import Path

import pytest
import torch
from PIL import Image

from models.lens_table_encoder import LensTableEncoder
from models.restoration_backbone import CoordGateNAFNetRestoration
from trainer import STAGE3, ThreeStageTrainer
from utils.omnilens_dataset import MixLibDataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_mixlib_fixture(root: Path, lens_names: list[str]) -> dict[str, Path]:
    ab = root / "ab"
    gt = root / "gt"
    label = root / "label"
    psf = root / "psf_sfr"
    for folder in (ab, gt, label, psf):
        folder.mkdir(parents=True, exist_ok=True)

    for index, lens in enumerate(lens_names):
        stem = f"sample_{index:03d}"
        _write_image(ab / f"{stem}.png", (10 + index, 20, 30))
        _write_image(gt / f"{stem}.png", (40, 50 + index, 60))
        (label / f"{stem}.txt").write_text(f"{lens}.pth", encoding="utf-8")
        torch.save(torch.full((64, 48, 67), float(index + 1)), psf / f"{lens}.pth")
    return {"ab": ab, "gt": gt, "label": label, "psf": psf}


def test_mixlib_dataset_returns_same_split_incorrect_prior(tmp_path):
    paths = _write_mixlib_fixture(tmp_path, ["lens_a", "lens_b", "lens_c"])
    manifest = {"train_lenses": ["lens_a", "lens_b"], "val_lenses": ["lens_c"], "test_lenses": []}

    dataset = MixLibDataset(
        ab_dir=str(paths["ab"]),
        gt_dir=str(paths["gt"]),
        label_dir=str(paths["label"]),
        psf_sfr_dir=str(paths["psf"]),
        crop_size=0,
        mode="train",
        require_psf_sfr=True,
        require_incorrect_psf_sfr=True,
        split_manifest=manifest,
    )

    sample = dataset[0]

    assert sample["gt_psf_sfr"].shape == (64, 48, 67)
    assert sample["incorrect_gt_psf_sfr"].shape == (64, 48, 67)
    assert sample["incorrect_lens_name"] != sample["lens_name"]
    assert sample["incorrect_lens_name"] in {"lens_a", "lens_b"}


def test_mixlib_dataset_rejects_incorrect_prior_for_single_lens_split(tmp_path):
    paths = _write_mixlib_fixture(tmp_path, ["lens_a", "lens_b"])
    manifest = {"train_lenses": ["lens_a"], "val_lenses": ["lens_b"], "test_lenses": []}

    with pytest.raises(ValueError, match="incorrect"):
        MixLibDataset(
            ab_dir=str(paths["ab"]),
            gt_dir=str(paths["gt"]),
            label_dir=str(paths["label"]),
            psf_sfr_dir=str(paths["psf"]),
            crop_size=0,
            mode="train",
            require_psf_sfr=True,
            require_incorrect_psf_sfr=True,
            split_manifest=manifest,
        )


def test_lens_table_encoder_padding_modes_keep_output_shapes():
    table = torch.randn(2, 64, 48, 67)
    circular = LensTableEncoder(channels=[8, 12, 16], blocks_per_level=[1, 1, 1], padding_mode="circular")
    zero = LensTableEncoder(channels=[8, 12, 16], blocks_per_level=[1, 1, 1], padding_mode="zero")

    circular_out = circular(table)
    zero_out = zero(table)

    assert circular_out.keys() == zero_out.keys() == {"F_1", "F_2", "F_3"}
    for key in circular_out:
        assert circular_out[key].shape == zero_out[key].shape


def test_restoration_forward_allows_no_lens_features():
    model = CoordGateNAFNetRestoration(
        encoder_channels=[8, 12, 16, 16],
        encoder_blocks=[1, 1, 1, 1],
        decoder_blocks=[1, 1, 1],
        coordgate_mlp_hidden=8,
        lens_table_channels=[8, 12, 16],
        use_lens_attention=False,
    )
    blur = torch.randn(1, 3, 16, 16)

    restored, attn = model(blur, lens_features=None, return_attn=True)

    assert restored.shape == blur.shape
    assert attn == {}


class _FailingPrior(torch.nn.Module):
    def forward(self, blur):  # pragma: no cover - failure path only
        raise AssertionError("stage3 should use gt_psf_sfr directly")


class _TinyLensEncoder(torch.nn.Module):
    def forward(self, table):
        return {"F_1": table}


class _TinyRestoration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, blur, lens_features=None, crop_info=None):
        return blur + self.bias


def test_stage3_loss_uses_gt_psf_sfr_without_prior_estimator():
    trainer = ThreeStageTrainer(
        prior_estimator=_FailingPrior(),
        lens_table_encoder=_TinyLensEncoder(),
        odn=torch.nn.Identity(),
        restoration_net=_TinyRestoration(),
        lr_prior=1e-4,
        lr_lens_encoder=1e-4,
        lr_odn=1e-4,
        lr_restoration=1e-4,
        optimizer_type="adamw",
        weight_decay=0.0,
        grad_clip_prior=1.0,
        grad_clip_lens_encoder=1.0,
        grad_clip_odn=1.0,
        grad_clip_restoration=1.0,
        stage_schedule=type("Schedule", (), {"stage1_iterations": 0, "stage2_iterations": 0, "stage3_iterations": 1})(),
        use_amp=False,
        perceptual_enabled=False,
        ms_ssim_enabled=False,
        train_prior_mode="correct_gt",
        eval_prior_mode="correct_gt",
        lens_encoder_enabled=True,
        ablation={"variant": "correctprior"},
        device="cpu",
    )
    batch = {
        "blur": torch.zeros(1, 3, 8, 8),
        "sharp": torch.zeros(1, 3, 8, 8),
        "gt_psf_sfr": torch.zeros(1, 64, 48, 67),
    }

    losses = trainer._forward_and_losses(batch, STAGE3)

    assert "loss_restoration" in losses
