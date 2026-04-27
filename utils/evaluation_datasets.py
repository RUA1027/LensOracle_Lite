"""Evaluation dataset loaders used by test.py.

This module intentionally contains only test-time datasets. Training uses
MixLibDataset with native lens-table supervision from ``omnilens_dataset.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _default_transform() -> Callable:
    return transforms.Compose([transforms.ToTensor()])


def _list_images(path: str | Path) -> list[str]:
    return sorted(name for name in os.listdir(path) if name.lower().endswith(IMAGE_SUFFIXES))


class DPDDTestDataset(Dataset):
    """Standard DPDD ``test_c/source`` and ``test_c/target`` paired evaluation set."""

    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else _default_transform()
        self.split_dir = self.root_dir / "test_c"
        self.blur_dir = self.split_dir / "source"
        self.sharp_dir = self.split_dir / "target"
        if not self.blur_dir.exists() or not self.sharp_dir.exists():
            raise FileNotFoundError(
                f"Test set directories not found in {self.split_dir}. "
                "Expected 'source' and 'target' subdirectories."
            )
        self.blur_files = _list_images(self.blur_dir)
        self.sharp_files = _list_images(self.sharp_dir)
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError("Mismatch in test set image counts")
        print(f"[DPDDTestDataset] Loaded {len(self.blur_files)} test image pairs")

    def __len__(self) -> int:
        return len(self.blur_files)

    def __getitem__(self, idx: int) -> dict:
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        blur_img = Image.open(self.blur_dir / blur_filename).convert("RGB")
        sharp_img = Image.open(self.sharp_dir / sharp_filename).convert("RGB")
        width, height = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return {
            "blur": self.transform(blur_img),
            "sharp": self.transform(sharp_img),
            "crop_info": crop_info,
            "filename": blur_filename,
            "original_size": (height, width),
        }


class GenericPairedTestDataset(Dataset):
    """Generic paired evaluation set with direct ``source`` and ``target`` folders."""

    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else _default_transform()
        self.blur_dir = self.root_dir / "source"
        self.sharp_dir = self.root_dir / "target"
        if not self.blur_dir.exists() or not self.sharp_dir.exists():
            raise FileNotFoundError(
                f"Source or Target directory not found in {self.root_dir}. "
                "Expected 'source' and 'target' subdirectories."
            )
        self.blur_files = _list_images(self.blur_dir)
        self.sharp_files = _list_images(self.sharp_dir)
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError(
                f"Mismatch: {len(self.blur_files)} source vs {len(self.sharp_files)} target in {self.root_dir}"
            )
        print(f"[GenericPairedTestDataset] Loaded {len(self.blur_files)} paired images from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.blur_files)

    def __getitem__(self, idx: int) -> dict:
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        blur_img = Image.open(self.blur_dir / blur_filename).convert("RGB")
        sharp_img = Image.open(self.sharp_dir / self.sharp_files[idx]).convert("RGB")
        width, height = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return {
            "blur": self.transform(blur_img),
            "sharp": self.transform(sharp_img),
            "crop_info": crop_info,
            "filename": blur_filename,
            "original_size": (height, width),
        }


class BlurOnlyTestDataset(Dataset):
    """Blur-only evaluation set for inference data without ground truth."""

    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else _default_transform()
        self.blur_files = _list_images(self.root_dir)
        if not self.blur_files:
            raise FileNotFoundError(f"No images found in {self.root_dir}")
        print(f"[BlurOnlyTestDataset] Loaded {len(self.blur_files)} images (no GT) from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.blur_files)

    def __getitem__(self, idx: int) -> dict:
        blur_filename = self.blur_files[idx]
        blur_img = Image.open(self.root_dir / blur_filename).convert("RGB")
        width, height = blur_img.size
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return {
            "blur": self.transform(blur_img),
            "sharp": None,
            "crop_info": crop_info,
            "filename": blur_filename,
            "original_size": (height, width),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "blur": torch.stack([item["blur"] for item in batch]),
            "sharp": None,
            "crop_info": torch.stack([item["crop_info"] for item in batch]),
            "filename": [item["filename"] for item in batch],
            "original_size": [item["original_size"] for item in batch],
        }
