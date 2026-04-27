"""Check MixLib image pairs and native PSF-SFR lens-table files.

The active LensOracle pipeline supervises ``gt_psf_sfr`` directly as a
``[64, 48, 67]`` tensor. This script intentionally validates only the data
contract used by that pipeline: blur/gt/label pairing and label-resolved
PSF-SFR tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(path: Path) -> List[Path]:
    """List direct child image files in stable name order."""

    return sorted(item for item in path.glob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def _resolve_gt_path(
    blur_path: Path,
    gt_lookup: Dict[str, Path],
    gt_stem_lookup: Dict[str, Path],
) -> Optional[Path]:
    """Resolve the gt image paired with a blur image."""

    if blur_path.name in gt_lookup:
        return gt_lookup[blur_path.name]
    if blur_path.stem in gt_stem_lookup:
        return gt_stem_lookup[blur_path.stem]
    if blur_path.stem.endswith("-0"):
        candidate_name = f"{blur_path.stem[:-2]}-1{blur_path.suffix}"
        if candidate_name in gt_lookup:
            return gt_lookup[candidate_name]
    if len(gt_lookup) == 1:
        return next(iter(gt_lookup.values()))
    return None


def _collect_tensor_name_index(root_dir: Path) -> Dict[str, Path]:
    """Build a file-name to best-path index for PSF-SFR ``.pth`` files."""

    patterns = ("*.pth", "**/Train/psf_sfr/*.pth", "**/psf_sfr/*.pth", "**/*.pth")
    files: List[Path] = []
    for pattern in patterns:
        files = sorted(root_dir.glob(pattern))
        if files:
            break

    index: Dict[str, Path] = {}
    for path in files:
        previous = index.get(path.name)
        if previous is None:
            index[path.name] = path
            continue
        path_text = str(path).replace("\\", "/")
        previous_text = str(previous).replace("\\", "/")
        path_preferred = "/psf_sfr/" in path_text
        previous_preferred = "/psf_sfr/" in previous_text
        if (path_preferred and not previous_preferred) or (
            path_preferred == previous_preferred and len(path_text) < len(previous_text)
        ):
            index[path.name] = path
    return index


def _verify_image(path: Path) -> Tuple[bool, str]:
    """Check whether an image can be opened by PIL."""

    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _verify_psf_sfr_tensor(path: Path) -> Tuple[bool, str]:
    """Validate a native PSF-SFR lens-table tensor."""

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return False, str(exc)
    if not isinstance(tensor, torch.Tensor):
        return False, "not a torch.Tensor"
    if tensor.shape != (64, 48, 67):
        return False, f"unexpected shape={tuple(tensor.shape)}"
    if not torch.isfinite(tensor).all():
        return False, "contains NaN/Inf"
    return True, ""


def _append_limited(items: List[str], value: str, limit: int) -> None:
    if len(items) < limit:
        items.append(value)


def main() -> None:
    """Run the integrity check and write a JSON report."""

    parser = argparse.ArgumentParser(description="Check MixLib/PSF-SFR integrity and correspondence")
    parser.add_argument("--ab-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--psf-sfr-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--verify-images", action="store_true", help="Verify all blur/gt images via PIL")
    parser.add_argument("--verify-psf-sfr", action="store_true", help="Verify all referenced PSF-SFR tensors")
    parser.add_argument("--max-errors", type=int, default=200)
    args = parser.parse_args()

    ab_dir = Path(args.ab_dir)
    gt_dir = Path(args.gt_dir)
    label_dir = Path(args.label_dir)
    psf_sfr_dir = Path(args.psf_sfr_dir)

    blur_files = _list_images(ab_dir)
    gt_files = _list_images(gt_dir)
    gt_lookup = {path.name: path for path in gt_files}
    gt_stem_lookup = {path.stem: path for path in gt_files}
    psf_sfr_name_index = _collect_tensor_name_index(psf_sfr_dir)

    missing_label: List[str] = []
    missing_gt: List[str] = []
    empty_label: List[str] = []
    unresolved_psf_sfr: List[str] = []
    missing_label_count = 0
    missing_gt_count = 0
    empty_label_count = 0
    unresolved_psf_sfr_count = 0
    paired_items = 0
    referenced_psf_sfr_names: Dict[str, int] = {}

    for blur_path in blur_files:
        label_path = label_dir / f"{blur_path.stem}.txt"
        if not label_path.exists():
            missing_label_count += 1
            _append_limited(missing_label, blur_path.name, args.max_errors)
            continue

        gt_path = _resolve_gt_path(blur_path, gt_lookup, gt_stem_lookup)
        if gt_path is None:
            missing_gt_count += 1
            _append_limited(missing_gt, blur_path.name, args.max_errors)
            continue

        label_value = label_path.read_text(encoding="utf-8").strip()
        if not label_value:
            empty_label_count += 1
            _append_limited(empty_label, label_path.name, args.max_errors)
            continue

        tensor_name = Path(label_value).name
        referenced_psf_sfr_names[tensor_name] = referenced_psf_sfr_names.get(tensor_name, 0) + 1
        if tensor_name not in psf_sfr_name_index:
            unresolved_psf_sfr_count += 1
            _append_limited(unresolved_psf_sfr, f"{blur_path.name} -> {tensor_name}", args.max_errors)
            continue
        paired_items += 1

    image_errors: List[str] = []
    image_errors_count = 0
    if args.verify_images:
        for path in blur_files:
            ok, reason = _verify_image(path)
            if not ok:
                image_errors_count += 1
                _append_limited(image_errors, f"blur:{path.name} -> {reason}", args.max_errors)
        for path in gt_files:
            ok, reason = _verify_image(path)
            if not ok:
                image_errors_count += 1
                _append_limited(image_errors, f"gt:{path.name} -> {reason}", args.max_errors)

    psf_sfr_errors: List[str] = []
    psf_sfr_errors_count = 0
    if args.verify_psf_sfr:
        for tensor_name in sorted(referenced_psf_sfr_names):
            tensor_path = psf_sfr_name_index.get(tensor_name)
            if tensor_path is None:
                continue
            ok, reason = _verify_psf_sfr_tensor(tensor_path)
            if not ok:
                psf_sfr_errors_count += 1
                _append_limited(psf_sfr_errors, f"{tensor_name} -> {reason}", args.max_errors)

    report = {
        "paths": {
            "ab_dir": str(ab_dir),
            "gt_dir": str(gt_dir),
            "label_dir": str(label_dir),
            "psf_sfr_dir": str(psf_sfr_dir),
        },
        "counts": {
            "blur_images": len(blur_files),
            "gt_images": len(gt_files),
            "labels_expected": len(blur_files),
            "labels_referenced_unique_psf_sfr": len(referenced_psf_sfr_names),
            "paired_items": paired_items,
        },
        "issues": {
            "missing_label": missing_label,
            "missing_gt": missing_gt,
            "empty_label": empty_label,
            "unresolved_psf_sfr": unresolved_psf_sfr,
            "corrupt_images": image_errors,
            "invalid_psf_sfr_tensors": psf_sfr_errors,
        },
        "issue_summary": {
            "missing_label_count": missing_label_count,
            "missing_gt_count": missing_gt_count,
            "empty_label_count": empty_label_count,
            "unresolved_psf_sfr_count": unresolved_psf_sfr_count,
            "corrupt_images_count": image_errors_count,
            "invalid_psf_sfr_tensors_count": psf_sfr_errors_count,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Integrity report written:", output_path)
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(json.dumps(report["issue_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
