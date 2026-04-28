"""Evaluate multiple Stage3 ablation variants on the same test sample order.

This script is metrics-only. Diagnostic visualizations are exported by
``train.py`` and ``test.py`` for the currently running experiment/checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from test import _load_checkpoint, _resolve_filename, _resolve_prior_table, _sanitize_for_json
from utils.metrics import PerformanceEvaluator
from utils.model_builder import build_models_from_config, build_test_dataloader_by_type, get_supported_dataset_types


def _load_suite(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Suite YAML root must be a mapping.")
    variants = data.get("variants") or data.get("models")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Suite YAML must contain a non-empty 'variants' list.")
    data["_base_dir"] = str(Path(path).resolve().parent)
    return data


def _variant_entries(suite: Dict[str, Any]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    base_dir = Path(str(suite.get("_base_dir") or "."))
    variants_list = suite.get("variants") or suite.get("models") or []
    for raw in variants_list:
        if not isinstance(raw, dict):
            raise ValueError("Each variant entry must be a mapping.")
        config_path = raw.get("config")
        checkpoint_path = raw.get("checkpoint")
        if not config_path or not checkpoint_path:
            raise ValueError("Each variant requires 'config' and 'checkpoint'.")
        config_obj = Path(str(config_path))
        checkpoint_obj = Path(str(checkpoint_path))
        if not config_obj.is_absolute():
            candidate = base_dir / config_obj
            config_obj = candidate if candidate.exists() else config_obj
        if not checkpoint_obj.is_absolute():
            candidate = base_dir / checkpoint_obj
            checkpoint_obj = candidate if candidate.exists() else checkpoint_obj
        name = str(raw.get("name") or raw.get("variant") or config_obj.stem)
        entries.append({"name": name, "config": str(config_obj), "checkpoint": str(checkpoint_obj)})
    return entries


def _build_output_dir(suite: Dict[str, Any], override: str | None) -> Path:
    if override:
        output_dir = Path(override)
    elif suite.get("output_dir"):
        output_dir = Path(str(suite["output_dir"]))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("result") / f"ablation_suite_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, list[float]]] = {}
    for row in rows:
        variant = str(row["variant"])
        grouped.setdefault(variant, {"PSNR": [], "SSIM": [], "MAE": [], "LPIPS": []})
        for key in grouped[variant]:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[variant][key].append(float(value))
    return {
        variant: {
            key: (sum(values) / len(values) if values else float("nan"))
            for key, values in metrics.items()
        }
        for variant, metrics in grouped.items()
    }


def _write_outputs(output_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    csv_path = output_dir / "ablation_results.csv"
    fieldnames = ["filename", "variant", "PSNR", "SSIM", "MAE", "LPIPS", "true_lens", "injected_lens"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with (output_dir / "ablation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(summary), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _resolve_device(config, suite: Dict[str, Any]) -> str:
    device = str(suite.get("device") or config.experiment.device)
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _run_variant(
    entry: Dict[str, str],
    suite: Dict[str, Any],
    dataset_type: str,
    data_root: str | None,
    expected_filenames: List[str] | None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    config = load_config(entry["config"])
    prior_mode = str(getattr(config.ablation, "eval_prior_mode", "correct_gt"))
    lens_encoder_enabled = bool(getattr(config.ablation, "lens_encoder_enabled", True))
    device = _resolve_device(config, suite)
    config.experiment.device = device

    prior_estimator, lens_encoder, odn, restoration_net = build_models_from_config(config, device)
    _load_checkpoint(prior_estimator, lens_encoder, odn, restoration_net, entry["checkpoint"], device)
    evaluator = PerformanceEvaluator(device=device)
    loader, _ = build_test_dataloader_by_type(
        dataset_type,
        config=config,
        data_root_override=data_root if data_root is not None else "",
        require_psf_sfr=prior_mode in {"correct_gt", "incorrect_gt"},
        require_incorrect_psf_sfr=prior_mode == "incorrect_gt",
    )

    rows: List[Dict[str, Any]] = []
    filenames: List[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=entry["name"], leave=False):
            blur = batch["blur"].to(device)
            sharp_raw = batch.get("sharp")
            sharp = sharp_raw.to(device) if torch.is_tensor(sharp_raw) else None
            crop_info = batch.get("crop_info")
            crop_info = crop_info.to(device) if torch.is_tensor(crop_info) else None
            prior_table = _resolve_prior_table(batch, prior_mode, device)
            lens_features = (
                lens_encoder(prior_table)
                if lens_encoder_enabled and prior_table is not None
                else None
            )
            restored = restoration_net(blur, lens_features, crop_info=crop_info)

            for index in range(int(blur.shape[0])):
                filename = _resolve_filename(batch.get("filename", "sample.png"), index)
                filenames.append(filename)
                if expected_filenames is not None:
                    expected = expected_filenames[len(filenames) - 1]
                    if filename != expected:
                        raise ValueError(f"Sample order mismatch for {entry['name']}: got {filename}, expected {expected}.")
                metrics = evaluator.compute_image_metrics(
                    restored[index : index + 1],
                    None if sharp is None else sharp[index : index + 1],
                )
                true_lens = _resolve_filename(batch.get("lens_name", ""), index)
                injected_lens = _resolve_filename(batch.get("incorrect_lens_name", ""), index)
                rows.append(
                    {
                        "filename": filename,
                        "variant": entry["name"],
                        **metrics,
                        "true_lens": true_lens,
                        "injected_lens": injected_lens,
                    }
                )
    return rows, filenames


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Stage3 ablation suite and write metrics only.")
    parser.add_argument("--suite", required=True, help="YAML file listing config/checkpoint pairs.")
    parser.add_argument("--dataset-type", default=None, choices=get_supported_dataset_types())
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    suite = _load_suite(args.suite)
    entries = _variant_entries(suite)
    dataset_type = str(args.dataset_type or suite.get("dataset_type") or "omnilens_mixlib")
    data_root = args.data_root if args.data_root is not None else suite.get("data_root")
    output_dir = _build_output_dir(suite, args.output)

    all_rows: List[Dict[str, Any]] = []
    expected_filenames: List[str] | None = None
    for entry in entries:
        rows, filenames = _run_variant(
            entry,
            suite,
            dataset_type,
            None if data_root is None else str(data_root),
            expected_filenames,
        )
        if expected_filenames is None:
            expected_filenames = filenames
        all_rows.extend(rows)

    summary = {
        "suite": args.suite,
        "dataset_type": dataset_type,
        "data_root": data_root,
        "variants": entries,
        "num_images": 0 if expected_filenames is None else len(expected_filenames),
        "average_metrics": _aggregate(all_rows),
        "per_image_results": all_rows,
    }
    _write_outputs(output_dir, all_rows, summary)
    print(f"Ablation suite results saved to: {output_dir}")


if __name__ == "__main__":
    main()
