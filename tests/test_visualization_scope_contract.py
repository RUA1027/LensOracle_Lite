from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_ablation_suite_has_no_cross_model_visualization_exports():
    source = _read_repo_file("scripts", "evaluate_ablation_suite.py")

    forbidden_tokens = (
        "visual_records",
        "visual_limit",
        "--visual-limit",
        "_export_visuals",
        "plot_ablation_",
        "plot_center_edge_ablation_crops",
        "plot_incorrect_prior_injection",
    )
    for token in forbidden_tokens:
        assert token not in source

    tree = ast.parse(source)
    visualize_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "utils.visualize"
    ]
    assert visualize_imports == []


def test_visualize_module_exposes_only_single_model_plotters():
    source = _read_repo_file("utils", "visualize.py")

    removed_multi_model_plotters = (
        "plot_ablation_recovery_comparison",
        "plot_center_edge_ablation_crops",
        "plot_ablation_error_maps",
        "plot_incorrect_prior_injection",
    )
    for name in removed_multi_model_plotters:
        assert f"def {name}" not in source


def test_training_and_testing_visuals_are_current_run_scoped():
    train_source = _read_repo_file("train.py")
    test_source = _read_repo_file("test.py")

    assert 'Path(output_dir) / "visualizations" / f"epoch_{epoch + 1:03d}"' in train_source
    assert 'visual_root = Path(output_dir) / "visualizations"' in test_source
    assert "plot_ablation_" not in train_source
    assert "plot_ablation_" not in test_source
