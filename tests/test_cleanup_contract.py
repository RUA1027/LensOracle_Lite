import ast
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dataset_registry_keeps_only_active_test_loaders():
    tree = ast.parse(_read("utils/model_builder.py"))
    registry = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DATASET_TYPE_REGISTRY":
                    registry = ast.literal_eval(node.value)
                    break
    assert registry is not None
    assert registry == {
        "dpdd": {"cls": "dpdd_test", "has_gt": True},
        "dpdd_canon": {"cls": "dpdd_test", "has_gt": True},
        "dpdd_pixel": {"cls": "generic_paired", "has_gt": True},
        "realdof": {"cls": "generic_paired", "has_gt": True},
        "extreme": {"cls": "generic_paired", "has_gt": True},
        "cuhk": {"cls": "blur_only", "has_gt": False},
        "omnilens_mixlib": {"cls": "omnilens_mixlib", "has_gt": True},
    }


def test_legacy_symbols_are_absent_from_active_runtime_files():
    active_files = (
        "config/default.yaml",
        "train.py",
        "test.py",
        "utils/__init__.py",
        "utils/evaluation_datasets.py",
        "utils/metrics.py",
        "utils/model_builder.py",
        "utils/omnilens_dataset.py",
    )
    banned = (
        "utils.dpdd_dataset",
        "DPDDDataset",
        "AODLibproPSFDataset",
        "build_psf_dataloader",
        "build_dataloader_from_config",
        "Val_PrototypeMargin",
        "prototype_margin_weight",
        "config.protocol",
        "top_k_candidates",
        "append_run_name",
        "save_best_per_stage",
        "repeat_factor",
        "image_height",
        "image_width",
    )

    for relative_path in active_files:
        text = _read(relative_path)
        for symbol in banned:
            assert symbol not in text, f"{symbol!r} still appears in {relative_path}"


def test_metric_aggregation_ignores_nan_and_infinite_values():
    source = _read("utils/metrics.py")
    tree = ast.parse(source)
    evaluator = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PerformanceEvaluator"
    )
    methods = [
        node
        for node in evaluator.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_accumulate_if_finite", "_aggregate_metric_list", "aggregate_metric_list"}
    ]
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math")]),
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name="Iterable"), ast.alias(name="Tuple")],
                level=0,
            ),
            ast.ClassDef(
                name="MiniEvaluator",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, "metric_subset", "exec"), namespace)

    evaluator_instance = namespace["MiniEvaluator"]()

    assert evaluator_instance.aggregate_metric_list([1.0, float("nan"), 3.0, float("inf")]) == 2.0
    assert math.isnan(evaluator_instance.aggregate_metric_list([float("nan"), float("-inf")]))


def test_stage2_odn_loss_weight_is_wired_into_total_loss():
    trainer_source = _read("trainer.py")
    model_builder_source = _read("utils/model_builder.py")
    default_config = _read("config/default.yaml")

    assert "odn_loss_weight: float = 0.5" in trainer_source
    assert "self.odn_loss_weight = float(odn_loss_weight)" in trainer_source
    assert "total = self.odn_loss_weight * odn_loss" in trainer_source
    assert '"loss_odn_weighted": total' in trainer_source
    assert 'odn_loss_weight=float(getattr(config.odn, "loss_weight", 0.5))' in model_builder_source
    assert "loss_weight: 0.5" in default_config
