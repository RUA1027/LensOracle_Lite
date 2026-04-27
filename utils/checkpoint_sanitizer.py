"""旧 checkpoint 键名清理工具。

历史模型可能包含 `global_head.*` 或 `film.*` 等已废弃参数。新版
lens-table fusion 架构不再使用这些模块，加载 warm-start 权重时需要先
剔除，避免错误地把旧结构带回新网络。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable


def sanitize_legacy_checkpoint(checkpoint: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, list[str]]]:
    """移除旧架构遗留键，并返回清理报告。

    当前约定：
    - prior_estimator 中 `global_head.*` 为历史分支残留；
    - restoration_net 中 `film.*` 为历史分支残留。

    为避免就地修改调用者数据，这里先 deepcopy，再做字段剔除。
    """

    sanitized = deepcopy(checkpoint)
    prior_state = dict(sanitized.get("prior_estimator", {}) or {})
    restoration_state = dict(sanitized.get("restoration_net", {}) or {})

    # 记录被清理的键，便于日志/报告追溯。
    removed_prior_keys = sorted(key for key in prior_state if str(key).startswith("global_head."))
    removed_restoration_keys = sorted(key for key in restoration_state if str(key).startswith("film."))

    for key in removed_prior_keys:
        prior_state.pop(key, None)
    for key in removed_restoration_keys:
        restoration_state.pop(key, None)

    sanitized["prior_estimator"] = prior_state
    sanitized["restoration_net"] = restoration_state
    return sanitized, {
        "removed_prior_keys": removed_prior_keys,
        "removed_restoration_keys": removed_restoration_keys,
    }


def summarize_removed_keys(report: Dict[str, Iterable[str]]) -> str:
    """把清理报告格式化成单行摘要文本。"""

    prior_keys = ", ".join(report.get("removed_prior_keys", [])) or "none"
    restoration_keys = ", ".join(report.get("removed_restoration_keys", [])) or "none"
    return f"prior_estimator[{prior_keys}] restoration_net[{restoration_keys}]"
