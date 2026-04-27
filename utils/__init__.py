"""utils 包统一导出入口。

本文件承担两项职责：
1) 直接导出常用可视化函数；
2) 对 model_builder 里的工厂函数做轻量转发，避免上层代码依赖
    具体实现文件路径。

注意：这里的函数均是“转发包装”，不包含业务逻辑。
"""

from .visualize import (
    plot_attention_weights,
    plot_lens_table_comparison,
    plot_odn_reconstruction,
    plot_restoration_with_zoom,
    plot_sfr_curves,
    plot_training_dashboard,
)


def build_models_from_config(*args, **kwargs):
    """构建四模块模型（prior/lens_encoder/odn/restoration）。"""

    from .model_builder import build_models_from_config as _impl

    return _impl(*args, **kwargs)


def build_trainer_from_config(*args, **kwargs):
    """基于配置构建 ThreeStageTrainer。"""

    from .model_builder import build_trainer_from_config as _impl

    return _impl(*args, **kwargs)


def build_mixlib_dataloader(*args, **kwargs):
    """构建 MixLib 数据加载器。"""

    from .model_builder import build_mixlib_dataloader as _impl

    return _impl(*args, **kwargs)


def build_test_dataloader_from_config(*args, **kwargs):
    """按默认测试集配置构建测试加载器。"""

    from .model_builder import build_test_dataloader_from_config as _impl

    return _impl(*args, **kwargs)


def build_test_dataloader_by_type(*args, **kwargs):
    """按 dataset_type 构建测试加载器（含 has_gt 标志）。"""

    from .model_builder import build_test_dataloader_by_type as _impl

    return _impl(*args, **kwargs)
