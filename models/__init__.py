"""LensOracle 模型层统一导出入口。

该文件的作用是把训练/测试流程真正使用的模型组件集中暴露给外部：
1) 便于上层（如 model_builder）统一 import；
2) 避免历史分支里的旧模块被误用；
3) 形成清晰的“公共 API 面”，降低重构时的影响范围。

当前导出的组件覆盖：
- 坐标门控与恢复主干；
- lens-table 编码与跨注意力路由；
- 退化模拟网络；
- 训练所需损失函数；
- NAF/Swin 基础积木。
"""

from .coordgate import CoordGate, CoordGateNAFBlock, build_polar_coords
from .cross_attention_router import CrossAttentionRouter
from .degradation_simulator import OpticalDegradationNetwork
from .lens_table_encoder import CircularConv2d, LensTableEncoder
from .losses import CharbonnierLoss, MSSSIMLoss, VGGPerceptualLoss
from .nafblock import NAFBlock, SimpleGate, SimplifiedChannelAttention
from .prior_estimator import BlindPriorEstimator
from .restoration_backbone import CoordGateNAFNetRestoration
from .swin_block import RSTB, SwinTransformerBlock, WindowAttention

