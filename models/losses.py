"""训练损失函数集合。

该文件实现了 Stage 3 主要会用到的三类损失：
1) CharbonnierLoss：平滑 L1，兼顾鲁棒性与可导性；
2) MSSSIMLoss：结构相似性约束，强调感知结构保真；
3) VGGPerceptualLoss：多层特征感知损失，增强高层语义与纹理一致性。

设计上均采用“可独立启用”的模块化形式，便于在配置中按权重组合。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier 损失：`sqrt((x-y)^2 + eps^2)`。

    作用：
    - 在零点附近相对 L1 更平滑，梯度更稳定；
    - 对离群误差的惩罚比 L2 温和，通常能减轻过度平滑。
    """

    def __init__(self, epsilon: float = 1.0e-3):
        super().__init__()
        self.eps_sq = epsilon ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


# ---------------------------------------------------------------------------
# MS-SSIM Loss
# ---------------------------------------------------------------------------
def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """构造一维高斯核（归一化）。"""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_filter(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """对 `[B,C,H,W]` 张量执行可分离高斯滤波。

    先做水平方向卷积，再做垂直方向卷积，等效二维高斯核但计算更高效。
    """
    channels = x.shape[1]
    # Horizontal pass
    w_h = window.view(1, 1, 1, -1).expand(channels, -1, -1, -1)
    x = F.conv2d(x, w_h, padding=(0, window.shape[0] // 2), groups=channels)
    # Vertical pass
    w_v = window.view(1, 1, -1, 1).expand(channels, -1, -1, -1)
    x = F.conv2d(x, w_v, padding=(window.shape[0] // 2, 0), groups=channels)
    return x


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算单尺度 SSIM 的两个组成项。

    返回：
    - `cs`：对比度-结构项；
    - `luminance`：亮度一致性项。

    x/y：待对比的两张图像张量（PyTorch 格式）
    window：高斯滤波核，用于局部区域加权计算均值 / 方差
    data_range：图像像素值范围（默认 0~1，值为 1.0）
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 用高斯滤波对图像做局部加权平均，得到局部均值（亮度）和局部方差/协方差（对比度-结构）。
    mu_x = _gaussian_filter(x, window) # x 的局部均值
    mu_y = _gaussian_filter(y, window) # y 的局部均值
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y # x 和 y 的局部均值乘积

    # 方差 = 局部像素平方的均值 - 均值平方，衡量局部对比度
    sigma_x_sq = _gaussian_filter(x * x, window) - mu_x_sq # x 的局部方差
    sigma_y_sq = _gaussian_filter(y * y, window) - mu_y_sq # y 的局部方差
    sigma_xy = _gaussian_filter(x * y, window) - mu_xy # x 和 y 的局部协方差

    # 数值稳定保护，避免浮点误差导致后续开方/除法异常。
    sigma_x_sq = sigma_x_sq.clamp(min=0.0)
    sigma_y_sq = sigma_y_sq.clamp(min=0.0)

    cs = (2.0 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)
    luminance = (2.0 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
    return cs, luminance


class MSSSIMLoss(nn.Module):
    """多尺度 SSIM 损失。

    返回 `1 - MS_SSIM`，从而最小化损失即最大化结构相似度。
    为了减少依赖，内部自实现了高斯滤波与多尺度聚合逻辑。
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        window_sigma: float = 1.5,
        num_scales: int = 5, # 多尺度数量（论文默认 5，实际可根据输入尺寸调整）
        weights: list[float] | None = None, # 每个尺度的权重（论文固定值）
    ):
        super().__init__()
        self.data_range = float(data_range)
        self.num_scales = int(num_scales)
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] # 论文默认权重，强调中间尺度
        weights = weights[: self.num_scales]
        total = sum(weights)
        self.scale_weights = [w / total for w in weights]
        self.register_buffer(
            "_window",
            _fspecial_gauss_1d(window_size, window_sigma),
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 MS-SSIM 损失。

        处理流程：
        1) 每个尺度计算 `(cs, luminance)`；
        2) 中间尺度使用 `cs`，最粗尺度使用 `luminance*cs`；
        3) 按权重做乘积聚合。
        """

        pred = pred.clamp(0.0, self.data_range)
        target = target.clamp(0.0, self.data_range)

        window = self._window.to(pred.device, pred.dtype)
        cs_products: list[torch.Tensor] = []

        for i in range(self.num_scales):
            # 空间尺寸不足时提前停止，避免窗口越界。
            if pred.shape[-1] < window.shape[0] or pred.shape[-2] < window.shape[0]:
                break
            cs, luminance = _ssim_components(pred, target, window, self.data_range)
            if i < self.num_scales - 1:
                # 中间尺度只累计对比度-结构项。
                cs_products.append(cs.mean(dim=[1, 2, 3]).clamp(min=1e-8))
                # 下采样进入下一尺度。
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
            else:
                # 最粗尺度使用完整 SSIM（亮度 * 对比结构）。
                cs_products.append(
                    (luminance * cs).mean(dim=[1, 2, 3]).clamp(min=1e-8)
                )

        # 多尺度加权乘积聚合。
        ms_ssim_val = torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype)
        for i, cs_val in enumerate(cs_products):
            weight = self.scale_weights[i] if i < len(self.scale_weights) else self.scale_weights[-1]
            ms_ssim_val = ms_ssim_val * cs_val.pow(weight)

        return 1.0 - ms_ssim_val.mean()


# ---------------------------------------------------------------------------
# VGG Multi-Layer Perceptual Loss  (extended to relu4_3)
# ---------------------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    """多层 VGG 感知损失。

    从 VGG16 的 `relu1_2/relu2_2/relu3_3/relu4_3` 提取特征，
    计算预测图与目标图在各层特征空间中的加权 MSE。

    与像素级损失相比，该损失更强调纹理与感知语义一致性。
    """

    # Per-layer weights (shallow layers get less weight, deep layers more)
    DEFAULT_LAYER_WEIGHTS = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]
    EXTRACT_INDICES = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3

    '''
    阶段一（如 relu1_2）：提取出非常浅层的特征（比如图片的颜色斑块、边缘线条）。计算二者浅层特征图的 MSE（均方误差），乘以一个非常小的权重 1.0 / 32。
    阶段二（如 relu2_2）：提取中层特征（比如简单的纹理）。计算 MSE 误差，乘以稍大的权重 1.0 / 16。
    阶段三（如 relu3_3）：特征越来越抽象。乘以权重 1.0 / 8。
    阶段四（如 relu4_3）：提取非常深层的特征（包含了物体高层级的语义特征，比如“这是一只猫的眼睛”）。计算误差，乘以最大的权重 1.0 / 4。
    '''

    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = None
        if use_pretrained:
            try:
                weights = VGG16_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            all_features = vgg16(weights=weights).features
        except Exception:
            all_features = vgg16(weights=None).features

        # Build sub-networks for each extraction point
        max_idx = max(self.EXTRACT_INDICES) + 1  # need up to index 22 inclusive
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.EXTRACT_INDICES:
            self.slices.append(nn.Sequential(*list(all_features.children())[prev : idx + 1]))
            prev = idx + 1

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.layer_weights = list(self.DEFAULT_LAYER_WEIGHTS)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算感知损失。

        先做 ImageNet 统计归一化，再逐层提特征并累加加权误差。
        """

        pred = (pred.clamp(0.0, 1.0) - self.mean) / self.std
        target = (target.clamp(0.0, 1.0) - self.mean) / self.std

        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        x_pred = pred
        x_target = target
        # 预测图和目标图依次通过 VGG16 的 4 个阶段
        for slice_net, w in zip(self.slices, self.layer_weights):
            x_pred = slice_net(x_pred)
            x_target = slice_net(x_target)
            loss = loss + w * F.mse_loss(x_pred, x_target)
        return loss
