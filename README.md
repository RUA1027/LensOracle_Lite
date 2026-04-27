# LensOracle

LensOracle 是一个离线机器学习训练/测试项目，用于基于镜头级物理先验的盲散焦图像恢复。当前仓库只包含数据加载、模型训练、checkpoint 管理、测试评估和可视化导出流程；不包含与真实硬件设备通信的协议层或控制流程。

## 核心架构

当前有效架构是三阶段 native lens-table pipeline：

```text
blur image
  -> BlindPriorEstimator
  -> pred_psf_sfr [B, 64, 48, 67]
  -> LensTableEncoder
  -> multi-scale lens features
  -> CoordGateNAFNetRestoration
  -> restored image
```

设计边界：

- Stage 1 只训练 `BlindPriorEstimator`，直接监督原生 `gt_psf_sfr [64,48,67]`。
- Stage 2 冻结 Stage 1，只训练 `OpticalDegradationNetwork`，用 `sharp + pred_psf_sfr -> simulated_blur` 建立退化闭环。
- Stage 3 冻结 Stage 1/2，只训练 `LensTableEncoder + Restoration` 完成图像恢复。
- 项目不再使用像素级 prior map、raw PSF PCA basis、DPDD 训练数据集或 `protocol_a` 旧划分逻辑。

## 主要入口

- `train.py`：三阶段训练入口，支持 `--stage 1|2|3|all`。
- `test.py`：checkpoint 测试入口，支持 OmniLens MixLib、DPDD、RealDOF、CUHK 等测试集类型。
- `config/default.yaml`：默认配置，保留远程服务器数据路径和输出路径。
- `utils/model_builder.py`：模型、trainer、MixLib dataloader、测试 dataloader 的集中构建入口。
- `utils/omnilens_dataset.py`：MixLib 图像对与 `gt_psf_sfr` lens-table 加载。
- `utils/evaluation_datasets.py`：仅用于 `test.py` 的评测数据集加载器。
- `utils/metrics.py`：训练验证与测试共享的指标计算和聚合逻辑。
- `scripts/check_omnilens_integrity.py`：检查 MixLib 图像配对与 PSF-SFR tensor 是否符合当前数据契约。

## 数据契约

训练使用 MixLib 图像对和 OmniLens++ PSF-SFR lens-table。`MixLibDataset` 输出：

- `blur`：模糊图像 tensor。
- `sharp`：清晰图像 tensor。
- `gt_psf_sfr`：镜头级监督 tensor，形状必须为 `[64, 48, 67]`。
- `crop_info`：归一化裁剪信息，用于恢复物理坐标。
- `filename`、`original_size`、`lens_name`：样本标识和镜头级评估信息。

默认路径由 `config/default.yaml` 管理：

- `omnilens2.psf_sfr_dir`
- `omnilens2.mixlib_ab_dir`
- `omnilens2.mixlib_gt_dir`
- `omnilens2.mixlib_label_dir`
- `ood_eval.*`
- `experiment.output_dir`

这些路径是训练服务器路径，清理任务不会把它们改成本机路径。

## 配置说明

配置加载使用 `config.load_config()`，会把 YAML 转为强类型 dataclass。当前主要配置段：

- `prior_estimator`：Stage 1 prior 预测器结构。
- `lens_table_encoder`：Stage 3 lens-table 编码器结构。
- `cross_attention`：ODN 和 Restoration 共享的 cross-attention 参数。
- `odn`：Stage 2 ODN 结构与 `loss_weight`。
- `lens_split`：镜头级 train/val/test 划分。旧 YAML 中的 `protocol` 会在加载时迁移到这里。
- `stage1_graduation`：Stage 1 报告评分权重与阈值。
- `training`：优化器、梯度裁剪、阶段迭代数、AMP、`tv_weight` 等训练项。
- `restoration.losses`：Stage 3 Charbonnier、perceptual、MS-SSIM 损失配置。
- `checkpoint`：各阶段 best checkpoint 的判定指标。

## 训练

Linux/服务器示例：

```bash
python train.py --config config/default.yaml --stage all
```

Windows 本机示例：

```powershell
D:\anaconda\python.exe train.py --config config/default.yaml --stage 1
```

可用 `--override a.b=value` 临时覆盖配置，例如：

```bash
python train.py --config config/default.yaml --stage 1 \
  --override training.stage_schedule.stage1_iterations=100 \
  --override data.num_workers=0
```

## 测试

```bash
python test.py \
  --checkpoint /path/to/checkpoint.pt \
  --config config/default.yaml \
  --dataset-type omnilens_mixlib
```

可通过 `--dataset-type` 选择：

- `omnilens_mixlib`
- `dpdd` / `dpdd_canon`
- `dpdd_pixel`
- `realdof`
- `extreme`
- `cuhk`

有 GT 的数据集会输出 PSNR、SSIM、MAE、LPIPS；无 GT 数据集会自动保存 restored 图像。

## 输出与 checkpoint

训练输出根目录来自 `experiment.output_dir`。单次运行会生成运行子目录，包含：

- `resolved_config.yaml`
- `tensorboard/`
- `visualizations/`
- `stage1_graduation_report.json`
- `checkpoints/`

固定 checkpoint 目录为：

```text
experiment.output_dir / experiment.name / experiment.checkpoints_subdir
```

常用 checkpoint 文件：

- `latest.pt`
- `latest_stage1.pt`、`latest_stage2.pt`、`latest_stage3.pt`
- `best.pt`
- `best_stage1.pt`、`best_stage2.pt`、`best_stage3.pt`
- `final_model.pt`
- `best_performance.json`

## 指标

- Stage 1：`Val_PSF_SFR_L1`、`Val_PSF_SFR_MSSSIM`、`Val_TV_r`、`Val_TV_theta`、`Val_LensIdentifiability`。
- Stage 2：`Val_ODN_L1`、`Val_ODN_PSNR`。
- Stage 3 / 测试：`PSNR`、`SSIM`、`MAE`、`LPIPS`。

指标解析和 NaN/Inf 聚合过滤集中在 `utils/metrics.py`，避免验证和测试路径各自维护一套逻辑。

## 数据检查

```bash
python scripts/check_omnilens_integrity.py \
  --ab-dir /path/to/ab \
  --gt-dir /path/to/gt \
  --label-dir /path/to/label \
  --psf-sfr-dir /path/to/psf_sfr \
  --output integrity_report.json \
  --verify-images \
  --verify-psf-sfr
```

该脚本只检查当前 pipeline 使用的数据契约，不再检查 raw PSF 或 PCA basis。

## 验证

```bash
python -m pytest tests -q
```

在 Windows 本机若 PyTorch 导入触发 `WinError 10106`，优先运行不依赖 PyTorch 导入的静态/配置测试；完整训练、测试和模型 smoke check 应在 PyTorch 环境正常的训练服务器上执行。

## 硬件协同说明

本仓库当前没有串口、相机、控制板或其他硬件设备通信层。若后续需要处理真实硬件协议、采集流程或在线闭环效率问题，应单独新增硬件协同模块或仓库，并通过明确的数据文件/接口与本离线训练管线衔接。
