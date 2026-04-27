"""OmniLens / MixLib 数据集与采样器定义。

该模块服务于 BPFR-Net 的 lens-table 监督训练，核心能力包括：
1) 读取并校验 PSF-SFR lens-table 张量；
2) 构建镜头级 train/val/test 划分；
3) 为 Stage1 提供按镜头分组采样器；
4) 为 MixLib 样本提供图像+lens-table 对齐数据输出。
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_psf_sfr(path: Path) -> torch.Tensor:
    """读取并校验 PSF-SFR 张量，要求形状为 `[64,48,67]`。"""

    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
    if tensor.shape != (64, 48, 67):
        raise ValueError(f"Expected psf_sfr shape (64, 48, 67), got {tuple(tensor.shape)}")
    return tensor.float()


def _collect_candidates(root_dir: Path) -> List[Path]:
    """按优先级收集候选 pth 文件。

    优先尝试规范 PSF-SFR 目录，再回退到递归全量搜索。
    """

    direct_files = sorted(root_dir.glob("*.pth"))
    if direct_files:
        return direct_files
    for pattern in ("**/Train/psf_sfr/*.pth", "**/psf_sfr/*.pth", "**/*.pth"):
        files = sorted(root_dir.glob(pattern))
        if files:
            return files
    return []


def _prefer_candidate(candidate: Path, current: Path, marker: str) -> bool:
    """比较同名候选路径优先级。

    规则：优先包含 marker 子目录的路径；若同级，则优先更短路径。
    """

    candidate_text = str(candidate).replace("\\", "/")
    current_text = str(current).replace("\\", "/")
    candidate_has_marker = f"/{marker}/" in candidate_text
    current_has_marker = f"/{marker}/" in current_text
    if candidate_has_marker != current_has_marker:
        return candidate_has_marker
    return len(candidate_text) < len(current_text)


def _build_name_index(root_dir: Path, marker: str) -> Dict[str, Path]:
    """构建 `文件名 -> 最优路径` 索引。"""

    index: Dict[str, Path] = {}
    for path in _collect_candidates(root_dir):
        existing = index.get(path.name)
        if existing is None or _prefer_candidate(path, existing, marker):
            index[path.name] = path
    return index


def load_lens_split_manifest(path: str | Path) -> Dict[str, object]:
    """读取镜头级 train/val/test 划分清单。"""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid split manifest in {manifest_path}")
    return payload


def create_lens_split_manifest(
    label_dir: str | Path,
    output_path: str | Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, object]:
    """按 lens_name 创建互斥的镜头级数据划分清单。

    特点：
    - train/val/test 按镜头而非按样本划分，降低镜头泄漏风险；
    - 在镜头数较少时会做最小集修正，尽量保证 val/test 非空。
    """

    label_root = Path(label_dir)
    label_files = sorted(label_root.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {label_root}")

    lens_names = sorted({Path(path.read_text(encoding="utf-8").strip()).stem for path in label_files})
    if not lens_names:
        raise FileNotFoundError(f"No lens names resolved from labels in {label_root}")

    # 固定随机种子，保证划分可复现。
    rng = random.Random(int(seed))
    shuffled = list(lens_names)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(round(total * float(train_ratio)))
    val_count = int(round(total * float(val_ratio)))
    test_count = total - train_count - val_count
    if test_count < 0:
        test_count = 0
    # 纠偏：避免 round 后总数溢出。
    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count >= test_count and train_count > 0:
            train_count -= 1
        elif val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    # 小规模数据集时尽量保证 val/test 至少包含 1 个镜头。
    if total >= 3 and val_count == 0:
        val_count = 1
        train_count = max(1, train_count - 1)
    if total >= 3 and test_count == 0:
        test_count = 1
        train_count = max(1, train_count - 1)

    train_end = train_count
    val_end = train_end + val_count
    manifest: Dict[str, object] = {
        "seed": int(seed),
        "label_dir": str(label_root),
        "train_lenses": sorted(shuffled[:train_end]),
        "val_lenses": sorted(shuffled[train_end:val_end]),
        "test_lenses": sorted(shuffled[val_end : val_end + test_count]),
    }
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


class LensGroupedBatchSampler(Sampler[list[int]]):
    """按镜头分组的批采样器。

    主要用于 Stage1：
    - 每个镜头采样固定 `samples_per_lens`；
    - 每个 batch 包含固定 `lenses_per_batch` 个镜头。
    """

    def __init__(
        self,
        lens_names: Sequence[str],
        samples_per_lens: int = 2,
        lenses_per_batch: int = 8,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """初始化分组采样器并构建 `lens -> indices` 映射。"""

        self.lens_names = list(lens_names)
        self.samples_per_lens = max(1, int(samples_per_lens))
        self.lenses_per_batch = max(1, int(lenses_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self._lens_to_indices: Dict[str, list[int]] = OrderedDict()
        for index, lens_name in enumerate(self.lens_names):
            self._lens_to_indices.setdefault(str(lens_name), []).append(index)

    def __iter__(self) -> Iterator[list[int]]:
        """按设置生成批索引序列。"""

        rng = random.Random(self.seed)
        grouped: list[tuple[str, list[int]]] = []
        for lens_name, indices in self._lens_to_indices.items():
            current = list(indices)
            if self.shuffle:
                rng.shuffle(current)
            for start in range(0, len(current), self.samples_per_lens):
                chunk = current[start : start + self.samples_per_lens]
                if len(chunk) == self.samples_per_lens:
                    grouped.append((lens_name, chunk))

        # 镜头片段级随机打散，增强 batch 组合多样性。
        if self.shuffle:
            rng.shuffle(grouped)

        current_batch: list[int] = []
        current_lenses = 0
        for _, chunk in grouped:
            current_batch.extend(chunk)
            current_lenses += 1
            if current_lenses == self.lenses_per_batch:
                yield current_batch
                current_batch = []
                current_lenses = 0

    def __len__(self) -> int:
        """返回理论上可形成的完整 batch 数。"""

        full_groups = sum(len(indices) // self.samples_per_lens for indices in self._lens_to_indices.values())
        return full_groups // self.lenses_per_batch


class MixLibDataset(Dataset):
    """MixLib 图像+lens-table 配对数据集。

    输出中可包含：
    - blur/sharp/crop_info/original_size；
    - gt_psf_sfr（当 `require_psf_sfr=True`）；
    - lens_name（用于镜头级评估与分组）。
    """

    def __init__(
        self,
        ab_dir: str,
        gt_dir: str,
        label_dir: str,
        psf_sfr_dir: str,
        crop_size: int = 512,
        mode: str = "train",
        random_flip: bool = False,
        random_rotate90: bool = False,
        psf_sfr_cache_size: int = 16,
        val_split_ratio: float = 0.0,
        test_split_ratio: float = 0.0,
        split_seed: int = 42,
        require_psf_sfr: bool = True,
        require_incorrect_psf_sfr: bool = False,
        incorrect_prior_policy: str = "same_split",
        split_manifest: Optional[Dict[str, object]] = None,
        split_manifest_path: Optional[str] = None,
    ):
        """初始化 MixLibDataset。

        说明：
        - strict 模式下禁用随机翻转/旋转，保证几何与物理监督一致；
        - 支持 split_manifest 进行镜头级划分；
        - 支持 LRU 缓存 PSF-SFR 张量以减少 I/O。
        """

        super().__init__()
        self.ab_dir = Path(ab_dir)
        self.gt_dir = Path(gt_dir)
        self.label_dir = Path(label_dir)
        self.psf_sfr_dir = Path(psf_sfr_dir)
        self.crop_size = int(crop_size)
        self.mode = mode
        self.random_flip = bool(random_flip)
        self.random_rotate90 = bool(random_rotate90)
        # BPFR strict 约束：避免几何增强破坏 lens-table 对齐关系。
        if self.random_flip or self.random_rotate90:
            raise ValueError(
                "BPFR strict mode forbids random_flip/random_rotate90 to keep "
                "lens-table supervision globally consistent with image geometry."
            )
        self.psf_sfr_cache_size = max(0, int(psf_sfr_cache_size))
        self.val_split_ratio = max(0.0, min(0.5, float(val_split_ratio)))
        self.test_split_ratio = max(0.0, min(0.5, float(test_split_ratio)))
        if self.val_split_ratio + self.test_split_ratio >= 1.0:
            raise ValueError("val_split_ratio + test_split_ratio must be < 1.0")
        self.split_seed = int(split_seed)
        self.require_psf_sfr = bool(require_psf_sfr)
        self.require_incorrect_psf_sfr = bool(require_incorrect_psf_sfr)
        self.incorrect_prior_policy = str(incorrect_prior_policy)
        if self.require_incorrect_psf_sfr and self.incorrect_prior_policy != "same_split":
            raise ValueError("Only incorrect_prior_policy='same_split' is supported.")
        self.split_manifest = dict(split_manifest) if split_manifest is not None else None
        if self.split_manifest is None and split_manifest_path:
            self.split_manifest = load_lens_split_manifest(split_manifest_path)
        self.transform = transforms.ToTensor()
        needs_psf_sfr = self.require_psf_sfr or self.require_incorrect_psf_sfr
        self._psf_sfr_name_index = _build_name_index(self.psf_sfr_dir, "psf_sfr") if needs_psf_sfr else {}
        # LRU 缓存：key 为解析后的绝对路径字符串。
        self._psf_sfr_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        self.blur_files = sorted(path for path in self.ab_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        gt_files = sorted(path for path in self.gt_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        self.gt_lookup = {path.name: path for path in gt_files}
        self.gt_stem_lookup = {path.stem: path for path in gt_files}

        if not self.blur_files:
            raise FileNotFoundError(f"No MixLib blur images found in {self.ab_dir}")

        # 构建有效样本列表，要求 blur 与 gt 可配对；若要求 psf_sfr，则 label 必须存在。
        self.samples = []
        for blur_path in self.blur_files:
            label_path = self.label_dir / f"{blur_path.stem}.txt"
            if needs_psf_sfr and not label_path.exists():
                continue
            try:
                gt_path = self._resolve_gt_path(blur_path)
            except FileNotFoundError:
                continue
            sample = {"blur_path": blur_path, "gt_path": gt_path}
            if label_path.exists():
                lens_profile = label_path.read_text(encoding="utf-8").strip()
                sample["label_path"] = label_path
                sample["lens_profile"] = lens_profile
                sample["lens_name"] = Path(lens_profile).stem
            self.samples.append(sample)
        if not self.samples:
            raise FileNotFoundError("No valid MixLib samples found.")

        self._apply_train_val_split()

        self._build_split_lens_lookup()

    def _build_split_lens_lookup(self) -> None:
        """Build same-split lens lookup used by incorrect-prior ablations."""

        self._lens_to_profile: Dict[str, str] = {}
        for sample in self.samples:
            lens_name = str(sample.get("lens_name") or "")
            lens_profile = str(sample.get("lens_profile") or "")
            if lens_name and lens_profile:
                self._lens_to_profile.setdefault(lens_name, lens_profile)

        split_lenses = sorted(self._lens_to_profile)
        self._incorrect_lens_by_lens: Dict[str, str] = {}
        if not self.require_incorrect_psf_sfr:
            return
        if len(split_lenses) < 2:
            raise ValueError("incorrectprior requires at least two lenses in the active same split.")
        for index, lens_name in enumerate(split_lenses):
            self._incorrect_lens_by_lens[lens_name] = split_lenses[(index + 1) % len(split_lenses)]

    def _apply_train_val_split(self) -> None:
        """按模式应用 train/val/test 划分策略。"""

        if len(self.samples) <= 1 or self.mode not in ("train", "val", "test"):
            if self.mode == "test":
                self.samples = []
            return

        # 优先使用外部 manifest（镜头级显式划分）。
        if self.split_manifest is not None:
            manifest = self.split_manifest
            target_lenses = set(manifest.get(f"{self.mode}_lenses", []))
            self.samples = [
                sample for sample in self.samples if str(sample.get("lens_name")) in target_lenses
            ]
            return

        # 向后兼容：未设置 ratio 时保留 train/val 全集，test 置空。
        if self.val_split_ratio <= 0.0 and self.test_split_ratio <= 0.0:
            if self.mode == "test":
                self.samples = []
            return

        lens_names = sorted(
            {
                str(sample.get("lens_name"))
                for sample in self.samples
                if sample.get("lens_name")
            }
        )

        # 若缺失 lens_name，则回退为旧的样本级划分。
        if not lens_names:
            if self.mode == "test":
                self.samples = []
                return
            rng = random.Random(self.split_seed)
            shuffled = list(self.samples)
            rng.shuffle(shuffled)
            split_index = max(1, int(round(len(shuffled) * (1.0 - self.val_split_ratio))))
            split_index = min(split_index, len(shuffled) - 1)
            self.samples = shuffled[split_index:] if self.mode == "val" else shuffled[:split_index]
            return

        rng = random.Random(self.split_seed)
        shuffled_lenses = list(lens_names)
        rng.shuffle(shuffled_lenses)
        total_lenses = len(shuffled_lenses)

        val_lens_count = int(round(total_lenses * self.val_split_ratio))
        test_lens_count = int(round(total_lenses * self.test_split_ratio))
        if self.val_split_ratio > 0.0 and val_lens_count == 0 and total_lenses >= 3:
            val_lens_count = 1
        if self.test_split_ratio > 0.0 and test_lens_count == 0 and total_lenses >= 3:
            test_lens_count = 1

        # 保证至少留一个镜头给 train。
        max_holdout = max(0, total_lenses - 1)
        if val_lens_count + test_lens_count > max_holdout:
            overflow = val_lens_count + test_lens_count - max_holdout
            reduce_val = min(overflow, val_lens_count)
            val_lens_count -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                test_lens_count = max(0, test_lens_count - overflow)

        train_lens_count = total_lenses - val_lens_count - test_lens_count
        train_end = train_lens_count
        val_end = train_end + val_lens_count

        train_lenses = set(shuffled_lenses[:train_end])
        val_lenses = set(shuffled_lenses[train_end:val_end])
        test_lenses = set(shuffled_lenses[val_end:])

        if self.mode == "train":
            target_lenses = train_lenses
        elif self.mode == "val":
            target_lenses = val_lenses
        else:
            target_lenses = test_lenses

        self.samples = [
            sample for sample in self.samples if str(sample.get("lens_name")) in target_lenses
        ]

    def _resolve_gt_path(self, blur_path: Path) -> Path:
        """解析 blur 对应的 gt 路径，包含若干兼容回退规则。"""

        if blur_path.name in self.gt_lookup:
            return self.gt_lookup[blur_path.name]
        if blur_path.stem in self.gt_stem_lookup:
            return self.gt_stem_lookup[blur_path.stem]
        if blur_path.stem.endswith("-0"):
            candidate = blur_path.with_name(f"{blur_path.stem[:-2]}-1{blur_path.suffix}")
            if candidate.name in self.gt_lookup:
                return self.gt_lookup[candidate.name]
        if len(self.gt_lookup) == 1:
            return next(iter(self.gt_lookup.values()))
        raise FileNotFoundError(f"Could not resolve GT image for blur sample {blur_path.name}")

    def __len__(self) -> int:
        """返回当前 split 的有效样本数。"""

        return len(self.samples)

    def _crop_pair(self, blur: Image.Image, sharp: Image.Image):
        """对 blur/sharp 同步裁剪，保证像素级对齐。"""

        width, height = blur.size
        if self.crop_size <= 0 or height < self.crop_size or width < self.crop_size:
            return blur, sharp, (0, 0, height, width)

        if self.mode == "train":
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = max(0, (height - self.crop_size) // 2)
            left = max(0, (width - self.crop_size) // 2)

        box = (left, top, left + self.crop_size, top + self.crop_size)
        return blur.crop(box), sharp.crop(box), (top, left, self.crop_size, self.crop_size)

    def _augment_pair(self, blur: Image.Image, sharp: Image.Image):
        """对 blur/sharp 做同步几何增强（仅 train 模式）。"""

        if self.mode != "train":
            return blur, sharp

        if self.random_rotate90:
            rotate_k = random.randint(0, 3)
            if rotate_k == 1:
                blur = blur.transpose(Image.Transpose.ROTATE_90)
                sharp = sharp.transpose(Image.Transpose.ROTATE_90)
            elif rotate_k == 2:
                blur = blur.transpose(Image.Transpose.ROTATE_180)
                sharp = sharp.transpose(Image.Transpose.ROTATE_180)
            elif rotate_k == 3:
                blur = blur.transpose(Image.Transpose.ROTATE_270)
                sharp = sharp.transpose(Image.Transpose.ROTATE_270)

        if self.random_flip:
            if random.random() < 0.5:
                blur = ImageOps.mirror(blur)
                sharp = ImageOps.mirror(sharp)
            if random.random() < 0.5:
                blur = ImageOps.flip(blur)
                sharp = ImageOps.flip(sharp)
        return blur, sharp

    def _resolve_path(self, lens_name: str, root_dir: Path, name_index: Dict[str, Path]) -> Optional[Path]:
        """解析 lens profile 对应的文件路径（先直连，再索引回退）。"""

        lens_basename = Path(lens_name).name
        direct_path = root_dir / lens_basename
        if direct_path.exists():
            return direct_path
        return name_index.get(lens_basename)

    def _get_cached_tensor(
        self,
        lens_name: str,
        root_dir: Path,
        name_index: Dict[str, Path],
        cache: OrderedDict[str, torch.Tensor],
        loader,
        artifact_name: str,
    ) -> torch.Tensor:
        """获取并缓存张量（LRU）。"""

        resolved_path = self._resolve_path(lens_name, root_dir, name_index)
        if resolved_path is None:
            raise FileNotFoundError(f"{artifact_name} file '{Path(lens_name).name}' not found under {root_dir}.")

        cache_key = str(resolved_path)
        if self.psf_sfr_cache_size > 0 and cache_key in cache:
            # 命中后移动到队尾，维持 LRU 顺序。
            cached = cache.pop(cache_key)
            cache[cache_key] = cached
            return cached.clone()

        tensor = loader(resolved_path)
        if self.psf_sfr_cache_size > 0:
            if len(cache) >= self.psf_sfr_cache_size:
                cache.popitem(last=False)
            cache[cache_key] = tensor
        return tensor.clone()

    def _get_psf_sfr(self, lens_name: str) -> torch.Tensor:
        """读取指定 lens 的 PSF-SFR 张量（含缓存）。"""

        return self._get_cached_tensor(
            lens_name=lens_name,
            root_dir=self.psf_sfr_dir,
            name_index=self._psf_sfr_name_index,
            cache=self._psf_sfr_cache,
            loader=_load_psf_sfr,
            artifact_name="PSF_SFR",
        )

    def __getitem__(self, index: int):
        """读取并组装单个 MixLib 样本。"""

        sample = self.samples[index]
        blur = Image.open(sample["blur_path"]).convert("RGB")
        sharp = Image.open(sample["gt_path"]).convert("RGB")
        width, height = blur.size
        blur, sharp, crop = self._crop_pair(blur, sharp)
        blur, sharp = self._augment_pair(blur, sharp)

        top, left, crop_h, crop_w = crop
        # 裁剪信息统一归一化，便于后续坐标映射。
        crop_info = torch.tensor(
            [top / height, left / width, crop_h / height, crop_w / width],
            dtype=torch.float32,
        )

        output = {
            "blur": self.transform(blur),
            "sharp": self.transform(sharp),
            "crop_info": crop_info,
            "filename": sample["blur_path"].name,
            "original_size": (height, width),
        }

        label_path = sample.get("label_path")
        if self.require_psf_sfr or self.require_incorrect_psf_sfr:
            # 训练/验证阶段要求每个样本都可对齐到 gt_psf_sfr。
            if label_path is None:
                raise FileNotFoundError(f"Missing label file for sample {sample['blur_path'].name}.")
            lens_profile = str(sample.get("lens_profile") or label_path.read_text(encoding="utf-8").strip())
            lens_name = str(sample.get("lens_name") or Path(lens_profile).stem)
            output["lens_name"] = lens_name
            if self.require_psf_sfr:
                output["gt_psf_sfr"] = self._get_psf_sfr(lens_profile)
            if self.require_incorrect_psf_sfr:
                incorrect_lens = self._incorrect_lens_by_lens.get(lens_name)
                if incorrect_lens is None:
                    raise ValueError(f"No same-split incorrect prior available for lens '{lens_name}'.")
                incorrect_profile = self._lens_to_profile[incorrect_lens]
                output["incorrect_gt_psf_sfr"] = self._get_psf_sfr(incorrect_profile)
                output["incorrect_lens_name"] = incorrect_lens
        else:
            output["lens_name"] = str(sample.get("lens_name") or "unknown")

        return output
