import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()

import transferattack
from transferattack.utils import AdvDataset, save_images


GLOBAL_SEED = 42
DEFAULT_SOURCES = "resnet18"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
DEFAULT_EPSILON = "16/255"
DEFAULT_GENERATION_BATCHSIZE = 32

ATTACK_DEFAULT_BATCHSIZE = {
    "l2t": 1,
    "bsr": 8,
    "decowa": 32,
    "ops": 32,
    "sid": 32,
    "eda": 32,
}
SOURCE_ATTACK_BATCHSIZE = {
    "resnet18": {
        "l2t": 2,
        "bsr": 32,
        "decowa": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "densenet121": {
        "l2t": 1,
        "bsr": 4,
        "decowa": 4,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "vgg19": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 8,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_v3": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_v4": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_resnet_v2": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
}

ATTACK_DISPLAY = {
    "l2t": "L2T",
    "bsr": "BSR",
    "decowa": "DeCoWA",
    "ops": "OPS",
    "sid": "SID",
    "eda": "EDA",
}
SOURCE_DISPLAY = {
    "resnet18": "RN-18",
    "inception_v3": "Inc-v3",
    "inception_v4": "Inc-v4",
    "inception_resnet_v2": "IncRes-v2",
}
METRIC_FIELDS = ["ssim", "psnr", "lpips", "tv", "nmse", "nlpd", "gmsd"]


def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = GLOBAL_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def parse_list(raw_values: str) -> List[str]:
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_attacks(raw_values: str) -> List[str]:
    attacks = [item.strip().lower() for item in raw_values.split(",") if item.strip()]
    if not attacks:
        raise ValueError("At least one attack is required.")
    for attack in attacks:
        if attack not in transferattack.attack_zoo:
            raise ValueError(f"Unsupported attack: {attack}")
    return attacks


def parse_epsilon_token(token: str) -> Tuple[str, float]:
    text = token.strip()
    if not text:
        raise ValueError("Empty epsilon token.")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        value = float(numerator) / float(denominator)
        label = f"{int(float(numerator))}/{int(float(denominator))}"
    else:
        value = float(text)
        scaled = value * 255.0
        label = f"{scaled:g}/255" if abs(round(scaled) - scaled) < 1e-8 else f"{value:g}"
    return label, value


def eps_dir_name(eps_label: str) -> str:
    return "eps_" + eps_label.replace("/", "_").replace(".", "p")


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_attack(attack: str) -> str:
    return ATTACK_DISPLAY.get(attack, attack.upper())


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def resolve_generation_batchsize(args: argparse.Namespace, source: str, attack: str) -> int:
    if args.batchsize > 0:
        return args.batchsize
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def make_loader(dataset: AdvDataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def case_key(source: str, attack: str) -> Tuple[str, str]:
    return source, attack


def case_output_dir(args: argparse.Namespace, source: str, attack: str) -> Path:
    eps_label, _ = parse_epsilon_token(args.eps)
    return Path(args.output_dir) / display_source(source).replace("-", "") / eps_dir_name(eps_label) / attack


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "perceptual_quality_results.csv"
    per_image_csv = (
        Path(args.per_image_csv)
        if args.per_image_csv
        else output_dir / "perceptual_quality_per_image.csv"
    )
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "perceptual_quality_table_rows.tex"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "perceptual_quality_results_analysis.txt"
    )
    return result_csv, per_image_csv, summary_tex, analysis_file


def build_attacker(
    args: argparse.Namespace,
    source: str,
    attack: str,
    eps_label: str,
    eps_value: float,
):
    alpha = args.alpha if args.alpha is not None else eps_value / float(args.epoch)
    attack_class = transferattack.load_attack_class(attack)
    common_kwargs = dict(
        model_name=source,
        epsilon=eps_value,
        alpha=alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
        seed=args.seed,
    )
    if attack == "bsr":
        common_kwargs.update(num_scale=args.num_warping)
    elif attack == "decowa":
        common_kwargs.update(num_warping=args.num_warping)
    elif attack == "sid":
        common_kwargs.update(num_scale=args.num_warping)
    elif attack == "eda":
        common_kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
        )
    return attack_class(**common_kwargs)


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    attack: str,
    eps_label: str,
    eps_value: float,
    elapsed_seconds: float,
    batch_size: int,
) -> None:
    alpha = args.alpha if args.alpha is not None else eps_value / float(args.epoch)
    metadata = {
        "source": source,
        "source_label": display_source(source),
        "attack": attack,
        "attack_label": display_attack(attack),
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": alpha,
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": args.targeted,
        "random_start": args.random_start,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "final_adv_coordinate_system": "original image coordinate system",
    }
    if attack == "eda":
        metadata.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
        )
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_cases(args: argparse.Namespace, sources: List[str], attacks: List[str]) -> None:
    eps_label, eps_value = parse_epsilon_token(args.eps)
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for attack in attacks:
            set_seed(args.seed)
            case_dir = case_output_dir(args, source, attack)
            if args.reuse_existing and has_expected_images(case_dir, expected):
                print(f"Skipping existing case: {source}/{attack}")
                continue
            if case_dir.exists() and not args.reuse_existing:
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Generating source={source}, attack={attack}, epsilon={eps_label}")
            print(f"Output: {case_dir}")

            attacker = build_attacker(args, source, attack, eps_label, eps_value)
            dataset = AdvDataset(
                input_dir=args.input_dir,
                output_dir=str(case_dir),
                targeted=args.targeted,
                eval=False,
            )
            batch_size = resolve_generation_batchsize(args, source, attack)
            print(f"Generation batchsize: {batch_size}")
            loader = make_loader(dataset, batch_size, args)

            start_time = time.perf_counter()
            for images, labels, filenames in tqdm(loader, desc=f"{source}/{attack}"):
                perturbations = attacker(images, labels)
                save_images(str(case_dir), images + perturbations.cpu(), filenames)
            elapsed_seconds = time.perf_counter() - start_time

            save_case_metadata(
                case_dir=case_dir,
                args=args,
                source=source,
                attack=attack,
                eps_label=eps_label,
                eps_value=eps_value,
                elapsed_seconds=elapsed_seconds,
                batch_size=batch_size,
            )
            print(f"Generation time: {elapsed_seconds:.2f}s")
            del attacker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).resize((image_size, image_size)).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def gaussian_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim_per_image(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    channels = clean.shape[1]
    window = gaussian_window(channels, clean.device, clean.dtype)
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(clean, window, padding=padding, groups=channels)
    mu_y = F.conv2d(adv, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(clean * clean, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(adv * adv, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(clean * adv, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return ssim_map.mean(dim=(1, 2, 3))


def psnr_per_image(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    mse = (clean - adv).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def nmse_per_image(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    numerator = (clean - adv).pow(2).sum(dim=(1, 2, 3))
    denominator = clean.pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return numerator / denominator


def tv_per_image(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    delta = adv - clean
    vertical = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    horizontal = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return vertical + horizontal


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def gmsd_per_image(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    clean_gray = rgb_to_gray(clean)
    adv_gray = rgb_to_gray(adv)
    kernel_x = clean.new_tensor(
        [[1.0 / 3.0, 0.0, -1.0 / 3.0],
         [1.0 / 3.0, 0.0, -1.0 / 3.0],
         [1.0 / 3.0, 0.0, -1.0 / 3.0]]
    ).view(1, 1, 3, 3)
    kernel_y = clean.new_tensor(
        [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
         [0.0, 0.0, 0.0],
         [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]]
    ).view(1, 1, 3, 3)
    clean_gx = F.conv2d(clean_gray, kernel_x, padding=1)
    clean_gy = F.conv2d(clean_gray, kernel_y, padding=1)
    adv_gx = F.conv2d(adv_gray, kernel_x, padding=1)
    adv_gy = F.conv2d(adv_gray, kernel_y, padding=1)
    clean_grad = torch.sqrt(clean_gx.pow(2) + clean_gy.pow(2) + 1e-12)
    adv_grad = torch.sqrt(adv_gx.pow(2) + adv_gy.pow(2) + 1e-12)
    c = 0.0026
    gms = (2.0 * clean_grad * adv_grad + c) / (clean_grad.pow(2) + adv_grad.pow(2) + c)
    return gms.flatten(1).std(dim=1)


def symmetric_pad_2d(x: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    height, width = x.shape[-2:]

    def symmetric_indices(length: int, before: int, after: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(-before, length + after, device=device, dtype=torch.float32)
        min_value = -0.5
        max_value = float(length) - 0.5
        span = max_value - min_value
        period = 2.0 * span
        mod = torch.remainder(idx - min_value, period)
        reflected = torch.where(mod >= span, period - mod, mod) + min_value
        return reflected.long()

    x_idx = symmetric_indices(width, left, right, x.device)
    y_idx = symmetric_indices(height, top, bottom, x.device)
    return x.index_select(-1, x_idx).index_select(-2, y_idx)


def exact_symmetric_padding_2d(x: torch.Tensor, kernel: int, stride: int = 1, dilation: int = 1) -> torch.Tensor:
    height, width = x.shape[-2:]
    out_height = math.ceil(height / stride)
    out_width = math.ceil(width / stride)
    pad_row = (out_height - 1) * stride + (kernel - 1) * dilation + 1 - height
    pad_col = (out_width - 1) * stride + (kernel - 1) * dilation + 1 - width
    pad_left = pad_col // 2
    pad_right = pad_col - pad_left
    pad_top = pad_row // 2
    pad_bottom = pad_row - pad_top
    return symmetric_pad_2d(x, (pad_left, pad_right, pad_top, pad_bottom))


def rgb_to_yiq_y(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


class LightweightNLPD(nn.Module):
    """NLPD metric following the normalized Laplacian pyramid formulation."""

    def __init__(self, levels: int = 6):
        super().__init__()
        self.levels = levels
        laplacian_filter = torch.tensor(
            [
                [0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                [0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
            ],
            dtype=torch.float32,
        ).view(1, 1, 5, 5)
        self.register_buffer("laplacian_filter", laplacian_filter)

        dn_filters = [
            [[0, 0.1011, 0], [0.1493, 0, 0.1460], [0, 0.1015, 0.0]],
            [[0, 0.0757, 0], [0.1986, 0, 0.1846], [0, 0.0837, 0]],
            [[0, 0.0477, 0], [0.2138, 0, 0.2243], [0, 0.0467, 0]],
            [[0, 0, 0], [0.2503, 0, 0.2616], [0, 0, 0]],
            [[0, 0, 0], [0.2598, 0, 0.2552], [0, 0, 0]],
            [[0, 0, 0], [0.2215, 0, 0.0717], [0, 0, 0]],
        ]
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        for index, filter_values in enumerate(dn_filters):
            filt = torch.tensor(filter_values, dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer(f"dn_filter_{index}", torch.flip(filt, dims=(-1, -2)))
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32).view(-1, 1, 1, 1, 1))

    def _dn_filter(self, index: int) -> torch.Tensor:
        return getattr(self, f"dn_filter_{index}")

    def pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        current = image
        pyramid_values: List[torch.Tensor] = []
        for index in range(self.levels - 1):
            lowpass = F.conv2d(
                exact_symmetric_padding_2d(current, kernel=5),
                self.laplacian_filter,
                stride=2,
                padding=0,
            )
            odd_height = 2 * lowpass.size(2) - current.size(2)
            odd_width = 2 * lowpass.size(3) - current.size(3)

            lowpass_pad = F.pad(lowpass, (1, 1, 1, 1), mode="replicate")
            zeros = torch.zeros_like(lowpass_pad)
            shuffled = torch.cat([lowpass_pad * 4.0, zeros, zeros, zeros], dim=1)
            upsampled = F.pixel_shuffle(shuffled, upscale_factor=2)
            upsampled_conv = F.conv2d(F.pad(upsampled, (2, 2, 2, 2)), self.laplacian_filter)
            upsampled_conv = upsampled_conv[
                :, :, 2:(upsampled.shape[2] - 2 - odd_height), 2:(upsampled.shape[3] - 2 - odd_width)
            ]

            bandpass = current - upsampled_conv
            norm = F.conv2d(F.pad(torch.abs(bandpass), (1, 1, 1, 1)), self._dn_filter(index))
            pyramid_values.append(bandpass / (self.sigmas[index] + norm))
            current = lowpass

        norm = F.conv2d(F.pad(torch.abs(current), (1, 1, 1, 1)), self._dn_filter(self.levels - 1))
        pyramid_values.append(current / (self.sigmas[self.levels - 1] + norm))
        return pyramid_values

    def forward(self, clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        if clean.shape != adv.shape:
            raise ValueError(f"NLPD inputs must have identical shape, got {clean.shape} and {adv.shape}")
        clean_y = rgb_to_yiq_y(clean)
        adv_y = rgb_to_yiq_y(adv)
        clean_pyramid = self.pyramid(clean_y)
        adv_pyramid = self.pyramid(adv_y)
        distances = [
            torch.sqrt(torch.mean((clean_level - adv_level).pow(2), dim=(1, 2, 3)))
            for clean_level, adv_level in zip(clean_pyramid, adv_pyramid)
        ]
        return torch.stack(distances, dim=1).mean(dim=1)


class OptionalPerceptualMetrics:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.lpips_model = None
        self.nlpd_model = None
        self.nlpd_fn = None
        self.nlpd_pyiqa_model = None
        self.nlpd_builtin_model = None

        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device)
            self.lpips_model.eval()
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad = False
        except Exception as exc:
            if not args.allow_missing_optional_metrics:
                raise RuntimeError(
                    "LPIPS requires the `lpips` package. Install it with `pip install lpips`, "
                    "or rerun with --allow_missing_optional_metrics for a dry run."
                ) from exc

        nlpd_errors = []
        try:
            import piq

            if hasattr(piq, "NLPD"):
                try:
                    self.nlpd_model = piq.NLPD(reduction="none").to(device)
                except TypeError:
                    self.nlpd_model = piq.NLPD().to(device)
                self.nlpd_model.eval()
            elif hasattr(piq, "nlpd"):
                self.nlpd_fn = piq.nlpd
            else:
                raise AttributeError("The installed piq package exposes neither NLPD nor nlpd.")
        except Exception as exc:
            nlpd_errors.append(f"piq: {exc}")

        if self.nlpd_model is None and self.nlpd_fn is None:
            self.nlpd_builtin_model = LightweightNLPD().to(device)
            self.nlpd_builtin_model.eval()

        if self.nlpd_model is None and self.nlpd_fn is None and self.nlpd_builtin_model is None:
            try:
                import pyiqa

                self.nlpd_pyiqa_model = pyiqa.create_metric("nlpd", device=device)
                if hasattr(self.nlpd_pyiqa_model, "eval"):
                    self.nlpd_pyiqa_model.eval()
            except Exception as exc:
                nlpd_errors.append(f"pyiqa: {exc}")

        if (
            self.nlpd_model is None
            and self.nlpd_fn is None
            and self.nlpd_pyiqa_model is None
            and self.nlpd_builtin_model is None
        ):
            warnings.warn(
                "NLPD is unavailable and will be reported as NaN. "
                "Do not install `pyiqa` into the attack environment with plain `pip install pyiqa`, "
                "because recent releases may try to upgrade torch. Use a separate metric-only "
                "environment if NLPD is required. "
                f"Backend errors: {'; '.join(nlpd_errors)}",
                RuntimeWarning,
            )

    def lpips(self, clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        if self.lpips_model is None:
            return clean.new_full((clean.shape[0],), float("nan"))
        with torch.no_grad():
            value = self.lpips_model(clean * 2.0 - 1.0, adv * 2.0 - 1.0)
        return value.view(value.shape[0], -1).mean(dim=1)

    def nlpd(self, clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        if (
            self.nlpd_model is None
            and self.nlpd_fn is None
            and self.nlpd_pyiqa_model is None
            and self.nlpd_builtin_model is None
        ):
            return clean.new_full((clean.shape[0],), float("nan"))
        with torch.no_grad():
            if self.nlpd_model is not None:
                value = self.nlpd_model(clean, adv)
            elif self.nlpd_builtin_model is not None:
                value = self.nlpd_builtin_model(clean, adv)
            elif self.nlpd_pyiqa_model is not None:
                value = self.nlpd_pyiqa_model(clean, adv)
                if torch.is_tensor(value) and value.ndim == 0 and clean.shape[0] > 1:
                    per_image_values = [
                        self.nlpd_pyiqa_model(clean[i: i + 1], adv[i: i + 1]).reshape(-1).mean()
                        for i in range(clean.shape[0])
                    ]
                    return torch.stack(per_image_values)
            else:
                try:
                    value = self.nlpd_fn(clean, adv, reduction="none")
                except TypeError:
                    value = self.nlpd_fn(clean, adv)
        if not torch.is_tensor(value):
            value = clean.new_tensor(value)
        if value.ndim == 0:
            return value.repeat(clean.shape[0])
        return value.view(value.shape[0], -1).mean(dim=1)


def compute_batch_metrics(
    clean: torch.Tensor,
    adv: torch.Tensor,
    optional_metrics: OptionalPerceptualMetrics,
) -> Dict[str, torch.Tensor]:
    return {
        "ssim": ssim_per_image(clean, adv),
        "psnr": psnr_per_image(clean, adv),
        "lpips": optional_metrics.lpips(clean, adv),
        "tv": tv_per_image(clean, adv),
        "nmse": nmse_per_image(clean, adv),
        "nlpd": optional_metrics.nlpd(clean, adv),
        "gmsd": gmsd_per_image(clean, adv),
    }


def load_generation_seconds(case_dir: Path) -> float:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.is_file():
        return float("nan")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return float(data.get("elapsed_seconds", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def evaluate_one_case(
    args: argparse.Namespace,
    source: str,
    attack: str,
    optional_metrics: OptionalPerceptualMetrics,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    eps_label, eps_value = parse_epsilon_token(args.eps)
    case_dir = case_output_dir(args, source, attack)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing adversarial examples for case: {case_dir}")

    filenames = expected_filenames(args.input_dir)
    metric_sums = {metric: 0.0 for metric in METRIC_FIELDS}
    metric_sq_sums = {metric: 0.0 for metric in METRIC_FIELDS}
    per_image_rows: List[Dict[str, object]] = []
    total = 0

    for start in tqdm(range(0, len(filenames), args.eval_batchsize), desc=f"metrics/{source}/{attack}"):
        batch_names = filenames[start: start + args.eval_batchsize]
        clean_images = []
        adv_images = []
        for filename in batch_names:
            clean_path = Path(args.input_dir) / "images" / filename
            adv_path = case_dir / filename
            if not adv_path.is_file():
                raise FileNotFoundError(f"Missing adversarial image: {adv_path}")
            clean_images.append(load_image_tensor(clean_path, args.image_size))
            adv_images.append(load_image_tensor(adv_path, args.image_size))

        clean = torch.stack(clean_images, dim=0).to(device)
        adv = torch.stack(adv_images, dim=0).to(device)
        batch_metrics = compute_batch_metrics(clean, adv, optional_metrics)
        batch_size = len(batch_names)

        batch_cpu = {
            metric: values.detach().cpu().numpy().astype(np.float64)
            for metric, values in batch_metrics.items()
        }
        for idx, filename in enumerate(batch_names):
            row = {
                "source": source,
                "source_label": display_source(source),
                "attack": attack,
                "attack_label": display_attack(attack),
                "epsilon_label": eps_label,
                "epsilon": eps_value,
                "filename": filename,
            }
            for metric in METRIC_FIELDS:
                row[metric] = float(batch_cpu[metric][idx])
            per_image_rows.append(row)

        for metric in METRIC_FIELDS:
            values = batch_cpu[metric]
            metric_sums[metric] += float(np.nansum(values))
            metric_sq_sums[metric] += float(np.nansum(values ** 2))
        total += batch_size

    summary: Dict[str, object] = {
        "source": source,
        "source_label": display_source(source),
        "attack": attack,
        "attack_label": display_attack(attack),
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": args.alpha if args.alpha is not None else eps_value / float(args.epoch),
        "num_images": total,
        "generation_seconds": load_generation_seconds(case_dir),
    }
    for metric in METRIC_FIELDS:
        mean = metric_sums[metric] / max(total, 1)
        variance = metric_sq_sums[metric] / max(total, 1) - mean ** 2
        summary[metric] = mean
        summary[f"{metric}_std"] = math.sqrt(max(variance, 0.0))
    return summary, per_image_rows


def summary_fieldnames() -> List[str]:
    fields = [
        "source",
        "source_label",
        "attack",
        "attack_label",
        "epsilon_label",
        "epsilon",
        "alpha",
        "num_images",
        "generation_seconds",
    ]
    for metric in METRIC_FIELDS:
        fields.append(metric)
        fields.append(f"{metric}_std")
    return fields


def per_image_fieldnames() -> List[str]:
    return [
        "source",
        "source_label",
        "attack",
        "attack_label",
        "epsilon_label",
        "epsilon",
        "filename",
    ] + METRIC_FIELDS


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    if math.isnan(value):
                        formatted[key] = "nan"
                    else:
                        formatted[key] = f"{value:.8f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def as_float(row: Dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def sort_rows(rows: List[Dict[str, object]], sources: List[str], attacks: List[str]) -> List[Dict[str, object]]:
    source_rank = {source: idx for idx, source in enumerate(sources)}
    attack_rank = {attack: idx for idx, attack in enumerate(attacks)}
    return sorted(
        rows,
        key=lambda row: (
            source_rank.get(str(row.get("source", "")), 10_000),
            attack_rank.get(str(row.get("attack", "")), 10_000),
        ),
    )


def load_existing_summary(result_csv: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not result_csv.is_file():
        return {}
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {case_key(row["source"], row["attack"]): row for row in rows}


def write_analysis(analysis_file: Path, rows: List[Dict[str, object]], sources: List[str], attacks: List[str]) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows, sources, attacks)
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Perceptual quality of final adversarial examples\n")
        f.write("=" * 56 + "\n")
        f.write("Metrics compare clean images with final adversarial images.\n")
        f.write("TV is computed on the perturbation x_adv - x.\n\n")
        for row in rows:
            f.write(
                f"{row['source_label']} | {row['attack_label']}: "
                f"SSIM={as_float(row, 'ssim'):.4f}, "
                f"PSNR={as_float(row, 'psnr'):.2f}, "
                f"LPIPS={as_float(row, 'lpips'):.4f}, "
                f"TV={as_float(row, 'tv'):.6f}, "
                f"NMSE={as_float(row, 'nmse'):.6f}, "
                f"NLPD={as_float(row, 'nlpd'):.4f}, "
                f"GMSD={as_float(row, 'gmsd'):.6f}, "
                f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
            )


def format_latex_metric(row: Dict[str, object], metric: str) -> str:
    value = as_float(row, metric)
    if math.isnan(value):
        return "--"
    if metric == "psnr":
        return f"{value:.2f}"
    if metric in {"ssim", "lpips", "nlpd"}:
        return f"{value:.4f}"
    return f"{value:.6f}"


def write_latex_rows(summary_tex: Path, rows: List[Dict[str, object]], sources: List[str], attacks: List[str]) -> None:
    summary_tex.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows, sources, attacks)
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_perceptual_quality_eda.py\n")
        f.write("% Columns: Attack, SSIM, PSNR, LPIPS, TV, NMSE, NLPD, GMSD.\n")
        if len(sources) == 1:
            for row in rows:
                line = (
                    f"{row['attack_label']} & "
                    + " & ".join(format_latex_metric(row, metric) for metric in METRIC_FIELDS)
                    + r" \\"
                )
                f.write(line + "\n")
        else:
            for source in sources:
                source_rows = [row for row in rows if row["source"] == source]
                if not source_rows:
                    continue
                f.write(rf"\multirow{{{len(source_rows)}}}{{*}}{{{display_source(source)}}}" + "\n")
                for idx, row in enumerate(source_rows):
                    prefix = "& " if idx > 0 else "& "
                    line = (
                        f"{prefix}{row['attack_label']} & "
                        + " & ".join(format_latex_metric(row, metric) for metric in METRIC_FIELDS)
                        + r" \\"
                    )
                    f.write(line + "\n")
                if source != sources[-1]:
                    f.write(r"\midrule" + "\n")


def evaluate_cases(args: argparse.Namespace, sources: List[str], attacks: List[str]) -> None:
    result_csv, per_image_csv, summary_tex, analysis_file = output_paths(args)
    existing = load_existing_summary(result_csv)
    summary_rows: Dict[Tuple[str, str], Dict[str, object]] = dict(existing)
    all_per_image_rows: List[Dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optional_metrics = OptionalPerceptualMetrics(args, device)

    for source in sources:
        for attack in attacks:
            key = case_key(source, attack)
            if args.reuse_existing and key in summary_rows:
                print(f"Skipping existing metrics: {source}/{attack}")
                continue
            print("\n" + "=" * 80)
            print(f"Computing metrics for source={source}, attack={attack}")
            summary, per_image_rows = evaluate_one_case(args, source, attack, optional_metrics, device)
            summary_rows[key] = summary
            all_per_image_rows.extend(per_image_rows)

            ordered = sort_rows(list(summary_rows.values()), sources, attacks)
            write_csv(result_csv, ordered, summary_fieldnames())
            write_analysis(analysis_file, ordered, sources, attacks)
            write_latex_rows(summary_tex, ordered, sources, attacks)

    ordered = sort_rows(list(summary_rows.values()), sources, attacks)
    write_csv(result_csv, ordered, summary_fieldnames())
    write_analysis(analysis_file, ordered, sources, attacks)
    write_latex_rows(summary_tex, ordered, sources, attacks)
    if all_per_image_rows:
        write_csv(per_image_csv, all_per_image_rows, per_image_fieldnames())


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final adversarial examples and compute perceptual quality "
            "metrics against clean images."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./perceptual_quality")
    parser.add_argument(
        "--batchsize",
        default=0,
        type=int,
        help="Generation batch size. Use 0 for automatic source/attack-specific values.",
    )
    parser.add_argument("--eval_batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)
    parser.add_argument("--reuse_existing", action="store_true")

    parser.add_argument("--eps", default=DEFAULT_EPSILON)
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--decay", default=1.0, type=float)
    parser.add_argument("--norm", default="linfty")
    parser.add_argument("--loss", default="crossentropy")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--mesh_width", default=3, type=int)
    parser.add_argument("--mesh_height", default=3, type=int)
    parser.add_argument("--noise_scale", default=0.45, type=float)
    parser.add_argument("--num_warping", default=25, type=int)

    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--allow_missing_optional_metrics", action="store_true")

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--per_image_csv", default="")
    parser.add_argument("--summary_tex", default="")
    parser.add_argument("--analysis_file", default="")
    return parser


def main() -> None:
    global GLOBAL_SEED
    parser = get_parser()
    args = parser.parse_args()
    GLOBAL_SEED = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    set_seed(args.seed)

    sources = parse_list(args.sources)
    attacks = parse_attacks(args.attacks)

    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, attacks)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources, attacks)


if __name__ == "__main__":
    main()
