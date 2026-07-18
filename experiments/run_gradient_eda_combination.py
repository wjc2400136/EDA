import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()

import transferattack
from transferattack.gradient.emifgsm import EMIFGSM
from transferattack.gradient.gaa import GAA
from transferattack.gradient.mef import MEF
from transferattack.gradient.pgn import PGN
from transferattack.gradient.vmifgsm import VMIFGSM
from transferattack.input_transformation.eda import (
    TPSGrid,
    center_grid as make_center_grid,
    regular_grid,
)
from transferattack.utils import AdvDataset, load_pretrained_model, save_images, wrap_model


GLOBAL_SEED = 42
DEFAULT_SOURCE = "resnet18"
DEFAULT_METHODS = "vmifgsm,emifgsm,pgn,mef,gaa"
DEFAULT_VARIANTS = "base,eda"

METHOD_DISPLAY = {
    "vmifgsm": "VMI-FGSM",
    "emifgsm": "EMI-FGSM",
    "pgn": "PGN",
    "mef": "MEF",
    "gaa": "GAA",
}

SOURCE_DISPLAY = {
    "resnet18": "RN-18",
    "inception_v3": "Inc-v3",
    "inception_v4": "Inc-v4",
    "inception_resnet_v2": "IncRes-v2",
}

PAPER_CNN_MODELS = [
    "vgg19",
    "resnet18",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "densenet121",
    "mobilenet_v2",
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
]
PAPER_VIT_MODELS = [
    "vit_base_patch16_224",
    "deit_base_distilled_patch16_224",
    "levit_256",
    "pit_b_224",
    "cait_s24_224",
    "convit_base",
    "tnt_s_patch16_224",
    "visformer_small",
    "swin_tiny_patch4_window7_224",
    "coat_tiny",
]
EVAL_CNN_LOAD_LIST = [
    "vgg19",
    "resnet18",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "densenet121",
    "mobilenet_v2",
    "inception_v3",
]
EVAL_VIT_LOAD_LIST = [
    "inception_v4",
    "inception_resnet_v2",
] + PAPER_VIT_MODELS
PAPER_EVAL_MODELS = PAPER_CNN_MODELS + PAPER_VIT_MODELS

BASE_CLASS = {
    "vmifgsm": VMIFGSM,
    "emifgsm": EMIFGSM,
    "pgn": PGN,
    "mef": MEF,
    "gaa": GAA,
}

DEFAULT_BATCHSIZE = {
    "vmifgsm": {"base": 32, "eda": 32},
    "emifgsm": {"base": 32, "eda": 32},
    "pgn": {"base": 32, "eda": 32},
    "mef": {"base": 32, "eda": 32},
    "gaa": {"base": 32, "eda": 32},
}


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


def parse_methods(raw_values: str) -> List[str]:
    methods = [item.strip().lower() for item in raw_values.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one gradient method is required.")
    for method in methods:
        if method not in BASE_CLASS:
            raise ValueError(f"Unsupported method: {method}. Supported: {sorted(BASE_CLASS)}")
    return methods


def parse_variants(raw_values: str) -> List[str]:
    variants = [item.strip().lower() for item in raw_values.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one variant is required.")
    for variant in variants:
        if variant not in {"base", "eda"}:
            raise ValueError("Variants must be selected from: base, eda")
    return variants


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def existing_expected_images(directory: Path, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if (directory / filename).is_file()]


def missing_expected_images(directory: Path, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if not (directory / filename).is_file()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def filter_dataset_by_filenames(dataset: AdvDataset, filenames: Iterable[str]) -> Subset:
    wanted = set(filenames)
    ordered_filenames = list(dataset.f2l.keys())
    indices = [idx for idx, filename in enumerate(ordered_filenames) if filename in wanted]
    if len(indices) != len(wanted):
        missing = sorted(wanted.difference(ordered_filenames))
        raise ValueError(f"Some filenames are missing from labels.csv: {missing[:5]}")
    return Subset(dataset, indices)


def make_loader(dataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method.upper())


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_variant(method: str, variant: str) -> str:
    if variant == "base":
        return display_method(method)
    return "+EDA"


def case_output_dir(args: argparse.Namespace, source: str, method: str, variant: str) -> Path:
    return Path(args.output_dir) / source / method / variant


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "gradient_eda_combination_results.csv"
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "gradient_eda_combination_rows.tex"
    table_tex = Path(args.table_tex) if args.table_tex else output_dir / "gradient_eda_combination_table.tex"
    analysis_file = Path(args.analysis_file) if args.analysis_file else output_dir / "gradient_eda_combination_analysis.txt"
    return result_csv, summary_tex, table_tex, analysis_file


def resolve_batchsize(args: argparse.Namespace, method: str, variant: str) -> int:
    if args.batchsize > 0:
        return args.batchsize
    return DEFAULT_BATCHSIZE.get(method, {}).get(variant, 1)


class EDAInputTransformMixin:
    def init_eda_transform(
        self,
        mesh_width=3,
        mesh_height=3,
        noise_scale=0.45,
        use_dual_grid=True,
        move_edge=True,
        use_appearance=True,
        brightness_min=0.0,
        brightness_max=2.0,
        noise_std=0.05,
    ):
        self.eda_mesh_width = mesh_width
        self.eda_mesh_height = mesh_height
        self.eda_noise_scale = noise_scale
        self.eda_use_dual_grid = use_dual_grid
        self.eda_move_edge = move_edge
        self.eda_use_appearance = use_appearance
        self.eda_brightness_min = brightness_min
        self.eda_brightness_max = brightness_max
        self.eda_noise_std = noise_std
        self.eda_full_grid = regular_grid(mesh_width, mesh_height, self.device)
        self.eda_center_grid = make_center_grid(mesh_width, mesh_height, self.device) if use_dual_grid else None
        self.eda_center_mesh_width = max(mesh_width - 1, 0)
        self.eda_center_mesh_height = max(mesh_height - 1, 0)
        self.eda_tps_cache = {}

    def select_eda_layout(self):
        if (
            self.eda_use_dual_grid
            and self.eda_center_grid is not None
            and self.eda_center_grid.numel() > 0
            and random.random() < 0.5
        ):
            return self.eda_center_grid, self.eda_center_mesh_width, self.eda_center_mesh_height
        return self.eda_full_grid, self.eda_mesh_width, self.eda_mesh_height

    def sample_eda_offsets(self, points, mesh_width, mesh_height):
        offsets = (torch.rand_like(points) - 0.5) * self.eda_noise_scale
        if self.eda_move_edge:
            return offsets
        if points.shape[0] == mesh_width * mesh_height:
            boundary = (
                torch.isclose(points[:, 0], points[:, 0].new_tensor(-1.0))
                | torch.isclose(points[:, 0], points[:, 0].new_tensor(1.0))
                | torch.isclose(points[:, 1], points[:, 1].new_tensor(-1.0))
                | torch.isclose(points[:, 1], points[:, 1].new_tensor(1.0))
            )
            offsets = offsets.clone()
            offsets[boundary] = 0.0
        return offsets

    def eda_padding_amounts(self, height, width):
        max_offset_norm = 0.5 * self.eda_noise_scale
        pad_w = int(torch.ceil(torch.tensor(max_offset_norm * (width - 1) / 2)).item())
        pad_h = int(torch.ceil(torch.tensor(max_offset_norm * (height - 1) / 2)).item())
        return pad_h, pad_w

    def eda_to_pixel_coords(self, points, height, width):
        scale = points.new_tensor([(width - 1) / 2.0, (height - 1) / 2.0])
        return (points + 1.0) * scale

    def eda_to_normalized_coords(self, points, height, width):
        scale = points.new_tensor([2.0 / (width - 1), 2.0 / (height - 1)])
        return points * scale - 1.0

    def get_eda_tps_grid(self, height, width, dtype):
        cache_key = (height, width, str(self.device), str(dtype))
        if cache_key not in self.eda_tps_cache:
            self.eda_tps_cache[cache_key] = TPSGrid(
                size=(height, width),
                device=self.device,
                dtype=dtype,
            )
        return self.eda_tps_cache[cache_key]

    def eda_elastic_warp(self, x):
        batch_size, _, height, width = x.size()
        source_points, mesh_width, mesh_height = self.select_eda_layout()
        source_points = source_points.to(device=x.device, dtype=x.dtype)
        target_points = source_points + self.sample_eda_offsets(source_points, mesh_width, mesh_height)

        pad_h, pad_w = self.eda_padding_amounts(height, width)
        padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        canvas_h = height + 2 * pad_h
        canvas_w = width + 2 * pad_w

        source_pixel = self.eda_to_pixel_coords(source_points, height, width)
        target_pixel = self.eda_to_pixel_coords(target_points, height, width)
        pad_vector = source_points.new_tensor([pad_w, pad_h])

        source_canvas = self.eda_to_normalized_coords(source_pixel + pad_vector, canvas_h, canvas_w)
        target_canvas = self.eda_to_normalized_coords(target_pixel + pad_vector, canvas_h, canvas_w)

        tps = self.get_eda_tps_grid(canvas_h, canvas_w, x.dtype)
        sampling_grid = tps(source_canvas[None, ...], target_canvas[None, ...])
        sampling_grid = sampling_grid.repeat(batch_size, 1, 1, 1)

        warped_canvas = F.grid_sample(
            padded,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped_canvas[:, :, pad_h : pad_h + height, pad_w : pad_w + width]

    def eda_brightness(self, x):
        factor = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(
            self.eda_brightness_min,
            self.eda_brightness_max,
        )
        return torch.clamp(x * factor, 0.0, 1.0)

    def eda_channel_noise(self, x):
        return torch.clamp(x + torch.randn_like(x) * self.eda_noise_std, 0.0, 1.0)

    def eda_apply_appearance(self, x):
        if not self.eda_use_appearance:
            return x
        transform = random.choice([self.eda_brightness, self.eda_channel_noise])
        return transform(x)

    def eda_transform(self, x):
        return self.eda_apply_appearance(self.eda_elastic_warp(x))


class EDAVMIFGSM(EDAInputTransformMixin, VMIFGSM):
    def __init__(self, *args, **kwargs):
        eda_kwargs = pop_eda_kwargs(kwargs)
        super().__init__(*args, attack="VMI-FGSM+EDA", **kwargs)
        self.init_eda_transform(**eda_kwargs)

    def transform(self, x, **kwargs):
        return self.eda_transform(x)


class EDAEMIFGSM(EDAInputTransformMixin, EMIFGSM):
    def __init__(self, *args, **kwargs):
        eda_kwargs = pop_eda_kwargs(kwargs)
        super().__init__(*args, attack="EMI-FGSM+EDA", **kwargs)
        self.init_eda_transform(**eda_kwargs)

    def transform(self, x, grad, **kwargs):
        sampled = EMIFGSM.transform(self, x, grad, **kwargs)
        return self.eda_transform(sampled)


class EDAPGN(EDAInputTransformMixin, PGN):
    def __init__(self, *args, **kwargs):
        eda_kwargs = pop_eda_kwargs(kwargs)
        super().__init__(*args, attack="PGN+EDA", **kwargs)
        self.init_eda_transform(**eda_kwargs)

    def transform(self, x, **kwargs):
        return self.eda_transform(x)

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        averaged_gradient = 0
        for _ in range(self.num_neighbor):
            noise = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device)
            x_near_base = data + delta + noise
            x_near = self.eda_transform(x_near_base)

            logits = self.get_logits(x_near)
            loss = self.get_loss(logits, label)
            g_1 = self.get_grad(loss, delta)

            x_next_base = data + delta + noise + self.alpha * (
                -g_1 / (torch.abs(g_1).mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            )
            x_next = self.eda_transform(x_next_base)

            logits = self.get_logits(x_next)
            loss = self.get_loss(logits, label)
            g_2 = self.get_grad(loss, delta)

            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2

        return averaged_gradient / self.num_neighbor


class EDAMEF(EDAInputTransformMixin, MEF):
    def __init__(self, *args, **kwargs):
        eda_kwargs = pop_eda_kwargs(kwargs)
        super().__init__(*args, attack="MEF+EDA", **kwargs)
        self.init_eda_transform(**eda_kwargs)

    def get_points_gradient(self, data, delta, label, **kwargs):
        b, c, h, w = data.shape
        loss_list = torch.zeros([self.num_neighbor, b], device=self.device)
        grad_list = torch.zeros([self.num_neighbor, b, c, h, w], device=self.device)
        for i in range(self.num_neighbor):
            x_min = (data + delta[i]).detach().requires_grad_(True)
            transformed = self.eda_transform(x_min)
            logits = self.get_logits(transformed)
            loss_list[i] = self.get_loss(logits, label)
            grad_list[i] = torch.autograd.grad(
                loss_list[i].mean(),
                x_min,
                retain_graph=False,
                create_graph=False,
            )[0]
        return (1 / self.num_neighbor) * grad_list


class EDAGAA(EDAInputTransformMixin, GAA):
    def __init__(self, *args, **kwargs):
        eda_kwargs = pop_eda_kwargs(kwargs)
        super().__init__(*args, attack="GAA+EDA", **kwargs)
        self.init_eda_transform(**eda_kwargs)

    def calculate_gradient(self, x, label):
        x.requires_grad_(True)
        logits = self.model(self.eda_transform(x))
        loss = self.get_loss(logits, label)
        return torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]


EDA_CLASS = {
    "vmifgsm": EDAVMIFGSM,
    "emifgsm": EDAEMIFGSM,
    "pgn": EDAPGN,
    "mef": EDAMEF,
    "gaa": EDAGAA,
}


def pop_eda_kwargs(kwargs: Dict[str, object]) -> Dict[str, object]:
    keys = {
        "mesh_width",
        "mesh_height",
        "noise_scale",
        "use_dual_grid",
        "move_edge",
        "use_appearance",
        "brightness_min",
        "brightness_max",
        "noise_std",
    }
    return {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in keys}


def build_attacker(
    args: argparse.Namespace,
    source: str,
    method: str,
    variant: str,
):
    attack_class = BASE_CLASS[method] if variant == "base" else EDA_CLASS[method]
    kwargs = {
        "model_name": source,
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "targeted": args.targeted,
        "random_start": args.random_start,
        "norm": args.norm,
        "seed": args.seed,
    }
    if args.decay is not None:
        kwargs["decay"] = args.decay
    if args.loss:
        kwargs["loss"] = args.loss
    if variant == "eda":
        kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            use_dual_grid=not args.single_grid,
            move_edge=not args.fixed_edge,
            use_appearance=not args.no_appearance,
            brightness_min=args.brightness_min,
            brightness_max=args.brightness_max,
            noise_std=args.noise_std,
        )
    return attack_class(**kwargs)


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    method: str,
    variant: str,
    elapsed_seconds: float,
    batch_size: int,
    completed_images: int,
    expected_images: int,
) -> None:
    metadata = {
        "source": source,
        "source_label": display_source(source),
        "method": method,
        "method_label": display_method(method),
        "variant": variant,
        "variant_label": display_variant(method, variant),
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": args.targeted,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "completed_images": completed_images,
        "expected_images": expected_images,
        "complete": completed_images >= expected_images,
    }
    if variant == "eda":
        metadata.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            brightness_range=[args.brightness_min, args.brightness_max],
            noise_std=args.noise_std,
        )
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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


def generate_cases(
    args: argparse.Namespace,
    sources: List[str],
    methods: List[str],
    variants: List[str],
) -> None:
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for method in methods:
            for variant in variants:
                set_seed(args.seed)
                case_dir = case_output_dir(args, source, method, variant)
                existing = existing_expected_images(case_dir, expected)
                missing = missing_expected_images(case_dir, expected)
                if args.reuse_existing and not missing:
                    print(f"Skipping existing case: {source}/{method}/{variant}")
                    continue
                if case_dir.exists() and not args.reuse_existing:
                    shutil.rmtree(case_dir)
                    existing = []
                    missing = list(expected)
                case_dir.mkdir(parents=True, exist_ok=True)

                print("\n" + "=" * 80)
                print(f"Generating source={source}, method={method}, variant={variant}")
                print(f"Output: {case_dir}")

                attacker = build_attacker(args, source, method, variant)
                dataset = AdvDataset(
                    input_dir=args.input_dir,
                    output_dir=str(case_dir),
                    targeted=args.targeted,
                    eval=False,
                )
                if args.reuse_existing and existing:
                    print(
                        f"Resume mode: found {len(existing)}/{len(expected)} images; "
                        f"generating {len(missing)} missing images."
                    )
                    dataset = filter_dataset_by_filenames(dataset, missing)

                batch_size = resolve_batchsize(args, method, variant)
                print(f"Generation batchsize: {batch_size}")
                loader = make_loader(dataset, batch_size, args)

                start_time = time.perf_counter()
                for images, labels, filenames in tqdm(loader, desc=f"{source}/{method}/{variant}"):
                    perturbations = attacker(images, labels)
                    save_images(str(case_dir), images + perturbations.cpu(), filenames)
                elapsed_seconds = time.perf_counter() - start_time
                final_completed = len(existing_expected_images(case_dir, expected))

                save_case_metadata(
                    case_dir=case_dir,
                    args=args,
                    source=source,
                    method=method,
                    variant=variant,
                    elapsed_seconds=elapsed_seconds,
                    batch_size=batch_size,
                    completed_images=final_completed,
                    expected_images=len(expected),
                )
                print(
                    f"Generation time: {elapsed_seconds:.2f}s; "
                    f"completed: {final_completed}/{len(expected)}"
                )
                del attacker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def eval_asr(model, loader: DataLoader, targeted: bool, device: torch.device) -> float:
    success = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            if targeted:
                labels = labels[1]
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.numpy()
            if targeted:
                success += (predictions == labels_np).sum()
            else:
                success += (predictions != labels_np).sum()
            total += labels_np.shape[0]
    return 100.0 * success / total


def evaluate_one_case(
    args: argparse.Namespace,
    source: str,
    method: str,
    variant: str,
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, source, method, variant)
    expected = expected_filenames(args.input_dir)
    if not has_expected_images(case_dir, expected):
        missing_count = len(missing_expected_images(case_dir, expected))
        raise FileNotFoundError(f"Incomplete adversarial examples: {case_dir}; missing {missing_count}")

    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=str(case_dir),
        targeted=args.targeted,
        eval=True,
    )
    loader = make_loader(dataset, args.eval_batchsize, args)

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(EVAL_CNN_LOAD_LIST, EVAL_VIT_LOAD_LIST):
        print(f"Evaluating source={source}, method={method}, variant={variant} on {model_name}")
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, targeted=args.targeted, device=device)
        results[model_name] = asr
        print(f"{source}/{method}/{variant} -> {model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(source)
    cnn_values = [
        results[name]
        for name in PAPER_CNN_MODELS
        if name in results and name not in source_models
    ]
    vit_values = [results[name] for name in PAPER_VIT_MODELS if name in results]
    transfer_values = [
        results[name]
        for name in PAPER_EVAL_MODELS
        if name in results and name not in source_models
    ]

    row: Dict[str, object] = {
        "source": source,
        "source_label": display_source(source),
        "method": method,
        "method_label": display_method(method),
        "variant": variant,
        "variant_label": display_variant(method, variant),
        "generation_seconds": load_generation_seconds(case_dir),
        "cnn_avg_excl_source": float(np.mean(cnn_values)),
        "vit_avg": float(np.mean(vit_values)),
        "group_avg": float((np.mean(cnn_values) + np.mean(vit_values)) / 2.0),
        "target_weighted_transfer_avg": float(np.mean(transfer_values)),
    }
    row.update(results)
    return row


def result_fieldnames() -> List[str]:
    return [
        "source",
        "source_label",
        "method",
        "method_label",
        "variant",
        "variant_label",
        "generation_seconds",
    ] + PAPER_EVAL_MODELS + [
        "cnn_avg_excl_source",
        "vit_avg",
        "group_avg",
        "target_weighted_transfer_avg",
    ]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in result_fieldnames()})


def load_existing_rows(path: Path) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    if not path.is_file():
        return {}
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["source"], row["method"], row["variant"])] = row
    return rows


def as_float(row: Dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def sort_rows(
    rows: List[Dict[str, object]],
    sources: List[str],
    methods: List[str],
    variants: List[str],
) -> List[Dict[str, object]]:
    source_rank = {name: idx for idx, name in enumerate(sources)}
    method_rank = {name: idx for idx, name in enumerate(methods)}
    variant_rank = {name: idx for idx, name in enumerate(variants)}
    return sorted(
        rows,
        key=lambda row: (
            source_rank.get(str(row["source"]), 10**6),
            method_rank.get(str(row["method"]), 10**6),
            variant_rank.get(str(row["variant"]), 10**6),
        ),
    )


def format_gain(value: float) -> str:
    return r"\textcolor{red}{\tiny" + f"{value:.1f}" + r"$\uparrow$}"


def build_latex_rows(rows: List[Dict[str, object]], sources: List[str], methods: List[str]) -> str:
    row_map = {(row["source"], row["method"], row["variant"]): row for row in rows}
    lines = []
    for source in sources:
        if len(sources) > 1:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{4}{c}{Source: " + display_source(source) + r"} \\")
        for method in methods:
            base = row_map.get((source, method, "base"))
            eda = row_map.get((source, method, "eda"))
            if base is None and eda is None:
                continue
            if base is not None:
                lines.append(
                    f"{display_method(method)}"
                    f"& {as_float(base, 'cnn_avg_excl_source'):.1f}"
                    f"& {as_float(base, 'vit_avg'):.1f}"
                    f"& {as_float(base, 'group_avg'):.1f} \\\\"
                )
            if eda is not None:
                cnn = as_float(eda, "cnn_avg_excl_source")
                vit = as_float(eda, "vit_avg")
                avg = as_float(eda, "group_avg")
                if base is not None:
                    cnn_gain = format_gain(cnn - as_float(base, "cnn_avg_excl_source"))
                    vit_gain = format_gain(vit - as_float(base, "vit_avg"))
                    avg_gain = format_gain(avg - as_float(base, "group_avg"))
                else:
                    cnn_gain = vit_gain = avg_gain = ""
                lines.append(
                    r"\textbf{+EDA}"
                    f"& \\textbf{{{cnn:.1f}}}{cnn_gain}"
                    f"& \\textbf{{{vit:.1f}}}{vit_gain}"
                    f"& \\textbf{{{avg:.1f}}}{avg_gain} \\\\"
                )
            if method != methods[-1]:
                lines.append(r"\midrule")
    return "\n".join(lines) + ("\n" if lines else "")


def write_latex_rows(path: Path, rows: List[Dict[str, object]], sources: List[str], methods: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_latex_rows(rows, sources, methods))


def write_latex_table(path: Path, rows: List[Dict[str, object]], sources: List[str], methods: List[str]) -> None:
    table_rows = build_latex_rows(rows, sources, methods)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table*}[!htb]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Combination with gradient-based attacks.}" + "\n")
        f.write(r"\label{Combination}" + "\n")
        f.write(r"\resizebox{0.5\textwidth}{!}{" + "\n")
        f.write(r"\begin{tabular}{c *{3}{c}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Attacks & CNNs & ViTs & Avg\\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(table_rows)
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table*}" + "\n")


def write_analysis(path: Path, rows: List[Dict[str, object]], sources: List[str], methods: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_map = {(row["source"], row["method"], row["variant"]): row for row in rows}
    with open(path, "w", encoding="utf-8") as f:
        f.write("Gradient-attack combination with EDA\n")
        f.write("====================================\n")
        f.write("CNN averages exclude source-identical targets whenever applicable.\n")
        f.write("Avg. is the unweighted mean of the reported CNN and ViT group averages.\n\n")
        for source in sources:
            f.write(f"Source: {display_source(source)} ({source})\n")
            for method in methods:
                base = row_map.get((source, method, "base"))
                eda = row_map.get((source, method, "eda"))
                if base is None or eda is None:
                    continue
                f.write(
                    f"  {display_method(method)} + EDA: "
                    f"CNN {as_float(base, 'cnn_avg_excl_source'):.2f}->{as_float(eda, 'cnn_avg_excl_source'):.2f}, "
                    f"ViT {as_float(base, 'vit_avg'):.2f}->{as_float(eda, 'vit_avg'):.2f}, "
                    f"Avg {as_float(base, 'group_avg'):.2f}->{as_float(eda, 'group_avg'):.2f}, "
                    f"generation={as_float(eda, 'generation_seconds'):.2f}s\n"
                )
            f.write("\n")


def evaluate_cases(
    args: argparse.Namespace,
    sources: List[str],
    methods: List[str],
    variants: List[str],
) -> None:
    result_csv, summary_tex, table_tex, analysis_file = output_paths(args)
    rows = dict(load_existing_rows(result_csv))
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")

    for source in sources:
        for method in methods:
            for variant in variants:
                key = (source, method, variant)
                if args.reuse_existing and key in rows:
                    print(f"Skipping existing evaluation: {source}/{method}/{variant}")
                    continue
                print("\n" + "=" * 80)
                print(f"Evaluating source={source}, method={method}, variant={variant}")
                rows[key] = evaluate_one_case(args, source, method, variant, device)
                ordered_rows = sort_rows(list(rows.values()), sources, methods, variants)
                write_csv(result_csv, ordered_rows)
                write_latex_rows(summary_tex, ordered_rows, sources, methods)
                write_latex_table(table_tex, ordered_rows, sources, methods)
                write_analysis(analysis_file, ordered_rows, sources, methods)

    ordered_rows = sort_rows(list(rows.values()), sources, methods, variants)
    write_csv(result_csv, ordered_rows)
    write_latex_rows(summary_tex, ordered_rows, sources, methods)
    write_latex_table(table_tex, ordered_rows, sources, methods)
    write_analysis(analysis_file, ordered_rows, sources, methods)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate gradient-attack combinations with EDA."
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--sources", default=DEFAULT_SOURCE)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./gradient_eda_combination")
    parser.add_argument("--batchsize", default=0, type=int)
    parser.add_argument("--eval_batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)
    parser.add_argument("--reuse_existing", action="store_true")

    parser.add_argument("--eps", default=16 / 255, type=float)
    parser.add_argument("--alpha", default=1.6 / 255, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--decay", default=None, type=float)
    parser.add_argument("--norm", default="linfty")
    parser.add_argument("--loss", default="")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--mesh_width", default=3, type=int)
    parser.add_argument("--mesh_height", default=3, type=int)
    parser.add_argument("--noise_scale", default=0.45, type=float)
    parser.add_argument("--single_grid", action="store_true")
    parser.add_argument("--fixed_edge", action="store_true")
    parser.add_argument("--no_appearance", action="store_true")
    parser.add_argument("--brightness_min", default=0.0, type=float)
    parser.add_argument("--brightness_max", default=2.0, type=float)
    parser.add_argument("--noise_std", default=0.05, type=float)

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--summary_tex", default="")
    parser.add_argument("--table_tex", default="")
    parser.add_argument("--analysis_file", default="")
    return parser


def main() -> None:
    global GLOBAL_SEED
    args = get_parser().parse_args()
    GLOBAL_SEED = args.seed
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.GPU_ID))

    sources = parse_list(args.sources)
    methods = parse_methods(args.methods)
    variants = parse_variants(args.variants)

    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, methods, variants)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources, methods, variants)


if __name__ == "__main__":
    main()
