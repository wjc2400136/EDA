import argparse
import csv
import json
import os
import random
import shutil
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()

import transferattack
from transferattack.utils import AdvDataset, load_pretrained_model, save_images, wrap_model


GLOBAL_SEED = 42
DEFAULT_SOURCE = "resnet18"
DEFAULT_NOISE_STDS = "0.05,0.35,0.65,0.95"
DEFAULT_BRIGHTNESS_RANGES = "0.8:1.2,0.5:1.5,0.2:1.8,0:2"
DEFAULT_NOISE_STD = 0.05
DEFAULT_BRIGHTNESS_RANGE = (0.0, 2.0)
OKABE_ITO_SKY_BLUE = "#56B4E9"
OKABE_ITO_GREEN = "#009E73"
COLORBREWER_EDA_RED = "#CB181D"
COLORBREWER_BLUES = [
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
]

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


def parse_float_list(raw_values: str) -> List[float]:
    values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_brightness_ranges(raw_values: str) -> List[Tuple[float, float]]:
    ranges = []
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Brightness range must be low:high, got {item}")
        low, high = item.split(":", 1)
        low_value = float(low.strip())
        high_value = float(high.strip())
        if low_value > high_value:
            raise ValueError(f"Invalid brightness range: {item}")
        ranges.append((low_value, high_value))
    if not ranges:
        raise ValueError("At least one brightness range is required.")
    return ranges


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "none"
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def format_range_label(value: Optional[Tuple[float, float]]) -> str:
    if value is None:
        return "w/o brightness"
    return f"U({value[0]:g},{value[1]:g})"


def case_dir_name(case: Dict[str, object]) -> str:
    return str(case["case_id"])


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def build_cases(args: argparse.Namespace) -> List[Dict[str, object]]:
    noise_stds = parse_float_list(args.noise_stds)
    brightness_ranges = parse_brightness_ranges(args.brightness_ranges)
    cases: List[Dict[str, object]] = []

    if args.case_set in {"all", "ablation"}:
        cases.extend(
            [
                {
                    "case_id": "a_wo_noise",
                    "case_type": "ablation",
                    "display_name": "w/o Noise",
                    "noise_std": DEFAULT_NOISE_STD,
                    "brightness_low": DEFAULT_BRIGHTNESS_RANGE[0],
                    "brightness_high": DEFAULT_BRIGHTNESS_RANGE[1],
                    "exclude_transform": "noise",
                    "is_default": False,
                },
                {
                    "case_id": "a_wo_brightness",
                    "case_type": "ablation",
                    "display_name": "w/o Brightness",
                    "noise_std": DEFAULT_NOISE_STD,
                    "brightness_low": None,
                    "brightness_high": None,
                    "exclude_transform": "brightness",
                    "is_default": False,
                },
            ]
        )

    if args.case_set in {"all", "grid"}:
        for noise_std in noise_stds:
            for low, high in brightness_ranges:
                is_default = (
                    abs(noise_std - DEFAULT_NOISE_STD) < 1e-12
                    and abs(low - DEFAULT_BRIGHTNESS_RANGE[0]) < 1e-12
                    and abs(high - DEFAULT_BRIGHTNESS_RANGE[1]) < 1e-12
                )
                cases.append(
                    {
                        "case_id": (
                            f"b_sigma{format_float(noise_std)}_"
                            f"bright{format_float(low)}_{format_float(high)}"
                        ),
                        "case_type": "grid",
                        "display_name": (
                            f"sigma={noise_std:g}, r~U({low:g},{high:g})"
                        ),
                        "noise_std": noise_std,
                        "brightness_low": low,
                        "brightness_high": high,
                        "exclude_transform": None,
                        "is_default": is_default,
                    }
                )
    return cases


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, evaluate, and plot EDA appearance-augmentation "
            "ablation and sensitivity experiments."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "plot", "both"], default="both")
    parser.add_argument("--case_set", choices=["all", "ablation", "grid"], default="all")
    parser.add_argument("--attack", default="eda", choices=["eda"])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./appearance_ablation/resnet18")
    parser.add_argument("--batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)
    parser.add_argument("--reuse_existing", action="store_true")

    parser.add_argument("--eps", default=16 / 255, type=float)
    parser.add_argument("--alpha", default=1.6 / 255, type=float)
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
    parser.add_argument("--noise_stds", default=DEFAULT_NOISE_STDS)
    parser.add_argument("--brightness_ranges", default=DEFAULT_BRIGHTNESS_RANGES)

    parser.add_argument(
        "--result_csv",
        default="",
        help="Defaults to output_dir/appearance_results.csv.",
    )
    parser.add_argument(
        "--analysis_file",
        default="",
        help="Defaults to output_dir/appearance_results_analysis.txt.",
    )
    parser.add_argument(
        "--figure_prefix",
        default="",
        help="Defaults to output_dir/figures/appearance_ablation_combined.",
    )
    parser.add_argument(
        "--font_family",
        default="Times New Roman",
        help="Font family used in the generated figure.",
    )
    parser.add_argument(
        "--font_path",
        default="",
        help=(
            "Optional path to a .ttf/.otf Times New Roman font file. "
            "Use this when the runtime cannot find Times New Roman automatically."
        ),
    )
    return parser


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "appearance_results.csv"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "appearance_results_analysis.txt"
    )
    figure_prefix = (
        Path(args.figure_prefix)
        if args.figure_prefix
        else output_dir / "figures" / "appearance_ablation_combined"
    )
    return result_csv, analysis_file, figure_prefix


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def make_loader(dataset: AdvDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def patch_appearance(attacker, case: Dict[str, object]) -> None:
    noise_std = float(case["noise_std"])
    brightness_low = case["brightness_low"]
    brightness_high = case["brightness_high"]

    def channel_noise(self, x):
        if noise_std == 0.0:
            return x
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def brightness(self, x):
        if brightness_low is None or brightness_high is None:
            return x
        factor = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(
            float(brightness_low), float(brightness_high)
        )
        return torch.clamp(x * factor, 0.0, 1.0)

    attacker.channel_noise = types.MethodType(channel_noise, attacker)
    attacker.brightness = types.MethodType(brightness, attacker)


def build_attacker(args: argparse.Namespace, case: Dict[str, object]):
    attack_class = transferattack.load_attack_class(args.attack)
    attacker = attack_class(
        model_name=args.source,
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
        mesh_width=args.mesh_width,
        mesh_height=args.mesh_height,
        noise_scale=args.noise_scale,
        num_warping=args.num_warping,
        seed=args.seed,
        use_appearance=True,
        exclude_transform=case["exclude_transform"],
    )
    patch_appearance(attacker, case)
    return attacker


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    case: Dict[str, object],
) -> None:
    metadata = {
        "source": args.source,
        "attack": args.attack,
        "seed": args.seed,
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "mesh_width": args.mesh_width,
        "mesh_height": args.mesh_height,
        "noise_scale": args.noise_scale,
        "num_warping": args.num_warping,
        "case": case,
    }
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_cases(args: argparse.Namespace, cases: List[Dict[str, object]]) -> None:
    expected = expected_filenames(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        set_seed(args.seed)
        case_dir = output_dir / case_dir_name(case)
        if args.reuse_existing and has_expected_images(case_dir, expected):
            print(f"Skipping existing case: {case['case_id']}")
            continue
        if case_dir.exists() and not args.reuse_existing:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Generating {case['case_id']}: {case['display_name']}")
        print(f"Output: {case_dir}")

        attacker = build_attacker(args, case)
        dataset = AdvDataset(
            input_dir=args.input_dir,
            output_dir=str(case_dir),
            targeted=args.targeted,
            eval=False,
        )
        loader = make_loader(dataset, args)

        for images, labels, filenames in tqdm(loader, desc=str(case["case_id"])):
            perturbations = attacker(images, labels)
            save_images(str(case_dir), images + perturbations.cpu(), filenames)

        save_case_metadata(case_dir, args, case)
        del attacker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def eval_asr(model, loader: DataLoader, is_targeted: bool) -> float:
    success = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            if is_targeted:
                labels = labels[1]
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.numpy()
            if is_targeted:
                success += (predictions == labels_np).sum()
            else:
                success += (predictions != labels_np).sum()
            total += labels_np.shape[0]
    return 100.0 * success / total


def load_existing_rows(result_csv: Path) -> Dict[str, Dict[str, str]]:
    if not result_csv.is_file():
        return {}
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["case_id"]: row for row in reader}


def evaluate_one_case(
    args: argparse.Namespace,
    case: Dict[str, object],
) -> Dict[str, object]:
    case_dir = Path(args.output_dir) / case_dir_name(case)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing adversarial examples for case: {case_dir}")

    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=str(case_dir),
        targeted=args.targeted,
        eval=True,
    )
    loader = make_loader(dataset, args)

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(EVAL_CNN_LOAD_LIST, EVAL_VIT_LOAD_LIST):
        print(f"Evaluating {case['case_id']} on {model_name}")
        model = wrap_model(model.eval().cuda())
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, args.targeted)
        results[model_name] = asr
        print(f"{model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(args.source)
    cnn_values_all = [
        results[name] for name in PAPER_CNN_MODELS if name in results
    ]
    vit_values_all = [
        results[name] for name in PAPER_VIT_MODELS if name in results
    ]
    all_values = [
        results[name] for name in PAPER_EVAL_MODELS if name in results
    ]
    cnn_values_excl = [
        results[name]
        for name in PAPER_CNN_MODELS
        if name in results and name not in source_models
    ]
    vit_values_excl = [
        results[name]
        for name in PAPER_VIT_MODELS
        if name in results and name not in source_models
    ]
    transfer_values_excl = [
        results[name]
        for name in PAPER_EVAL_MODELS
        if name in results and name not in source_models
    ]

    row: Dict[str, object] = {
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "display_name": case["display_name"],
        "noise_std": case["noise_std"],
        "brightness_low": case["brightness_low"],
        "brightness_high": case["brightness_high"],
        "brightness_range": format_range_label(
            None
            if case["brightness_low"] is None
            else (float(case["brightness_low"]), float(case["brightness_high"]))
        ),
        "exclude_transform": case["exclude_transform"] or "",
        "is_default": bool(case["is_default"]),
        "cnn_avg_incl_source": float(np.mean(cnn_values_all)),
        "vit_avg_incl_source": float(np.mean(vit_values_all)),
        "overall_avg_incl_source": float(np.mean(all_values)),
        "cnn_avg_excl_source": float(np.mean(cnn_values_excl)),
        "vit_avg_excl_source": float(np.mean(vit_values_excl)),
        "transfer_avg_excl_source": float(np.mean(transfer_values_excl)),
    }
    row.update(results)
    return row


def result_fieldnames() -> List[str]:
    return [
        "case_id",
        "case_type",
        "display_name",
        "noise_std",
        "brightness_low",
        "brightness_high",
        "brightness_range",
        "exclude_transform",
        "is_default",
    ] + PAPER_EVAL_MODELS + [
        "cnn_avg_incl_source",
        "vit_avg_incl_source",
        "overall_avg_incl_source",
        "cnn_avg_excl_source",
        "vit_avg_excl_source",
        "transfer_avg_excl_source",
    ]


def write_result_csv(result_csv: Path, rows: List[Dict[str, object]]) -> None:
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames()
    with open(result_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    formatted[key] = f"{value:.4f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def evaluate_cases(args: argparse.Namespace, cases: List[Dict[str, object]]) -> None:
    result_csv, analysis_file, _ = output_paths(args)
    existing_rows = load_existing_rows(result_csv) if args.reuse_existing else {}
    rows: Dict[str, Dict[str, object]] = {
        case_id: dict(row) for case_id, row in existing_rows.items()
    }

    for case in cases:
        if args.reuse_existing and case["case_id"] in rows:
            print(f"Skipping existing evaluation: {case['case_id']}")
            continue
        print("\n" + "=" * 80)
        print(f"Evaluating case {case['case_id']}: {case['display_name']}")
        rows[str(case["case_id"])] = evaluate_one_case(args, case)
        ordered_rows = [rows[str(item["case_id"])] for item in cases if str(item["case_id"]) in rows]
        write_result_csv(result_csv, ordered_rows)
        write_analysis(analysis_file, ordered_rows)

    ordered_rows = [rows[str(item["case_id"])] for item in cases if str(item["case_id"]) in rows]
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows)


def as_float(row: Dict[str, object], key: str) -> float:
    value = row[key]
    if value == "" or value is None:
        return float("nan")
    return float(value)


def write_analysis(analysis_file: Path, rows: List[Dict[str, object]]) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    default_rows = [row for row in rows if str(row.get("is_default", "")).lower() in {"true", "1"}]
    grid_rows = [row for row in rows if row.get("case_type") == "grid"]
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("EDA appearance augmentation ablation and sensitivity analysis\n")
        f.write("=" * 68 + "\n")
        if default_rows:
            row = default_rows[0]
            f.write(
                "Default EDA: "
                f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                f"Transfer={as_float(row, 'transfer_avg_excl_source'):.2f}%\n"
            )
        for case_id in ["a_wo_noise", "a_wo_brightness"]:
            matches = [row for row in rows if row.get("case_id") == case_id]
            if matches:
                row = matches[0]
                f.write(
                    f"{row['display_name']}: "
                    f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                    f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                    f"Transfer={as_float(row, 'transfer_avg_excl_source'):.2f}%\n"
                )
        if grid_rows:
            best_cnn = max(grid_rows, key=lambda row: as_float(row, "cnn_avg_excl_source"))
            best_vit = max(grid_rows, key=lambda row: as_float(row, "vit_avg_excl_source"))
            best_transfer = max(grid_rows, key=lambda row: as_float(row, "transfer_avg_excl_source"))
            f.write("\nBest grid settings:\n")
            for name, row, metric in [
                ("CNN", best_cnn, "cnn_avg_excl_source"),
                ("ViT", best_vit, "vit_avg_excl_source"),
                ("Transfer", best_transfer, "transfer_avg_excl_source"),
            ]:
                f.write(
                    f"{name}: {row['display_name']}, "
                    f"{metric}={as_float(row, metric):.2f}%\n"
                )


def load_result_rows(result_csv: Path) -> List[Dict[str, object]]:
    if not result_csv.is_file():
        raise FileNotFoundError(f"Missing result CSV: {result_csv}")
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_default_row(rows: List[Dict[str, object]]) -> Dict[str, object]:
    matches = [
        row for row in rows if str(row.get("is_default", "")).lower() in {"true", "1"}
    ]
    if not matches:
        raise ValueError("Default grid setting sigma=0.05 and U(0,2) was not found.")
    return matches[0]


def heatmap_matrix(
    rows: List[Dict[str, object]],
    noise_stds: List[float],
    brightness_ranges: List[Tuple[float, float]],
    metric: str,
) -> np.ndarray:
    grid_rows = [row for row in rows if row.get("case_type") == "grid"]
    matrix = np.full((len(noise_stds), len(brightness_ranges)), np.nan)
    for row in grid_rows:
        noise_std = float(row["noise_std"])
        low = float(row["brightness_low"])
        high = float(row["brightness_high"])
        for i, expected_noise in enumerate(noise_stds):
            if abs(noise_std - expected_noise) > 1e-12:
                continue
            for j, (expected_low, expected_high) in enumerate(brightness_ranges):
                if abs(low - expected_low) < 1e-12 and abs(high - expected_high) < 1e-12:
                    matrix[i, j] = as_float(row, metric)
    return matrix


def annotate_heatmap(ax, image, data: np.ndarray) -> None:
    facecolors = image.cmap(image.norm(data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                text = "--"
                color = "black"
            else:
                rgba = facecolors[i, j]
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if brightness > 0.5 else "white"
                text = f"{value:.1f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="semibold",
                color=color,
            )


def configure_plot_font(args: argparse.Namespace) -> None:
    font_family = args.font_family
    if args.font_path:
        font_path = Path(args.font_path)
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file does not exist: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=font_family),
                fallback_to_default=False,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Font family '{font_family}' was not found. Install it in the "
                "runtime environment or pass --font_path /path/to/times.ttf."
            ) from exc

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.serif"] = [font_family]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.transparent"] = False


def plot_results(args: argparse.Namespace) -> None:
    result_csv, _, figure_prefix = output_paths(args)
    rows = load_result_rows(result_csv)
    noise_stds = parse_float_list(args.noise_stds)
    brightness_ranges = parse_brightness_ranges(args.brightness_ranges)
    default_row = find_default_row(rows)

    wo_noise = next((row for row in rows if row.get("case_id") == "a_wo_noise"), None)
    wo_brightness = next((row for row in rows if row.get("case_id") == "a_wo_brightness"), None)
    if wo_noise is None or wo_brightness is None:
        raise ValueError("Plotting panel (a) requires a_wo_noise and a_wo_brightness rows.")

    panel_a = {
        "w/o Noise": [
            as_float(wo_noise, "cnn_avg_excl_source"),
            as_float(wo_noise, "vit_avg_excl_source"),
        ],
        "w/o Brightness": [
            as_float(wo_brightness, "cnn_avg_excl_source"),
            as_float(wo_brightness, "vit_avg_excl_source"),
        ],
        "EDA": [
            as_float(default_row, "cnn_avg_excl_source"),
            as_float(default_row, "vit_avg_excl_source"),
        ],
    }
    cnn_matrix = heatmap_matrix(rows, noise_stds, brightness_ranges, "cnn_avg_excl_source")
    vit_matrix = heatmap_matrix(rows, noise_stds, brightness_ranges, "vit_avg_excl_source")

    configure_plot_font(args)
    cmap = LinearSegmentedColormap.from_list(
        "colorbrewer_light_to_dark_blue",
        COLORBREWER_BLUES,
        N=256,
    )

    fig = plt.figure(figsize=(18, 5.4))
    fig.patch.set_facecolor("white")
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.08], wspace=0.12)
    heatmap_grid = outer_grid[0, 1].subgridspec(1, 2, wspace=0.28)
    ax_bar = fig.add_subplot(outer_grid[0, 0])
    ax_cnn = fig.add_subplot(heatmap_grid[0, 0])
    ax_vit = fig.add_subplot(heatmap_grid[0, 1])

    x = np.arange(2)
    width = 0.24
    bar_colors = [OKABE_ITO_SKY_BLUE, OKABE_ITO_GREEN, COLORBREWER_EDA_RED]
    labels = ["CNNs", "ViTs"]
    for idx, (name, values) in enumerate(panel_a.items()):
        positions = x + (idx - 1) * width
        bars = ax_bar.bar(
            positions,
            values,
            width,
            label=name,
            color=bar_colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.12,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color=COLORBREWER_EDA_RED if name == "EDA" else "black",
                fontweight="bold" if name == "EDA" else "normal",
            )
    ax_bar.set_ylabel("Average Attack Success Rate (%)", fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=14)
    ax_bar.set_ylim(75, 100)
    ax_bar.tick_params(axis="y", labelsize=12)
    ax_bar.legend(
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=11,
    )

    x_labels = [format_range_label(item) for item in brightness_ranges]
    y_labels = [f"{item:g}" for item in noise_stds]

    def draw_heatmap(ax, data: np.ndarray, colorbar_label: str):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            vmin, vmax = 0, 1
        else:
            vmin = float(np.floor(np.min(finite)))
            vmax = float(np.ceil(np.max(finite)))
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1.0
        image = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        annotate_heatmap(ax, image, data)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=12)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_xlabel("Brightness range", fontsize=16)
        ax.set_ylabel("Noise std.", fontsize=16)
        default_i = min(
            range(len(noise_stds)),
            key=lambda idx: abs(noise_stds[idx] - DEFAULT_NOISE_STD),
        )
        default_j = min(
            range(len(brightness_ranges)),
            key=lambda idx: (
                abs(brightness_ranges[idx][0] - DEFAULT_BRIGHTNESS_RANGE[0])
                + abs(brightness_ranges[idx][1] - DEFAULT_BRIGHTNESS_RANGE[1])
            ),
        )
        ax.text(
            default_j + 0.28,
            default_i + 0.14,
            "*",
            color=COLORBREWER_EDA_RED,
            fontsize=20,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_ticks(np.arange(vmin, vmax + 0.5, 1.0))
        cbar.ax.text(
            1.18,
            1.015,
            colorbar_label,
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
            clip_on=False,
        )
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cbar.ax.tick_params(labelsize=11)

    draw_heatmap(ax_cnn, cnn_matrix, "CNNs")
    draw_heatmap(ax_vit, vit_matrix, "ViTs")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg", "eps"]:
        kwargs = {"bbox_inches": "tight", "transparent": False, "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(str(figure_prefix.with_suffix(f".{ext}")), **kwargs)
    plt.close(fig)
    print(f"Saved figure prefix: {figure_prefix}")


def main() -> None:
    global GLOBAL_SEED
    parser = get_parser()
    args = parser.parse_args()
    GLOBAL_SEED = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    set_seed(args.seed)

    cases = build_cases(args)
    if not cases:
        raise ValueError("No experiment cases were selected.")

    if args.mode in {"generate", "both"}:
        generate_cases(args, cases)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, cases)
    if args.mode in {"plot", "both"}:
        plot_results(args)


if __name__ == "__main__":
    main()
