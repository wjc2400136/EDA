import argparse
import csv
import json
import os
import random
import shutil
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

PART_CASES = [
    {
        "case_id": "base",
        "display_name": "Base",
        "use_expansion": False,
        "use_dual_grid": False,
        "use_appearance": False,
    },
    {
        "case_id": "expansion",
        "display_name": "Expansion",
        "use_expansion": True,
        "use_dual_grid": False,
        "use_appearance": False,
    },
    {
        "case_id": "dual_grid",
        "display_name": "Dual grid",
        "use_expansion": False,
        "use_dual_grid": True,
        "use_appearance": False,
    },
    {
        "case_id": "augmentation",
        "display_name": "Augmentation",
        "use_expansion": False,
        "use_dual_grid": False,
        "use_appearance": True,
    },
    {
        "case_id": "expansion_dual_grid",
        "display_name": "Expansion + Dual grid",
        "use_expansion": True,
        "use_dual_grid": True,
        "use_appearance": False,
    },
    {
        "case_id": "expansion_augmentation",
        "display_name": "Expansion + Augmentation",
        "use_expansion": True,
        "use_dual_grid": False,
        "use_appearance": True,
    },
    {
        "case_id": "dual_grid_augmentation",
        "display_name": "Dual grid + Augmentation",
        "use_expansion": False,
        "use_dual_grid": True,
        "use_appearance": True,
    },
    {
        "case_id": "full_eda",
        "display_name": "Full EDA",
        "use_expansion": True,
        "use_dual_grid": True,
        "use_appearance": True,
    },
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


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def source_label(source: str) -> str:
    mapping = {
        "resnet18": "RN-18",
        "inception_v3": "Inc-v3",
        "inception_v4": "Inc-v4",
        "inception_resnet_v2": "IR-v2",
    }
    return mapping.get(source, source)


def make_loader(dataset: AdvDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def case_output_dir(args: argparse.Namespace, case: Dict[str, object]) -> Path:
    return Path(args.output_dir) / args.source / str(case["case_id"])


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir) / args.source
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "parts_results.csv"
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "parts_table_rows.tex"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "parts_results_analysis.txt"
    )
    return result_csv, summary_tex, analysis_file


def patch_no_expansion(attacker) -> None:
    def elastic_warp(self, x):
        batch_size, _, height, width = x.size()
        source_points, mesh_width, mesh_height = self._select_layout()
        source_points = source_points.to(device=x.device, dtype=x.dtype)
        target_points = source_points + self._sample_offsets(
            source_points,
            mesh_width,
            mesh_height,
        )

        source_pixel = self._to_pixel_coords(source_points, height, width)
        target_pixel = self._to_pixel_coords(target_points, height, width)
        source_canvas = self._to_normalized_coords(source_pixel, height, width)
        target_canvas = self._to_normalized_coords(target_pixel, height, width)

        tps = self._get_tps_grid(height, width, x.dtype)
        sampling_grid = tps(source_canvas[None, ...], target_canvas[None, ...])
        sampling_grid = sampling_grid.repeat(batch_size, 1, 1, 1)

        return F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    attacker.elastic_warp = types.MethodType(elastic_warp, attacker)
    attacker.use_canvas_expansion = False


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
        use_dual_grid=bool(case["use_dual_grid"]),
        move_edge=True,
        use_appearance=bool(case["use_appearance"]),
    )
    if not bool(case["use_expansion"]):
        patch_no_expansion(attacker)
    else:
        attacker.use_canvas_expansion = True
    return attacker


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    case: Dict[str, object],
    elapsed_seconds: float,
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
        "elapsed_seconds": elapsed_seconds,
        "case": case,
    }
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_cases(args: argparse.Namespace) -> None:
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for case in PART_CASES:
        set_seed(args.seed)
        case_dir = case_output_dir(args, case)
        if args.reuse_existing and has_expected_images(case_dir, expected):
            print(f"Skipping existing case: {args.source}/{case['case_id']}")
            continue
        if case_dir.exists() and not args.reuse_existing:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Generating source={args.source}, case={case['display_name']}")
        print(f"Output: {case_dir}")

        attacker = build_attacker(args, case)
        dataset = AdvDataset(
            input_dir=args.input_dir,
            output_dir=str(case_dir),
            targeted=args.targeted,
            eval=False,
        )
        loader = make_loader(dataset, args)

        start_time = time.perf_counter()
        for images, labels, filenames in tqdm(loader, desc=str(case["case_id"])):
            perturbations = attacker(images, labels)
            save_images(str(case_dir), images + perturbations.cpu(), filenames)
        elapsed_seconds = time.perf_counter() - start_time

        save_case_metadata(case_dir, args, case, elapsed_seconds)
        print(f"Generation time: {elapsed_seconds:.2f}s")
        del attacker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def eval_asr(model, loader: DataLoader, is_targeted: bool, device: torch.device) -> float:
    success = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            if is_targeted:
                labels = labels[1]
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.numpy()
            if is_targeted:
                success += (predictions == labels_np).sum()
            else:
                success += (predictions != labels_np).sum()
            total += labels_np.shape[0]
    return 100.0 * success / total


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
    case: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, case)
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
        print(f"Evaluating source={args.source}, case={case['case_id']} on {model_name}")
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, args.targeted, device)
        results[model_name] = asr
        print(f"{args.source}/{case['case_id']} -> {model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(args.source)
    cnn_values_all = [results[name] for name in PAPER_CNN_MODELS if name in results]
    vit_values_all = [results[name] for name in PAPER_VIT_MODELS if name in results]
    all_values = [results[name] for name in PAPER_EVAL_MODELS if name in results]
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

    cnn_avg_incl = float(np.mean(cnn_values_all))
    vit_avg_incl = float(np.mean(vit_values_all))
    cnn_avg_excl = float(np.mean(cnn_values_excl))
    vit_avg_excl = float(np.mean(vit_values_excl))

    row: Dict[str, object] = {
        "source": args.source,
        "source_label": source_label(args.source),
        "case_id": case["case_id"],
        "display_name": case["display_name"],
        "use_expansion": bool(case["use_expansion"]),
        "use_dual_grid": bool(case["use_dual_grid"]),
        "use_appearance": bool(case["use_appearance"]),
        "generation_seconds": load_generation_seconds(case_dir),
        "cnn_avg_incl_source": cnn_avg_incl,
        "vit_avg_incl_source": vit_avg_incl,
        "overall_avg_incl_source": float(np.mean(all_values)),
        "group_avg_incl_source": float((cnn_avg_incl + vit_avg_incl) / 2.0),
        "cnn_avg_excl_source": cnn_avg_excl,
        "vit_avg_excl_source": vit_avg_excl,
        "target_weighted_transfer_avg_excl_source": float(np.mean(transfer_values_excl)),
        "group_avg_excl_source": float((cnn_avg_excl + vit_avg_excl) / 2.0),
    }
    row.update(results)
    return row


def result_fieldnames() -> List[str]:
    return [
        "source",
        "source_label",
        "case_id",
        "display_name",
        "use_expansion",
        "use_dual_grid",
        "use_appearance",
        "generation_seconds",
    ] + PAPER_EVAL_MODELS + [
        "cnn_avg_incl_source",
        "vit_avg_incl_source",
        "overall_avg_incl_source",
        "group_avg_incl_source",
        "cnn_avg_excl_source",
        "vit_avg_excl_source",
        "target_weighted_transfer_avg_excl_source",
        "group_avg_excl_source",
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


def as_float(row: Dict[str, object], key: str) -> float:
    value = row[key]
    if value == "" or value is None:
        return float("nan")
    return float(value)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def sort_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    case_rank = {case["case_id"]: idx for idx, case in enumerate(PART_CASES)}
    return sorted(rows, key=lambda row: case_rank.get(str(row["case_id"]), 10_000))


def load_existing_rows(result_csv: Path) -> Dict[str, Dict[str, object]]:
    if not result_csv.is_file():
        return {}
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["case_id"]: row for row in rows}


def checkmark(value: object) -> str:
    return r"\checkmark" if as_bool(value) else ""


def write_analysis(analysis_file: Path, rows: List[Dict[str, object]]) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows)
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("EDA component ablation analysis\n")
        f.write("=" * 32 + "\n")
        f.write("CNN averages exclude source-identical targets whenever applicable.\n")
        f.write("Table Avg. is the simple mean of the reported CNN and ViT group averages.\n\n")
        for row in rows:
            f.write(
                f"{row['display_name']}: "
                f"Expansion={as_bool(row['use_expansion'])}, "
                f"Dual grid={as_bool(row['use_dual_grid'])}, "
                f"Augmentation={as_bool(row['use_appearance'])}, "
                f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                f"Table Avg={as_float(row, 'group_avg_excl_source'):.2f}%, "
                f"Target-weighted transfer={as_float(row, 'target_weighted_transfer_avg_excl_source'):.2f}%, "
                f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
            )


def write_latex_rows(summary_tex: Path, rows: List[Dict[str, object]]) -> None:
    summary_tex.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows)
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_parts_ablation_eda.py\n")
        f.write("% CNN averages exclude source-identical targets when applicable.\n")
        f.write("% Avg. is (CNN Avg. + ViT Avg.) / 2, matching Table~\\ref{tab:ablation_parts}.\n")
        for row in rows:
            line = (
                f"{checkmark(row['use_expansion'])} & "
                f"{checkmark(row['use_dual_grid'])} & "
                f"{checkmark(row['use_appearance'])} & "
                f"{as_float(row, 'cnn_avg_excl_source'):.1f} & "
                f"{as_float(row, 'vit_avg_excl_source'):.1f} & "
                f"{as_float(row, 'group_avg_excl_source'):.1f}"
                r" \\"
            )
            f.write(line + "\n")


def evaluate_cases(args: argparse.Namespace) -> None:
    result_csv, summary_tex, analysis_file = output_paths(args)
    existing = load_existing_rows(result_csv) if args.reuse_existing else {}
    rows: Dict[str, Dict[str, object]] = dict(existing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for case in PART_CASES:
        key = str(case["case_id"])
        if args.reuse_existing and key in rows:
            print(f"Skipping existing evaluation: {args.source}/{case['case_id']}")
            continue
        print("\n" + "=" * 80)
        print(f"Evaluating source={args.source}, case={case['display_name']}")
        rows[key] = evaluate_one_case(args, case, device)
        ordered_rows = sort_rows(list(rows.values()))
        write_result_csv(result_csv, ordered_rows)
        write_analysis(analysis_file, ordered_rows)
        write_latex_rows(summary_tex, ordered_rows)

    ordered_rows = sort_rows(list(rows.values()))
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows)
    write_latex_rows(summary_tex, ordered_rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate the 8-row EDA component ablation table "
            "over Expansion, Dual grid, and Appearance augmentation."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--attack", default="eda", choices=["eda"])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./parts_ablation")
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

    parser.add_argument("--result_csv", default="")
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

    if args.mode in {"generate", "both"}:
        generate_cases(args)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args)


if __name__ == "__main__":
    main()
