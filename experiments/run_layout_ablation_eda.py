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
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"

LAYOUT_CASES = [
    {
        "case_id": "single_wo_edge",
        "display_name": "single (w/o edge)",
        "layout_type": "single",
        "use_dual_grid": False,
        "move_edge": False,
    },
    {
        "case_id": "single_w_edge",
        "display_name": "single (w/ edge)",
        "layout_type": "single",
        "use_dual_grid": False,
        "move_edge": True,
    },
    {
        "case_id": "dual_w_edge",
        "display_name": "dual (w/ edge)",
        "layout_type": "dual",
        "use_dual_grid": True,
        "move_edge": True,
    },
    {
        "case_id": "dual_wo_edge",
        "display_name": "dual (w/o edge)",
        "layout_type": "dual",
        "use_dual_grid": True,
        "move_edge": False,
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


def parse_list(raw_values: str) -> List[str]:
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def make_loader(dataset: AdvDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def source_label(source: str) -> str:
    mapping = {
        "resnet18": "RN-18",
        "inception_v3": "Inc-v3",
        "inception_v4": "Inc-v4",
        "inception_resnet_v2": "IR-v2",
    }
    return mapping.get(source, source)


def case_output_dir(args: argparse.Namespace, source: str, case: Dict[str, object]) -> Path:
    return Path(args.output_dir) / source / str(case["case_id"])


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "layout_results.csv"
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "layout_table_rows.tex"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "layout_results_analysis.txt"
    )
    return result_csv, summary_tex, analysis_file


def build_attacker(args: argparse.Namespace, source: str, case: Dict[str, object]):
    attack_class = transferattack.load_attack_class(args.attack)
    return attack_class(
        model_name=source,
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
        move_edge=bool(case["move_edge"]),
        use_appearance=True,
    )


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    case: Dict[str, object],
    elapsed_seconds: float,
) -> None:
    metadata = {
        "source": source,
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


def generate_cases(args: argparse.Namespace, sources: List[str]) -> None:
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for case in LAYOUT_CASES:
            set_seed(args.seed)
            case_dir = case_output_dir(args, source, case)
            if args.reuse_existing and has_expected_images(case_dir, expected):
                print(f"Skipping existing case: {source}/{case['case_id']}")
                continue
            if case_dir.exists() and not args.reuse_existing:
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Generating source={source}, layout={case['display_name']}")
            print(f"Output: {case_dir}")

            attacker = build_attacker(args, source, case)
            dataset = AdvDataset(
                input_dir=args.input_dir,
                output_dir=str(case_dir),
                targeted=args.targeted,
                eval=False,
            )
            loader = make_loader(dataset, args)

            start_time = time.perf_counter()
            for images, labels, filenames in tqdm(loader, desc=f"{source}/{case['case_id']}"):
                perturbations = attacker(images, labels)
                save_images(str(case_dir), images + perturbations.cpu(), filenames)
            elapsed_seconds = time.perf_counter() - start_time

            save_case_metadata(case_dir, args, source, case, elapsed_seconds)
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
    value = data.get("elapsed_seconds", float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_one_case(
    args: argparse.Namespace,
    source: str,
    case: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, source, case)
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
        print(f"Evaluating source={source}, layout={case['case_id']} on {model_name}")
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, args.targeted, device)
        results[model_name] = asr
        print(f"{source}/{case['case_id']} -> {model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(source)
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

    row: Dict[str, object] = {
        "source": source,
        "source_label": source_label(source),
        "case_id": case["case_id"],
        "display_name": case["display_name"],
        "layout_type": case["layout_type"],
        "use_dual_grid": bool(case["use_dual_grid"]),
        "move_edge": bool(case["move_edge"]),
        "generation_seconds": load_generation_seconds(case_dir),
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
        "source",
        "source_label",
        "case_id",
        "display_name",
        "layout_type",
        "use_dual_grid",
        "move_edge",
        "generation_seconds",
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


def as_float(row: Dict[str, object], key: str) -> float:
    value = row[key]
    if value == "" or value is None:
        return float("nan")
    return float(value)


def sort_rows(rows: List[Dict[str, object]], sources: List[str]) -> List[Dict[str, object]]:
    source_rank = {source: idx for idx, source in enumerate(sources)}
    case_rank = {case["case_id"]: idx for idx, case in enumerate(LAYOUT_CASES)}
    return sorted(
        rows,
        key=lambda row: (
            source_rank.get(str(row["source"]), 10_000),
            case_rank.get(str(row["case_id"]), 10_000),
        ),
    )


def load_existing_rows(result_csv: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not result_csv.is_file():
        return {}
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["source"], row["case_id"]): row for row in rows}


def write_analysis(analysis_file: Path, rows: List[Dict[str, object]], sources: List[str]) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows, sources)
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("EDA control-point layout ablation analysis\n")
        f.write("=" * 48 + "\n")
        f.write("CNN averages exclude source-identical targets whenever applicable.\n\n")
        for row in rows:
            f.write(
                f"{row['source_label']} | {row['display_name']}: "
                f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                f"Transfer={as_float(row, 'transfer_avg_excl_source'):.2f}%, "
                f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
            )


def write_latex_rows(summary_tex: Path, rows: List[Dict[str, object]], sources: List[str]) -> None:
    summary_tex.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows, sources)
    by_case_source = {(row["case_id"], row["source"]): row for row in rows}
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_layout_ablation_eda.py\n")
        f.write("% Entries are CNN/ViT source-excluded averages.\n")
        for case in LAYOUT_CASES:
            values = []
            for source in sources:
                row = by_case_source.get((case["case_id"], source))
                if row is None:
                    values.append("--/--")
                else:
                    values.append(
                        f"{as_float(row, 'cnn_avg_excl_source'):.1f}/"
                        f"{as_float(row, 'vit_avg_excl_source'):.1f}"
                    )
            line = f"{case['display_name']} & " + " & ".join(values) + r" \\"
            f.write(line + "\n")


def evaluate_cases(args: argparse.Namespace, sources: List[str]) -> None:
    result_csv, summary_tex, analysis_file = output_paths(args)
    existing = load_existing_rows(result_csv) if args.reuse_existing else {}
    rows: Dict[Tuple[str, str], Dict[str, object]] = dict(existing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for source in sources:
        for case in LAYOUT_CASES:
            key = (source, str(case["case_id"]))
            if args.reuse_existing and key in rows:
                print(f"Skipping existing evaluation: {source}/{case['case_id']}")
                continue
            print("\n" + "=" * 80)
            print(f"Evaluating source={source}, layout={case['display_name']}")
            rows[key] = evaluate_one_case(args, source, case, device)
            ordered_rows = sort_rows(list(rows.values()), sources)
            write_result_csv(result_csv, ordered_rows)
            write_analysis(analysis_file, ordered_rows, sources)
            write_latex_rows(summary_tex, ordered_rows, sources)

    ordered_rows = sort_rows(list(rows.values()), sources)
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows, sources)
    write_latex_rows(summary_tex, ordered_rows, sources)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate EDA control-point layout ablations over "
            "four source models and four layout modes."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--attack", default="eda", choices=["eda"])
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./layout_ablation")
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

    sources = parse_list(args.sources)
    if args.mode in {"generate", "both"}:
        generate_cases(args, sources)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources)


if __name__ == "__main__":
    main()
