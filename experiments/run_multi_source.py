import argparse
import csv
import json
import os
import random
import shutil
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import transferattack
from transferattack.utils import (
    AdvDataset,
    cnn_model_paper,
    generation_target_classes,
    load_pretrained_model,
    save_images,
    vit_model_paper,
    wrap_model,
)


GLOBAL_SEED = 42
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
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
CNN_EXTRA_FROM_TIMM = ["inception_v4", "inception_resnet_v2"]
PAPER_CNN_MODELS = cnn_model_paper + [
    name for name in CNN_EXTRA_FROM_TIMM if name in vit_model_paper
]
PAPER_VIT_MODELS = [
    name for name in vit_model_paper if name not in CNN_EXTRA_FROM_TIMM
]
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


def parse_sources(raw_sources: str) -> List[str]:
    sources = [source.strip() for source in raw_sources.split(",") if source.strip()]
    if not sources:
        raise ValueError("At least one source model must be provided.")
    return sources


def parse_attacks(raw_attacks: str) -> List[str]:
    attacks = [attack.strip().lower() for attack in raw_attacks.split(",") if attack.strip()]
    if not attacks:
        raise ValueError("At least one attack method must be provided.")
    for attack in attacks:
        if attack not in transferattack.attack_zoo:
            raise ValueError(f"Unsupported attack method: {attack}")
    return attacks


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def source_output_dir(args: argparse.Namespace, source: str) -> str:
    if args.run_name:
        return os.path.join(args.output_root, args.run_name, f"{source}-{args.attack}")
    return os.path.join(args.output_root, f"{source}-{args.attack}")


def generation_meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, "generation_meta.json")


def format_seconds(seconds) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}"


def save_generation_meta(
    args: argparse.Namespace,
    source: str,
    output_dir: str,
    elapsed_seconds: float,
    batch_size: int,
    elapsed_seconds_this_run: float = None,
    completed_images: int = None,
    expected_images: int = None,
    resume_runs: int = 1,
) -> None:
    metadata = {
        "source": source,
        "attack": args.attack,
        "output_dir": output_dir,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_total": elapsed_seconds,
        "elapsed_seconds_this_run": elapsed_seconds if elapsed_seconds_this_run is None else elapsed_seconds_this_run,
        "seed": args.seed,
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "batchsize": batch_size,
        "completed_images": completed_images,
        "expected_images": expected_images,
        "complete": (
            completed_images >= expected_images
            if completed_images is not None and expected_images is not None
            else None
        ),
        "resume_runs": resume_runs,
    }
    attack = args.attack.lower()
    metadata.update(attack_sampling_metadata(args, attack))
    if attack == "eda":
        metadata.update(
            {
                "mesh_width": args.mesh_width,
                "mesh_height": args.mesh_height,
                "noise_scale": args.noise_scale,
                "num_warping": args.num_warping,
            }
        )
    with open(generation_meta_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_generation_time(args: argparse.Namespace, source: str):
    metadata_path = generation_meta_path(source_output_dir(args, source))
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return float(metadata["elapsed_seconds"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate adversarial examples for multiple source models."
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--attack", default="eda", choices=transferattack.attack_zoo.keys())
    parser.add_argument(
        "--attacks",
        default="",
        help=(
            "Comma-separated attack list. If provided, this overrides --attack. "
            f"For example: {DEFAULT_ATTACKS}"
        ),
    )
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_root", default="./results/eda")
    parser.add_argument(
        "--run_name",
        default="",
        help="Optional subdirectory under output_root, e.g. eda_s045.",
    )
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete each source output directory before generation.",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help=(
            "Resume generation by keeping existing adversarial images and "
            "generating only missing files listed in input_dir/labels.csv."
        ),
    )

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
    parser.add_argument(
        "--num_warping",
        default=25,
        type=int,
        help=(
            "Matched transformed-sample count N: mapped to num_scale for "
            "BSR/SID and num_warping for DeCoWA/EDA. OPS and L2T retain "
            "their original method-specific settings."
        ),
    )

    parser.add_argument(
        "--result_file",
        default="",
        help="Defaults to output_root/results_eval_<attack>_multi_source.txt.",
    )
    parser.add_argument(
        "--analysis_file",
        default="",
        help="Defaults to output_root/results_eval_<attack>_multi_source_analysis.txt.",
    )
    return parser


def expected_filenames(input_dir: str) -> List[str]:
    labels_path = os.path.join(input_dir, "labels.csv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "filename" not in rows[0]:
        raise ValueError(f"labels.csv must contain a filename column: {labels_path}")
    return [str(row["filename"]) for row in rows]


def existing_expected_images(directory: str, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if os.path.isfile(os.path.join(directory, filename))]


def missing_expected_images(directory: str, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if not os.path.isfile(os.path.join(directory, filename))]


def filter_dataset_by_filenames(dataset: AdvDataset, filenames: Iterable[str]) -> Subset:
    wanted = set(filenames)
    ordered_filenames = list(dataset.f2l.keys())
    indices = [idx for idx, filename in enumerate(ordered_filenames) if filename in wanted]
    if len(indices) != len(wanted):
        missing = sorted(wanted.difference(ordered_filenames))
        raise ValueError(f"Some filenames are missing from labels.csv: {missing[:5]}")
    return Subset(dataset, indices)


def append_generation_progress(progress_csv: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(progress_csv), exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "source",
        "attack",
        "batch_size",
        "batch_images",
        "completed_images",
        "expected_images",
        "batch_elapsed_seconds",
        "run_elapsed_seconds",
        "first_filename",
        "last_filename",
    ]
    write_header = not os.path.isfile(progress_csv)
    with open(progress_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_generation_metadata(output_dir: str) -> Dict[str, object]:
    metadata_path = generation_meta_path(output_dir)
    if not os.path.isfile(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def metadata_float(metadata: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def attack_sampling_metadata(args: argparse.Namespace, attack: str) -> Dict[str, object]:
    """Describe the transformed-sample setting used by each paper attack."""
    if attack == "bsr":
        return {"sample_parameter": "num_scale", "num_scale": args.num_warping}
    if attack == "decowa":
        return {"sample_parameter": "num_warping", "num_warping": args.num_warping}
    if attack == "sid":
        return {"sample_parameter": "num_scale", "num_scale": args.num_warping}
    if attack == "eda":
        return {"sample_parameter": "num_warping", "num_warping": args.num_warping}
    if attack == "ops":
        return {
            "sample_parameter": "original_method_setting",
            "num_sample_operator": 5,
            "num_sample_neighbor": 5,
        }
    if attack == "l2t":
        return {"sample_parameter": "original_method_setting", "num_scale": 2}
    return {"sample_parameter": "original_method_setting"}


def validate_reuse_metadata(
    args: argparse.Namespace, source: str, output_dir: str
) -> None:
    """Reject complete-looking outputs generated under a different protocol."""
    metadata = load_generation_metadata(output_dir)
    if not metadata:
        raise RuntimeError(
            f"Cannot reuse {output_dir}: generation_meta.json is missing. "
            "Regenerate the case without --reuse_existing."
        )
    expected = {
        "source": source,
        "attack": args.attack,
        "seed": args.seed,
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        **attack_sampling_metadata(args, args.attack.lower()),
    }
    mismatches = [
        f"{key}: existing={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Cannot reuse incompatible outputs in {output_dir}:\n- "
            + "\n- ".join(mismatches)
            + "\nRegenerate the case without --reuse_existing."
        )


def build_attacker(args: argparse.Namespace, source: str):
    attack_class = transferattack.load_attack_class(args.attack)
    kwargs = dict(
        model_name=source,
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
    )

    attack = args.attack.lower()
    if attack == "bsr":
        kwargs.update(num_scale=args.num_warping, seed=args.seed)
    elif attack == "decowa":
        kwargs.update(num_warping=args.num_warping, seed=args.seed)
    elif attack == "sid":
        kwargs.update(num_scale=args.num_warping)
    elif attack in {"l2t", "ops"}:
        kwargs.update(seed=args.seed)
    elif attack == "eda":
        kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
            seed=args.seed,
        )

    return attack_class(**kwargs)


def resolve_generation_batchsize(args: argparse.Namespace, source: str) -> int:
    if args.batchsize > 0:
        return args.batchsize
    attack = args.attack.lower()
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def resolve_eval_batchsize(args: argparse.Namespace) -> int:
    return int(args.batchsize if args.batchsize > 0 else args.eval_batchsize)


def make_loader(
    dataset: AdvDataset,
    args: argparse.Namespace,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def generate_for_source(args: argparse.Namespace, source: str) -> Tuple[str, float]:
    set_seed(args.seed)
    output_dir = source_output_dir(args, source)
    if args.clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    expected = expected_filenames(args.input_dir)
    existing = existing_expected_images(output_dir, expected)
    missing = missing_expected_images(output_dir, expected)
    if args.reuse_existing and existing:
        validate_reuse_metadata(args, source, output_dir)
    if args.reuse_existing and args.attack in ["ttp", "m3d"]:
        raise ValueError("--reuse_existing is not supported for ttp/m3d multi-target directory outputs.")
    if args.reuse_existing and not missing:
        elapsed_seconds = load_generation_time(args, source)
        elapsed_seconds = 0.0 if elapsed_seconds is None else elapsed_seconds
        print(
            f"Skipping complete generation: source={source}, attack={args.attack}, "
            f"completed={len(existing)}/{len(expected)}"
        )
        return output_dir, elapsed_seconds

    print(f"\nGenerating adversarial examples: source={source}, attack={args.attack}")
    print(f"Output directory: {output_dir}")
    batch_size = resolve_generation_batchsize(args, source)
    print(f"Generation batchsize: {batch_size}")
    if args.reuse_existing and existing:
        print(
            f"Resume mode: found {len(existing)}/{len(expected)} images; "
            f"generating {len(missing)} missing images."
        )

    start_time = time.perf_counter()
    attacker = build_attacker(args, source)
    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=output_dir,
        targeted=args.targeted,
        eval=False,
    )
    if args.reuse_existing and existing:
        dataset = filter_dataset_by_filenames(dataset, missing)
    loader = make_loader(dataset, args, batch_size)

    previous_metadata = load_generation_metadata(output_dir)
    previous_elapsed = (
        metadata_float(
            previous_metadata,
            "elapsed_seconds_total",
            metadata_float(previous_metadata, "elapsed_seconds"),
        )
        if args.reuse_existing
        else 0.0
    )
    previous_resume_runs = int(metadata_float(previous_metadata, "resume_runs")) if args.reuse_existing else 0
    run_id = time.strftime("%Y%m%d_%H%M%S")
    progress_csv = os.path.join(output_dir, "generation_progress.csv")
    completed_images = len(existing) if args.reuse_existing else 0
    for images, labels, filenames in tqdm(loader):
        batch_start = time.perf_counter()
        if args.attack in ["ttp", "m3d"]:
            for idx, target_class in enumerate(generation_target_classes):
                perturbations = attacker(images, labels, idx)
                target_dir = os.path.join(output_dir, str(target_class))
                os.makedirs(target_dir, exist_ok=True)
                save_images(target_dir, images + perturbations.cpu(), filenames)
        else:
            perturbations = attacker(images, labels)
            save_images(output_dir, images + perturbations.cpu(), filenames)
        batch_elapsed = time.perf_counter() - batch_start
        completed_images += len(filenames)
        append_generation_progress(
            progress_csv,
            {
                "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": source,
                "attack": args.attack,
                "batch_size": batch_size,
                "batch_images": len(filenames),
                "completed_images": completed_images,
                "expected_images": len(expected),
                "batch_elapsed_seconds": f"{batch_elapsed:.6f}",
                "run_elapsed_seconds": f"{time.perf_counter() - start_time:.6f}",
                "first_filename": filenames[0],
                "last_filename": filenames[-1],
            },
        )

    del attacker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elapsed_seconds_this_run = time.perf_counter() - start_time
    elapsed_seconds = previous_elapsed + elapsed_seconds_this_run
    final_completed = len(existing_expected_images(output_dir, expected))
    save_generation_meta(
        args,
        source,
        output_dir,
        elapsed_seconds,
        batch_size,
        elapsed_seconds_this_run=elapsed_seconds_this_run,
        completed_images=final_completed,
        expected_images=len(expected),
        resume_runs=previous_resume_runs + 1,
    )
    print(
        f"Generation time this run for {source}: {elapsed_seconds_this_run:.2f} seconds; "
        f"cumulative: {elapsed_seconds:.2f} seconds; "
        f"completed: {final_completed}/{len(expected)}"
    )
    return output_dir, elapsed_seconds


def eval_asr(model, loader: DataLoader, is_targeted: bool) -> float:
    correct = 0
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
                correct += (predictions == labels_np).sum()
            else:
                correct += (predictions != labels_np).sum()
            total += labels_np.shape[0]
    return 100.0 * correct / total


def evaluate_for_source(
    args: argparse.Namespace,
    source: str,
    model_names: List[str],
) -> Tuple[Dict[str, float], float, float, float, float, float, float]:
    set_seed(args.seed)
    output_dir = source_output_dir(args, source)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Missing adversarial image directory for source={source}: {output_dir}"
        )

    print(f"\nEvaluating adversarial examples: source={source}, attack={args.attack}")
    print(f"Input adversarial directory: {output_dir}")

    eval_dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=output_dir,
        targeted=args.targeted,
        eval=True,
    )
    eval_loader = make_loader(eval_dataset, args, resolve_eval_batchsize(args))

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
        print(f"Evaluating {source} -> {model_name}")
        model = wrap_model(model.eval().cuda())
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        if args.attack in ["ttp", "m3d"]:
            asr = 0.0
            for target_class in generation_target_classes:
                target_dir = os.path.join(output_dir, str(target_class))
                target_dataset = AdvDataset(
                    input_dir=args.input_dir,
                    output_dir=target_dir,
                    targeted=True,
                    target_class=target_class,
                    eval=True,
                )
                target_loader = make_loader(target_dataset, args, resolve_eval_batchsize(args))
                asr += eval_asr(model, target_loader, True)
            asr /= len(generation_target_classes)
        else:
            asr = eval_asr(model, eval_loader, args.targeted)

        results[model_name] = asr
        print(f"{model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(source)
    cnn_values_all = [
        results[name] for name in PAPER_CNN_MODELS if name in results
    ]
    vit_values_all = [
        results[name] for name in PAPER_VIT_MODELS if name in results
    ]
    all_values = [results[name] for name in model_names if name in results]
    cnn_values = [
        results[name]
        for name in PAPER_CNN_MODELS
        if name in results and name not in source_models
    ]
    vit_values = [
        results[name]
        for name in PAPER_VIT_MODELS
        if name in results and name not in source_models
    ]
    transfer_values = [
        results[name]
        for name in model_names
        if name in results and name not in source_models
    ]

    cnn_avg_all = float(np.mean(cnn_values_all)) if cnn_values_all else float("nan")
    vit_avg_all = float(np.mean(vit_values_all)) if vit_values_all else float("nan")
    overall_avg_all = float(np.mean(all_values)) if all_values else float("nan")
    cnn_avg = float(np.mean(cnn_values)) if cnn_values else float("nan")
    vit_avg = float(np.mean(vit_values)) if vit_values else float("nan")
    transfer_avg = float(np.mean(transfer_values)) if transfer_values else float("nan")
    return (
        results,
        cnn_avg_all,
        vit_avg_all,
        overall_avg_all,
        cnn_avg,
        vit_avg,
        transfer_avg,
    )


def write_results(
    args: argparse.Namespace,
    sources: List[str],
    rows: List[Dict],
) -> None:
    result_root = os.path.join(args.output_root, args.run_name) if args.run_name else args.output_root
    os.makedirs(result_root, exist_ok=True)
    if args.result_file:
        if getattr(args, "multi_attack_run", False):
            root, ext = os.path.splitext(args.result_file)
            result_file = f"{root}_{args.attack}{ext or '.txt'}"
        else:
            result_file = args.result_file
    else:
        result_file = os.path.join(result_root, f"results_eval_{args.attack}_multi_source.txt")

    if args.analysis_file:
        if getattr(args, "multi_attack_run", False):
            root, ext = os.path.splitext(args.analysis_file)
            analysis_file = f"{root}_{args.attack}{ext or '.txt'}"
        else:
            analysis_file = args.analysis_file
    else:
        analysis_file = os.path.join(result_root, f"results_eval_{args.attack}_multi_source_analysis.txt")
    model_names = PAPER_EVAL_MODELS

    with open(result_file, "w", encoding="utf-8") as f:
        header = [
            "source",
            "attack",
            "output_dir",
            "generation_time_sec",
            *model_names,
            "CNN Avg incl source",
            "ViT Avg incl source",
            "Overall Avg incl source",
            "CNN Avg excl source",
            "ViT Avg excl source",
            "Transfer Avg excl source",
        ]
        f.write(" | ".join(header) + "\n")
        f.write("-" * 160 + "\n")
        for row in rows:
            values = [
                row["source"],
                args.attack,
                row["output_dir"],
                format_seconds(row.get("generation_time_sec")),
            ]
            values.extend(f"{row['results'][name]:.2f}" for name in model_names)
            values.extend(
                [
                    f"{row['cnn_avg_all']:.2f}",
                    f"{row['vit_avg_all']:.2f}",
                    f"{row['overall_avg_all']:.2f}",
                    f"{row['cnn_avg']:.2f}",
                    f"{row['vit_avg']:.2f}",
                    f"{row['transfer_avg']:.2f}",
                ]
            )
            f.write(" | ".join(values) + "\n")

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Multi-source adversarial generation/evaluation summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Attack: {args.attack}\n")
        f.write(f"Sources: {', '.join(sources)}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Output root: {args.output_root}\n")
        f.write(f"epsilon: {args.eps}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"epoch: {args.epoch}\n")
        f.write(f"seed: {args.seed}\n")
        if args.attack.lower() == "eda":
            f.write(f"mesh_width: {args.mesh_width}\n")
            f.write(f"mesh_height: {args.mesh_height}\n")
            f.write(f"noise_scale: {args.noise_scale}\n")
            f.write(f"num_warping: {args.num_warping}\n")
        f.write("\nCNN group: " + ", ".join(PAPER_CNN_MODELS) + "\n")
        f.write("ViT group: " + ", ".join(PAPER_VIT_MODELS) + "\n\n")
        f.write(
            "Source-excluded averages omit the source model if it appears in the "
            "evaluation target group.\n\n"
        )

        for row in rows:
            f.write(
                f"{row['source']}: "
                f"generation_time={format_seconds(row.get('generation_time_sec'))}s, "
                f"CNN excl={row['cnn_avg']:.2f}%, "
                f"ViT excl={row['vit_avg']:.2f}%, "
                f"Transfer excl={row['transfer_avg']:.2f}%\n"
            )

    print(f"\nResults saved to: {result_file}")
    print(f"Analysis saved to: {analysis_file}")


def write_generation_summary(
    args: argparse.Namespace,
    sources: List[str],
    generation_times: Dict[str, float],
) -> None:
    result_root = os.path.join(args.output_root, args.run_name) if args.run_name else args.output_root
    os.makedirs(result_root, exist_ok=True)
    summary_file = os.path.join(
        result_root, f"generation_time_{args.attack}_multi_source.txt"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("source | attack | output_dir | generation_time_sec\n")
        f.write("-" * 80 + "\n")
        for source in sources:
            f.write(
                f"{source} | {args.attack} | {source_output_dir(args, source)} | "
                f"{format_seconds(generation_times.get(source))}\n"
            )
    print(f"\nGeneration-time summary saved to: {summary_file}")


def main() -> None:
    global GLOBAL_SEED
    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    GLOBAL_SEED = args.seed
    set_seed(args.seed)
    sources = parse_sources(args.sources)
    attacks = parse_attacks(args.attacks) if args.attacks else [args.attack]
    args.multi_attack_run = len(attacks) > 1

    for attack in attacks:
        args.attack = attack
        generation_times = {}
        print("\n" + "#" * 80)
        print(f"Running attack method: {args.attack}")
        print("#" * 80)

        if args.mode in ["generate", "both"]:
            for source in sources:
                _, elapsed_seconds = generate_for_source(args, source)
                generation_times[source] = elapsed_seconds

        if args.mode in ["eval", "both"]:
            rows = []
            for source in sources:
                (
                    results,
                    cnn_avg_all,
                    vit_avg_all,
                    overall_avg_all,
                    cnn_avg,
                    vit_avg,
                    transfer_avg,
                ) = evaluate_for_source(args, source, PAPER_EVAL_MODELS)
                rows.append(
                    {
                        "source": source,
                        "output_dir": source_output_dir(args, source),
                        "generation_time_sec": generation_times.get(
                            source, load_generation_time(args, source)
                        ),
                        "results": results,
                        "cnn_avg_all": cnn_avg_all,
                        "vit_avg_all": vit_avg_all,
                        "overall_avg_all": overall_avg_all,
                        "cnn_avg": cnn_avg,
                        "vit_avg": vit_avg,
                        "transfer_avg": transfer_avg,
                    }
                )
            write_results(args, sources, rows)
        elif args.mode == "generate":
            write_generation_summary(args, sources, generation_times)


if __name__ == "__main__":
    main()
