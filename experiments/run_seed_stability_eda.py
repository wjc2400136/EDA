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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()

import transferattack
from transferattack.utils import AdvDataset, load_pretrained_model, save_images, wrap_model


GLOBAL_SEED = 42
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_EPSILON = "16/255"
DEFAULT_GENERATION_BATCHSIZE = 32

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
    "vit_base_patch16_224": "ViT-B",
    "levit_256": "LeViT",
}

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
    "vit_base_patch16_224": {
        "l2t": 1,
        "bsr": 4,
        "decowa": 8,
        "ops": 16,
        "sid": 16,
        "eda": 16,
    },
    "levit_256": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 16,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int):
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    return _worker_init_fn


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


def parse_seeds(raw_values: str) -> List[int]:
    seeds = [int(item.strip()) for item in raw_values.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


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


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_attack(attack: str) -> str:
    return ATTACK_DISPLAY.get(attack, attack.upper())


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def existing_expected_images(directory: Path, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if (directory / filename).is_file()]


def missing_expected_images(directory: Path, filenames: Iterable[str]) -> List[str]:
    return [filename for filename in filenames if not (directory / filename).is_file()]


def filter_dataset_by_filenames(dataset: AdvDataset, filenames: Iterable[str]) -> Subset:
    wanted = set(filenames)
    ordered_filenames = list(dataset.f2l.keys())
    indices = [idx for idx, filename in enumerate(ordered_filenames) if filename in wanted]
    if len(indices) != len(wanted):
        missing = sorted(wanted.difference(ordered_filenames))
        raise ValueError(f"Some filenames are missing from the dataset labels: {missing[:5]}")
    return Subset(dataset, indices)


def append_generation_progress(progress_csv: Path, row: Dict[str, object]) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "source",
        "epsilon_label",
        "seed",
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
    write_header = not progress_csv.is_file()
    with open(progress_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_case_metadata(case_dir: Path) -> Dict[str, object]:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    eps_label: str,
    eps_value: float,
    seed: int,
    attack: str,
) -> None:
    """Reject outputs generated with a different seed or sampling protocol."""
    metadata = load_case_metadata(case_dir)
    if not metadata:
        raise RuntimeError(
            f"Cannot reuse {case_dir}: case_meta.json is missing. "
            "Regenerate the case without --reuse_existing."
        )
    alpha = args.alpha if args.alpha is not None else eps_value / float(args.epoch)
    expected = {
        "source": source,
        "attack": attack,
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": alpha,
        "epoch": args.epoch,
        "seed": seed,
        **attack_sampling_metadata(args, attack),
    }
    mismatches = [
        f"{key}: existing={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Cannot reuse incompatible outputs in {case_dir}:\n- "
            + "\n- ".join(mismatches)
            + "\nRegenerate the case without --reuse_existing."
        )


def resolve_generation_batchsize(args: argparse.Namespace, source: str, attack: str) -> int:
    if args.batchsize > 0:
        return int(args.batchsize)
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def make_loader(dataset: AdvDataset, batch_size: int, args: argparse.Namespace, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=make_worker_init_fn(seed),
    )


def case_output_dir(
    args: argparse.Namespace,
    source: str,
    eps_label: str,
    seed: int,
    attack: str,
) -> Path:
    return Path(args.output_dir) / source / eps_dir_name(eps_label) / f"seed_{seed}" / attack


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    per_seed_csv = Path(args.per_seed_csv) if args.per_seed_csv else output_dir / "seed_stability_per_seed.csv"
    aggregate_csv = (
        Path(args.aggregate_csv) if args.aggregate_csv else output_dir / "seed_stability_aggregate.csv"
    )
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "seed_stability_table_rows.tex"
    analysis_file = (
        Path(args.analysis_file) if args.analysis_file else output_dir / "seed_stability_analysis.txt"
    )
    return per_seed_csv, aggregate_csv, summary_tex, analysis_file


def build_attacker(
    args: argparse.Namespace,
    source: str,
    eps_value: float,
    attack: str,
    seed: int,
):
    alpha = args.alpha if args.alpha is not None else eps_value / float(args.epoch)
    attack_class = transferattack.load_attack_class(attack)
    kwargs = dict(
        model_name=source,
        epsilon=eps_value,
        alpha=alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
        seed=seed,
    )
    if attack == "bsr":
        kwargs.update(num_scale=args.num_warping)
    elif attack == "decowa":
        kwargs.update(num_warping=args.num_warping)
    elif attack == "sid":
        kwargs.update(num_scale=args.num_warping)
    elif attack == "eda":
        kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
        )
    return attack_class(**kwargs)


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    eps_label: str,
    eps_value: float,
    seed: int,
    attack: str,
    elapsed_seconds: float,
    batch_size: int,
    elapsed_seconds_this_run: float,
    completed_images: int,
    expected_images: int,
    resume_runs: int,
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
        "seed": seed,
        "targeted": args.targeted,
        "random_start": args.random_start,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_total": elapsed_seconds,
        "elapsed_seconds_this_run": elapsed_seconds_this_run,
        "completed_images": completed_images,
        "expected_images": expected_images,
        "complete": completed_images >= expected_images,
        "resume_runs": resume_runs,
    }
    metadata.update(attack_sampling_metadata(args, attack))
    if attack == "eda":
        metadata.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
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
    attacks: List[str],
    seeds: List[int],
    eps_label: str,
    eps_value: float,
) -> None:
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for seed in seeds:
            for attack in attacks:
                set_seed(seed)
                case_dir = case_output_dir(args, source, eps_label, seed, attack)
                existing = existing_expected_images(case_dir, expected)
                missing = missing_expected_images(case_dir, expected)
                if args.reuse_existing and existing:
                    validate_reuse_metadata(
                        case_dir=case_dir,
                        args=args,
                        source=source,
                        eps_label=eps_label,
                        eps_value=eps_value,
                        seed=seed,
                        attack=attack,
                    )
                if args.reuse_existing and not missing:
                    print(f"Skipping existing case: source={source}, seed={seed}, attack={attack}")
                    continue
                if case_dir.exists() and not args.reuse_existing:
                    shutil.rmtree(case_dir)
                    existing = []
                    missing = list(expected)
                case_dir.mkdir(parents=True, exist_ok=True)

                print("\n" + "=" * 80)
                print(
                    f"Generating source={source}, attack={attack}, "
                    f"seed={seed}, epsilon={eps_label}"
                )
                print(f"Output: {case_dir}")

                attacker = build_attacker(args, source, eps_value, attack, seed)
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
                batch_size = resolve_generation_batchsize(args, source, attack)
                print(f"Generation batchsize: {batch_size}")
                loader = make_loader(dataset, batch_size, args, seed)

                previous_metadata = load_case_metadata(case_dir)
                previous_elapsed = metadata_float(
                    previous_metadata,
                    "elapsed_seconds_total",
                    metadata_float(previous_metadata, "elapsed_seconds"),
                )
                previous_resume_runs = int(metadata_float(previous_metadata, "resume_runs"))
                run_id = time.strftime("%Y%m%d_%H%M%S")
                progress_csv = case_dir / "generation_progress.csv"
                completed_images = len(existing)
                start_time = time.perf_counter()
                for images, labels, filenames in tqdm(loader, desc=f"{source}/seed_{seed}/{attack}"):
                    batch_start = time.perf_counter()
                    perturbations = attacker(images, labels)
                    save_images(str(case_dir), images + perturbations.cpu(), filenames)
                    batch_elapsed = time.perf_counter() - batch_start
                    completed_images += len(filenames)
                    append_generation_progress(
                        progress_csv,
                        {
                            "run_id": run_id,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source": source,
                            "epsilon_label": eps_label,
                            "seed": seed,
                            "attack": attack,
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
                elapsed_seconds_this_run = time.perf_counter() - start_time
                elapsed_seconds = previous_elapsed + elapsed_seconds_this_run
                final_completed = len(existing_expected_images(case_dir, expected))

                save_case_metadata(
                    case_dir=case_dir,
                    args=args,
                    source=source,
                    eps_label=eps_label,
                    eps_value=eps_value,
                    seed=seed,
                    attack=attack,
                    elapsed_seconds=elapsed_seconds,
                    batch_size=batch_size,
                    elapsed_seconds_this_run=elapsed_seconds_this_run,
                    completed_images=final_completed,
                    expected_images=len(expected),
                    resume_runs=previous_resume_runs + 1,
                )
                print(
                    f"Generation time this run: {elapsed_seconds_this_run:.2f}s; "
                    f"cumulative: {elapsed_seconds:.2f}s; "
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
    eps_label: str,
    eps_value: float,
    seed: int,
    attack: str,
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, source, eps_label, seed, attack)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing adversarial examples for case: {case_dir}")

    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=str(case_dir),
        targeted=args.targeted,
        eval=True,
    )
    loader = make_loader(dataset, args.eval_batchsize, args, seed)

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(EVAL_CNN_LOAD_LIST, EVAL_VIT_LOAD_LIST):
        print(f"Evaluating source={source}, seed={seed}, attack={attack} on {model_name}")
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, targeted=args.targeted, device=device)
        results[model_name] = asr
        print(f"{source}/seed_{seed}/{attack} -> {model_name}: {asr:.2f}%")

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
        "source_label": display_source(source),
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "attack": attack,
        "attack_label": display_attack(attack),
        "seed": seed,
        "targeted": args.targeted,
        "generation_seconds": load_generation_seconds(case_dir),
        "cnn_avg_incl_source": float(np.mean(cnn_values_all)),
        "vit_avg_incl_source": float(np.mean(vit_values_all)),
        "overall_avg_incl_source": float(np.mean(all_values)),
        "cnn_avg_excl_source": float(np.mean(cnn_values_excl)),
        "vit_avg_excl_source": float(np.mean(vit_values_excl)),
        "target_weighted_transfer_avg_excl_source": float(np.mean(transfer_values_excl)),
        "group_avg_excl_source": float((np.mean(cnn_values_excl) + np.mean(vit_values_excl)) / 2.0),
    }
    row.update(results)
    return row


def per_seed_fieldnames() -> List[str]:
    return [
        "source",
        "source_label",
        "epsilon_label",
        "epsilon",
        "attack",
        "attack_label",
        "seed",
        "targeted",
        "generation_seconds",
    ] + PAPER_EVAL_MODELS + [
        "cnn_avg_incl_source",
        "vit_avg_incl_source",
        "overall_avg_incl_source",
        "cnn_avg_excl_source",
        "vit_avg_excl_source",
        "target_weighted_transfer_avg_excl_source",
        "group_avg_excl_source",
    ]


def aggregate_fieldnames() -> List[str]:
    return [
        "source",
        "source_label",
        "attack",
        "attack_label",
        "epsilon_label",
        "num_seeds",
        "seed_list",
        "generation_seconds_mean",
        "generation_seconds_std",
        "cnn_avg_excl_source_mean",
        "cnn_avg_excl_source_std",
        "vit_avg_excl_source_mean",
        "vit_avg_excl_source_std",
        "target_weighted_transfer_avg_excl_source_mean",
        "target_weighted_transfer_avg_excl_source_std",
        "group_avg_excl_source_mean",
        "group_avg_excl_source_std",
    ]


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
                    formatted[key] = f"{value:.6f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def mean_std(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, 0.0
    return mean, float(np.std(array, ddof=1))


def sort_per_seed_rows(rows: List[Dict[str, object]], sources: List[str], attacks: List[str], seeds: List[int]):
    source_order = {value: index for index, value in enumerate(sources)}
    attack_order = {value: index for index, value in enumerate(attacks)}
    seed_order = {value: index for index, value in enumerate(seeds)}
    return sorted(
        rows,
        key=lambda row: (
            source_order.get(str(row["source"]), 999),
            attack_order.get(str(row["attack"]), 999),
            seed_order.get(int(row["seed"]), 999),
        ),
    )


def aggregate_rows(
    per_seed_rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    eps_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source in sources:
        for attack in attacks:
            subset = [
                row for row in per_seed_rows
                if str(row["source"]) == source and str(row["attack"]) == attack
            ]
            if not subset:
                continue
            seeds = [int(row["seed"]) for row in subset]
            gen_mean, gen_std = mean_std([float(row["generation_seconds"]) for row in subset])
            cnn_mean, cnn_std = mean_std([float(row["cnn_avg_excl_source"]) for row in subset])
            vit_mean, vit_std = mean_std([float(row["vit_avg_excl_source"]) for row in subset])
            transfer_mean, transfer_std = mean_std(
                [float(row["target_weighted_transfer_avg_excl_source"]) for row in subset]
            )
            group_mean, group_std = mean_std([float(row["group_avg_excl_source"]) for row in subset])
            rows.append(
                {
                    "source": source,
                    "source_label": display_source(source),
                    "attack": attack,
                    "attack_label": display_attack(attack),
                    "epsilon_label": eps_label,
                    "num_seeds": len(subset),
                    "seed_list": ",".join(str(seed) for seed in sorted(seeds)),
                    "generation_seconds_mean": gen_mean,
                    "generation_seconds_std": gen_std,
                    "cnn_avg_excl_source_mean": cnn_mean,
                    "cnn_avg_excl_source_std": cnn_std,
                    "vit_avg_excl_source_mean": vit_mean,
                    "vit_avg_excl_source_std": vit_std,
                    "target_weighted_transfer_avg_excl_source_mean": transfer_mean,
                    "target_weighted_transfer_avg_excl_source_std": transfer_std,
                    "group_avg_excl_source_mean": group_mean,
                    "group_avg_excl_source_std": group_std,
                }
            )
    return rows


def format_mean_std(row: Dict[str, object], mean_key: str, std_key: str) -> str:
    return f"{float(row[mean_key]):.2f} $\\pm$ {float(row[std_key]):.2f}"


def write_summary_tex(path: Path, aggregate: List[Dict[str, object]], sources: List[str], attacks: List[str]) -> None:
    lookup = {(row["attack"], row["source"]): row for row in aggregate}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated rows for Table seed_stability.\n")
        f.write("% Columns: Attack, then CNNs and ViTs for each source, mean $\\pm$ std.\n")
        for attack in attacks:
            cells = [display_attack(attack)]
            for source in sources:
                row = lookup.get((attack, source))
                if row is None:
                    cells.extend(["-- $\\pm$ --", "-- $\\pm$ --"])
                else:
                    cnn = format_mean_std(row, "cnn_avg_excl_source_mean", "cnn_avg_excl_source_std")
                    vit = format_mean_std(row, "vit_avg_excl_source_mean", "vit_avg_excl_source_std")
                    cells.extend([cnn, vit])
            f.write(" & ".join(cells) + " \\\\\n")


def write_analysis(path: Path, aggregate: List[Dict[str, object]], sources: List[str], attacks: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("EDA random-seed stability analysis\n")
        f.write("==================================\n")
        f.write("Each seed corresponds to an independent adversarial-example generation run.\n")
        f.write("All attack hyperparameters, sources, targets, and datasets are fixed across seeds.\n")
        f.write("CNN and ViT averages exclude source-identical targets whenever applicable.\n")
        f.write("Standard deviations are sample standard deviations across seeds (ddof=1).\n\n")
        for source in sources:
            f.write(f"Source: {display_source(source)} ({source})\n")
            for attack in attacks:
                row = next(
                    (
                        item for item in aggregate
                        if item["source"] == source and item["attack"] == attack
                    ),
                    None,
                )
                if row is None:
                    continue
                f.write(
                    f"  {display_attack(attack)}: "
                    f"CNN={format_mean_std(row, 'cnn_avg_excl_source_mean', 'cnn_avg_excl_source_std')}, "
                    f"ViT={format_mean_std(row, 'vit_avg_excl_source_mean', 'vit_avg_excl_source_std')}, "
                    f"Transfer={format_mean_std(row, 'target_weighted_transfer_avg_excl_source_mean', 'target_weighted_transfer_avg_excl_source_std')}, "
                    f"generation={format_mean_std(row, 'generation_seconds_mean', 'generation_seconds_std')} s\n"
                )
            f.write("\n")


def evaluate_cases(
    args: argparse.Namespace,
    sources: List[str],
    attacks: List[str],
    seeds: List[int],
    eps_label: str,
    eps_value: float,
) -> None:
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    for source in sources:
        for attack in attacks:
            for seed in seeds:
                set_seed(seed)
                row = evaluate_one_case(args, source, eps_label, eps_value, seed, attack, device)
                rows.append(row)

    rows = sort_per_seed_rows(rows, sources, attacks, seeds)
    aggregate = aggregate_rows(rows, sources, attacks, eps_label)

    per_seed_csv, aggregate_csv, summary_tex, analysis_file = output_paths(args)
    write_csv(per_seed_csv, rows, per_seed_fieldnames())
    write_csv(aggregate_csv, aggregate, aggregate_fieldnames())
    write_summary_tex(summary_tex, aggregate, sources, attacks)
    write_analysis(analysis_file, aggregate, sources, attacks)

    print("\nSaved:")
    print(f"  Per-seed CSV: {per_seed_csv}")
    print(f"  Aggregate CSV: {aggregate_csv}")
    print(f"  LaTeX rows: {summary_tex}")
    print(f"  Analysis: {analysis_file}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate random-seed stability experiments. "
            "Each seed reruns adversarial-example generation with fixed attack settings."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./seed_stability")
    parser.add_argument(
        "--batchsize",
        default=0,
        type=int,
        help="Generation batch size. Use 0 for automatic source/attack-specific values.",
    )
    parser.add_argument("--eval_batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
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

    parser.add_argument("--per_seed_csv", default="")
    parser.add_argument("--aggregate_csv", default="")
    parser.add_argument("--summary_tex", default="")
    parser.add_argument("--analysis_file", default="")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    sources = parse_list(args.sources)
    attacks = parse_attacks(args.attacks)
    seeds = parse_seeds(args.seeds)
    eps_label, eps_value = parse_epsilon_token(args.eps)

    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, attacks, seeds, eps_label, eps_value)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources, attacks, seeds, eps_label, eps_value)


if __name__ == "__main__":
    main()
