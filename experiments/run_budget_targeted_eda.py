import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
DEFAULT_EPSILONS = "4/255,8/255,12/255,16/255"
DEFAULT_GENERATION_BATCHSIZE = 32
SAMPLING_POLICY = "matched_transform_samples_v1"

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


def parse_epsilons(raw_values: str) -> List[Tuple[str, float]]:
    epsilons = [parse_epsilon_token(item) for item in raw_values.split(",") if item.strip()]
    if not epsilons:
        raise ValueError("At least one epsilon value is required.")
    return epsilons


def eps_dir_name(eps_label: str) -> str:
    return "eps_" + eps_label.replace("/", "_").replace(".", "p")


def threat_from_mode(mode: str) -> str:
    if mode.startswith("untargeted"):
        return "untargeted"
    if mode.startswith("targeted"):
        return "targeted"
    raise ValueError(f"Cannot infer threat setting from mode: {mode}")


def mode_runs_generate(mode: str) -> bool:
    return mode.endswith("generate") or mode.endswith("both")


def mode_runs_eval(mode: str) -> bool:
    return mode.endswith("eval") or mode.endswith("both")


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def validate_targeted_labels(input_dir: str) -> None:
    labels_path = Path(input_dir) / "labels.csv"
    labels = pd.read_csv(labels_path)
    required = {"filename", "label", "targeted_label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Missing columns in {labels_path}: {sorted(missing)}")
    same = labels["label"].astype(int) == labels["targeted_label"].astype(int)
    if same.any():
        examples = labels.loc[same, "filename"].head(5).tolist()
        raise ValueError(
            "targeted_label must differ from label for targeted evaluation. "
            f"Examples with identical labels: {examples}"
        )


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
        "threat",
        "source",
        "epsilon_label",
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


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_attack(attack: str) -> str:
    return ATTACK_DISPLAY.get(attack, attack.upper())


def resolve_generation_batchsize(args: argparse.Namespace, source: str, attack: str) -> int:
    if args.batchsize > 0:
        return args.batchsize
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def resolve_eval_batchsize(args: argparse.Namespace) -> int:
    return int(args.batchsize if args.batchsize > 0 else args.eval_batchsize)


def make_loader(dataset: AdvDataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def case_key(threat: str, source: str, eps_label: str, attack: str) -> Tuple[str, str, str, str]:
    return threat, source, eps_label, attack


def case_output_dir(
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    attack: str,
) -> Path:
    return Path(args.output_dir) / threat / source / eps_dir_name(eps_label) / attack


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "budget_targeted_results.csv"
    summary_tex = Path(args.summary_tex) if args.summary_tex else output_dir / "budget_targeted_table_rows.tex"
    full_table_tex = (
        Path(args.full_table_tex)
        if args.full_table_tex
        else output_dir / "budget_targeted_full_table.tex"
    )
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "budget_targeted_results_analysis.txt"
    )
    return result_csv, summary_tex, full_table_tex, analysis_file


def build_attacker(
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    eps_value: float,
    attack: str,
):
    alpha = eps_value / float(args.epoch)
    attack_class = transferattack.load_attack_class(attack)
    kwargs = dict(
        model_name=source,
        epsilon=eps_value,
        alpha=alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=(threat == "targeted"),
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
    )

    # Match the transformed-sample count while retaining each method's other
    # attack-specific settings. OPS(10, 5, 5) and L2T keep their own setup.
    if attack == "l2t":
        kwargs["seed"] = args.seed
    elif attack == "bsr":
        kwargs.update(num_scale=args.num_warping, seed=args.seed)
    elif attack == "decowa":
        kwargs.update(num_warping=args.num_warping, seed=args.seed)
    elif attack == "ops":
        kwargs["seed"] = args.seed
    elif attack == "sid":
        kwargs["num_scale"] = args.num_warping
    elif attack == "eda":
        kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
            seed=args.seed,
        )

    attacker = attack_class(**kwargs)
    validate_attacker_configuration(attacker, args, attack)
    return attacker


def validate_attacker_configuration(attacker, args: argparse.Namespace, attack: str) -> None:
    expected_attributes: Dict[str, object] = {}
    if attack == "l2t":
        expected_attributes = {"num_scale": 2}
    elif attack == "bsr":
        expected_attributes = {"num_scale": args.num_warping}
    elif attack == "decowa":
        expected_attributes = {
            "num_warping": args.num_warping,
            "noise_scale": 2,
        }
    elif attack == "ops":
        expected_attributes = {
            "num_sample_operator": 5,
            "num_sample_neighbor": 5,
        }
    elif attack == "sid":
        expected_attributes = {"num_scale": args.num_warping}
    elif attack == "eda":
        expected_attributes = {
            "num_warping": args.num_warping,
            "noise_scale": args.noise_scale,
            "mesh_width": args.mesh_width,
            "mesh_height": args.mesh_height,
        }

    for name, expected_value in expected_attributes.items():
        actual_value = getattr(attacker, name, None)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Invalid {attack} configuration: {name}={actual_value!r}; "
                f"expected {expected_value!r} under the fairness protocol."
            )


def attack_sampling_metadata(args: argparse.Namespace, attack: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "sampling_policy": SAMPLING_POLICY,
        "reference_transform_samples": args.num_warping,
    }
    if attack == "bsr":
        metadata.update(
            sample_parameter="num_scale",
            effective_transform_samples=args.num_warping,
            transformed_views_per_iteration=args.num_warping,
            num_block=2,
        )
    elif attack == "sid":
        metadata.update(
            sample_parameter="num_scale",
            effective_transform_samples=args.num_warping,
            transformed_views_per_iteration=args.num_warping,
            num_block=2,
            beta=0.1,
            p=0.5,
            omega=0.5,
        )
    elif attack == "decowa":
        metadata.update(
            sample_parameter="num_warping",
            effective_transform_samples=args.num_warping,
            transformed_views_per_iteration=args.num_warping,
            mesh_width=3,
            mesh_height=3,
            noise_scale=2,
            rho=0.01,
        )
    elif attack == "eda":
        metadata.update(
            sample_parameter="num_warping",
            effective_transform_samples=args.num_warping,
            transformed_views_per_iteration=args.num_warping,
        )
    elif attack == "ops":
        metadata.update(
            sample_parameter="num_sample_operator_x_num_sample_neighbor",
            num_sample_operator=5,
            num_sample_neighbor=5,
            sampled_operator_perturbation_pairs=25,
            sampled_views_plus_base_gradient=26,
        )
    elif attack == "l2t":
        metadata.update(
            sample_parameter="original_method_setting",
            effective_transform_samples="original",
            num_scale=2,
        )
    else:
        metadata.update(sample_parameter="original_method_setting")
    return metadata


def reuse_metadata_mismatches(
    metadata: Dict[str, object],
    args: argparse.Namespace,
    attack: str,
) -> List[str]:
    expected = attack_sampling_metadata(args, attack)
    if attack == "eda":
        expected.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
        )

    mismatches = []
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            mismatches.append(
                f"{key}: existing={metadata.get(key)!r}, expected={expected_value!r}"
            )
    return mismatches


def load_generation_config(case_dir: Path) -> Dict[str, object]:
    config_path = case_dir / "generation_config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return load_case_metadata(case_dir)


def metadata_values_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def generation_metadata_mismatches(
    case_dir: Path,
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    eps_value: float,
    attack: str,
    expected_images: int,
) -> List[str]:
    metadata = load_case_metadata(case_dir)
    if not metadata:
        return ["missing case_meta.json"]

    expected_metadata: Dict[str, object] = {
        "threat": threat,
        "source": source,
        "attack": attack,
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": eps_value / float(args.epoch),
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": threat == "targeted",
        "expected_images": expected_images,
        "completed_images": expected_images,
        "complete": True,
    }
    if threat == "targeted":
        expected_metadata["target_label_source"] = "input_dir/labels.csv: targeted_label"

    mismatches = []
    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if not metadata_values_equal(actual_value, expected_value):
            mismatches.append(
                f"{key}: existing={actual_value!r}, expected={expected_value!r}"
            )

    config_path = case_dir / "generation_config.json"
    config = load_generation_config(case_dir)
    if not config_path.is_file():
        mismatches.append("missing generation_config.json")
    else:
        mismatches.extend(reuse_metadata_mismatches(config, args, attack))
    return mismatches


def audit_generation_inventory(
    args: argparse.Namespace,
    threats: List[str],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> Path:
    expected = expected_filenames(args.input_dir)
    expected_count = len(expected)
    audit_path = (
        Path(args.audit_csv)
        if args.audit_csv
        else Path(args.output_dir) / "budget_targeted_generation_audit.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    if "targeted" in threats:
        validate_targeted_labels(args.input_dir)

    for threat in threats:
        for source in sources:
            for eps_label, eps_value in epsilons:
                for attack in attacks:
                    case_dir = case_output_dir(
                        args, threat, source, eps_label, attack
                    )
                    missing = (
                        missing_expected_images(case_dir, expected)
                        if case_dir.is_dir()
                        else list(expected)
                    )
                    found_count = expected_count - len(missing)
                    issues = []
                    if not case_dir.is_dir():
                        issues.append("missing case directory")
                    elif missing:
                        sample = ", ".join(missing[:3])
                        issues.append(
                            f"missing {len(missing)} image(s)"
                            + (f": {sample}" if sample else "")
                        )
                    if case_dir.is_dir() and not missing:
                        issues.extend(
                            generation_metadata_mismatches(
                                case_dir=case_dir,
                                args=args,
                                threat=threat,
                                source=source,
                                eps_label=eps_label,
                                eps_value=eps_value,
                                attack=attack,
                                expected_images=expected_count,
                            )
                        )

                    status = "failed" if issues else "ok"
                    detail = "; ".join(issues)
                    rows.append(
                        {
                            "threat": threat,
                            "source": source,
                            "epsilon_label": eps_label,
                            "attack": attack,
                            "case_dir": str(case_dir),
                            "expected_images": expected_count,
                            "found_images": found_count,
                            "missing_images": len(missing),
                            "status": status,
                            "details": detail,
                        }
                    )
                    if issues:
                        failures.append(
                            f"{threat}/{source}/{eps_label}/{attack}: {detail}"
                        )

    fieldnames = [
        "threat",
        "source",
        "epsilon_label",
        "attack",
        "case_dir",
        "expected_images",
        "found_images",
        "missing_images",
        "status",
        "details",
    ]
    with open(audit_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evaluable_count = len(rows) - len(failures)
    print(f"Generation audit: {evaluable_count}/{len(rows)} cases passed.")
    print(f"Audit CSV: {audit_path}")
    if failures:
        preview = "\n".join(f"- {item}" for item in failures[:20])
        remainder = len(failures) - min(len(failures), 20)
        suffix = f"\n- ... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "Generation inventory is incomplete or incompatible; evaluation "
            f"was not started.\n{preview}{suffix}\nSee {audit_path}"
        )
    return audit_path


def save_generation_config(
    case_dir: Path,
    args: argparse.Namespace,
    attack: str,
) -> None:
    config = attack_sampling_metadata(args, attack)
    if attack == "eda":
        config.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
        )
    with open(case_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def remove_incompatible_case(case_dir: Path, output_dir: str) -> None:
    output_root = Path(output_dir).resolve()
    resolved_case = case_dir.resolve()
    if output_root not in resolved_case.parents:
        raise RuntimeError(
            f"Refusing to remove case outside output directory: {resolved_case}"
        )
    shutil.rmtree(resolved_case)


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    eps_value: float,
    attack: str,
    elapsed_seconds: float,
    batch_size: int,
    elapsed_seconds_this_run: float,
    completed_images: int,
    expected_images: int,
    resume_runs: int,
) -> None:
    metadata = {
        "threat": threat,
        "source": source,
        "attack": attack,
        "attack_display": display_attack(attack),
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": eps_value / float(args.epoch),
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": threat == "targeted",
        "target_label_source": "input_dir/labels.csv: targeted_label",
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
        )
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_cases(
    args: argparse.Namespace,
    threat: str,
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> None:
    if threat == "targeted":
        validate_targeted_labels(args.input_dir)
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for eps_label, eps_value in epsilons:
            for attack in attacks:
                set_seed(args.seed)
                case_dir = case_output_dir(args, threat, source, eps_label, attack)
                existing = existing_expected_images(case_dir, expected)
                missing = missing_expected_images(case_dir, expected)
                if args.reuse_existing and existing:
                    mismatches = reuse_metadata_mismatches(
                        load_generation_config(case_dir), args, attack
                    )
                    if mismatches:
                        details = "; ".join(mismatches)
                        print(
                            f"Regenerating incompatible case {case_dir}: {details}"
                        )
                        remove_incompatible_case(case_dir, args.output_dir)
                        existing = []
                        missing = list(expected)
                if args.reuse_existing and not missing:
                    print(f"Skipping existing case: {threat}/{source}/{eps_label}/{attack}")
                    continue
                if case_dir.exists() and not args.reuse_existing:
                    shutil.rmtree(case_dir)
                    existing = []
                    missing = list(expected)
                case_dir.mkdir(parents=True, exist_ok=True)
                save_generation_config(case_dir, args, attack)

                print("\n" + "=" * 80)
                print(
                    f"Generating threat={threat}, source={source}, "
                    f"epsilon={eps_label}, attack={attack}"
                )
                print(f"Output: {case_dir}")

                attacker = build_attacker(
                    args=args,
                    threat=threat,
                    source=source,
                    eps_label=eps_label,
                    eps_value=eps_value,
                    attack=attack,
                )
                dataset = AdvDataset(
                    input_dir=args.input_dir,
                    output_dir=str(case_dir),
                    targeted=(threat == "targeted"),
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
                loader = make_loader(dataset, batch_size, args)

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
                for images, labels, filenames in tqdm(
                    loader,
                    desc=f"{threat}/{source}/{eps_label}/{attack}",
                ):
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
                            "threat": threat,
                            "source": source,
                            "epsilon_label": eps_label,
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
                    threat=threat,
                    source=source,
                    eps_label=eps_label,
                    eps_value=eps_value,
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


def base_result_row(
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    eps_value: float,
    attack: str,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, threat, source, eps_label, attack)
    return {
        "threat": threat,
        "source": source,
        "source_label": display_source(source),
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": eps_value / float(args.epoch),
        "attack": attack,
        "attack_label": display_attack(attack),
        "targeted": threat == "targeted",
        "generation_seconds": load_generation_seconds(case_dir),
    }


def finalize_result_row(row: Dict[str, object]) -> Dict[str, object]:
    results = {name: float(row[name]) for name in PAPER_EVAL_MODELS}
    source_models = source_model_names(str(row["source"]))
    cnn_values_all = [results[name] for name in PAPER_CNN_MODELS]
    vit_values_all = [results[name] for name in PAPER_VIT_MODELS]
    all_values = [results[name] for name in PAPER_EVAL_MODELS]
    cnn_values_excl = [
        results[name] for name in PAPER_CNN_MODELS if name not in source_models
    ]
    vit_values_excl = [
        results[name] for name in PAPER_VIT_MODELS if name not in source_models
    ]
    transfer_values_excl = [
        results[name] for name in PAPER_EVAL_MODELS if name not in source_models
    ]
    row.update(
        {
            "cnn_avg_incl_source": float(np.mean(cnn_values_all)),
            "vit_avg_incl_source": float(np.mean(vit_values_all)),
            "overall_avg_incl_source": float(np.mean(all_values)),
            "cnn_avg_excl_source": float(np.mean(cnn_values_excl)),
            "vit_avg_excl_source": float(np.mean(vit_values_excl)),
            "target_weighted_transfer_avg_excl_source": float(
                np.mean(transfer_values_excl)
            ),
            "group_avg_excl_source": float(
                (np.mean(cnn_values_excl) + np.mean(vit_values_excl)) / 2.0
            ),
        }
    )
    return row


def evaluate_one_case(
    args: argparse.Namespace,
    threat: str,
    source: str,
    eps_label: str,
    eps_value: float,
    attack: str,
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, threat, source, eps_label, attack)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing adversarial examples for case: {case_dir}")
    if threat == "targeted":
        validate_targeted_labels(args.input_dir)

    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=str(case_dir),
        targeted=(threat == "targeted"),
        eval=True,
    )
    loader = make_loader(dataset, resolve_eval_batchsize(args), args)

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(EVAL_CNN_LOAD_LIST, EVAL_VIT_LOAD_LIST):
        print(
            f"Evaluating threat={threat}, source={source}, epsilon={eps_label}, "
            f"attack={attack} on {model_name}"
        )
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, targeted=(threat == "targeted"), device=device)
        results[model_name] = asr
        print(f"{threat}/{source}/{eps_label}/{attack} -> {model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    row = base_result_row(args, threat, source, eps_label, eps_value, attack)
    row.update(results)
    return finalize_result_row(row)


def result_fieldnames() -> List[str]:
    return [
        "threat",
        "source",
        "source_label",
        "epsilon_label",
        "epsilon",
        "alpha",
        "attack",
        "attack_label",
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
                    formatted[key] = f"{value:.6f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def as_float(row: Dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def sort_rows(
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> List[Dict[str, object]]:
    threat_rank = {"untargeted": 0, "targeted": 1}
    source_rank = {source: idx for idx, source in enumerate(sources)}
    eps_rank = {eps_label: idx for idx, (eps_label, _) in enumerate(epsilons)}
    attack_rank = {attack: idx for idx, attack in enumerate(attacks)}
    return sorted(
        rows,
        key=lambda row: (
            threat_rank.get(str(row.get("threat", "")), 10_000),
            source_rank.get(str(row.get("source", "")), 10_000),
            eps_rank.get(str(row.get("epsilon_label", "")), 10_000),
            attack_rank.get(str(row.get("attack", "")), 10_000),
        ),
    )


def load_existing_rows(result_csv: Path) -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    if not result_csv.is_file():
        return {}
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        case_key(row["threat"], row["source"], row["epsilon_label"], row["attack"]): row
        for row in rows
    }


def ranked_values(
    rows: List[Dict[str, object]], key: str
) -> Tuple[Optional[float], Optional[float]]:
    values = [as_float(row, key) for row in rows]
    unique = sorted({value for value in values if math.isfinite(value)}, reverse=True)
    if len(unique) < 2:
        return None, None
    return unique[0], unique[1]


def format_ranked_value(
    value: float, best: Optional[float], second: Optional[float]
) -> str:
    formatted = f"{value:.1f}"
    if best is not None and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-9):
        return rf"\textbf{{{formatted}}}"
    if second is not None and math.isclose(value, second, rel_tol=0.0, abs_tol=1e-9):
        return rf"\underline{{{formatted}}}"
    return formatted


def table_entries_for_block(
    block_rows: List[Dict[str, object]], attacks: List[str]
) -> List[str]:
    rows_by_attack = {str(row["attack"]): row for row in block_rows}
    cnn_best, cnn_second = ranked_values(block_rows, "cnn_avg_excl_source")
    vit_best, vit_second = ranked_values(block_rows, "vit_avg_excl_source")
    entries = []
    for attack in attacks:
        row = rows_by_attack.get(attack)
        if row is None:
            entries.append("--/--")
            continue
        cnn_value = as_float(row, "cnn_avg_excl_source")
        vit_value = as_float(row, "vit_avg_excl_source")
        entries.append(
            f"{format_ranked_value(cnn_value, cnn_best, cnn_second)}/"
            f"{format_ranked_value(vit_value, vit_best, vit_second)}"
        )
    return entries


def write_analysis(
    analysis_file: Path,
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows, sources, attacks, epsilons)
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Budget and targeted transferability analysis\n")
        f.write("=" * 48 + "\n")
        f.write("CNN averages exclude source-identical targets whenever applicable.\n")
        f.write("Targeted ASR uses targeted_label from input_dir/labels.csv.\n")
        f.write("alpha is epsilon / epoch.\n\n")
        for row in rows:
            f.write(
                f"{row['threat']} | {row['source_label']} | {row['epsilon_label']} | "
                f"{row['attack_label']}: "
                f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                f"Group Avg={as_float(row, 'group_avg_excl_source'):.2f}%, "
                f"Target-weighted transfer={as_float(row, 'target_weighted_transfer_avg_excl_source'):.2f}%, "
                f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
            )


def build_latex_table_rows(
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> List[str]:
    row_map = {
        case_key(row["threat"], row["source"], row["epsilon_label"], row["attack"]): row
        for row in rows
    }
    lines = []
    for source_idx, source in enumerate(sources):
        if source_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{{len(epsilons)}}}{{*}}{{{display_source(source)}}}")
        for eps_label, _ in epsilons:
            untargeted_rows = [
                row_map[key]
                for attack in attacks
                if (key := case_key("untargeted", source, eps_label, attack)) in row_map
            ]
            targeted_rows = [
                row_map[key]
                for attack in attacks
                if (key := case_key("targeted", source, eps_label, attack)) in row_map
            ]
            untargeted_entries = table_entries_for_block(untargeted_rows, attacks)
            targeted_entries = table_entries_for_block(targeted_rows, attacks)
            lines.append(
                f"& ${eps_label}$ & "
                + " & ".join(untargeted_entries)
                + " & "
                + " & ".join(targeted_entries)
                + r" \\"
            )
    return lines


def write_latex_table_rows(
    summary_tex: Path,
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> None:
    summary_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = build_latex_table_rows(rows, sources, attacks, epsilons)
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_budget_targeted_eda.py\n")
        f.write("% Each entry is CNN/ViT source-excluded group averages.\n")
        f.write("\n".join(lines) + "\n")


def write_full_latex_table(
    full_table_tex: Path,
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> None:
    full_table_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = build_latex_table_rows(rows, sources, attacks, epsilons)
    attack_headers = " & ".join(display_attack(attack) for attack in attacks)
    attack_count = len(attacks)
    untargeted_start = 3
    untargeted_end = 2 + attack_count
    targeted_start = untargeted_end + 1
    targeted_end = 2 + 2 * attack_count
    with open(full_table_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_budget_targeted_eda.py\n")
        f.write("\\begin{table*}[!htb]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Transfer ASR (\\%) under different perturbation budgets "
            "and attack objectives. Each entry reports CNN/ViT group averages. "
            "CNN averages exclude the source-identical target whenever applicable. "
            "Bold and underline indicate the best and second-best results within "
            "each source-budget-objective group, respectively.}\n"
        )
        f.write("\\label{tab:budget_targeted_sensitivity}\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.08}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(
            rf"\begin{{tabular}}{{c c *{{{attack_count}}}{{c}} "
            rf"*{{{attack_count}}}{{c}}}}" + "\n"
        )
        f.write("\\toprule\n")
        f.write(
            "\\multirow{2}{*}{Source} & \\multirow{2}{*}{$\\epsilon$} "
            f"& \\multicolumn{{{attack_count}}}{{c}}{{Untargeted ASR (CNNs/ViTs)}} "
            f"& \\multicolumn{{{attack_count}}}{{c}}{{Targeted ASR (CNNs/ViTs)}} \\\\\n"
        )
        f.write(
            f"\\cmidrule(lr){{{untargeted_start}-{untargeted_end}}} "
            f"\\cmidrule(lr){{{targeted_start}-{targeted_end}}}\n"
        )
        f.write(f"& & {attack_headers} & {attack_headers} \\\\\n")
        f.write("\\midrule\n")
        f.write("\n".join(lines) + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")


def has_numeric_result(row: Dict[str, object], key: str) -> bool:
    try:
        return math.isfinite(float(row.get(key, "")))
    except (TypeError, ValueError):
        return False


def load_one_evaluation_model(model_name: str):
    cnn_models = [model_name] if model_name in EVAL_CNN_LOAD_LIST else []
    vit_models = [model_name] if model_name in EVAL_VIT_LOAD_LIST else []
    loaded = list(load_pretrained_model(cnn_models, vit_models))
    if len(loaded) != 1 or loaded[0][0] != model_name:
        raise RuntimeError(f"Failed to load exactly one target model: {model_name}")
    return loaded[0][1]


def evaluate_full_table(
    args: argparse.Namespace,
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
) -> None:
    threats = ["untargeted", "targeted"]
    audit_generation_inventory(args, threats, sources, attacks, epsilons)
    result_csv, summary_tex, full_table_tex, analysis_file = output_paths(args)
    existing = load_existing_rows(result_csv) if args.resume_full_eval else {}

    cases = [
        (threat, source, eps_label, eps_value, attack)
        for threat in threats
        for source in sources
        for eps_label, eps_value in epsilons
        for attack in attacks
    ]
    rows: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for threat, source, eps_label, eps_value, attack in cases:
        key = case_key(threat, source, eps_label, attack)
        row = base_result_row(args, threat, source, eps_label, eps_value, attack)
        if key in existing:
            for model_name in PAPER_EVAL_MODELS:
                if has_numeric_result(existing[key], model_name):
                    row[model_name] = float(existing[key][model_name])
        rows[key] = row

    if not args.resume_full_eval:
        # Clear potentially stale evaluations before the first checkpoint. The
        # adversarial images are always reused; only cached ASR values are reset.
        write_result_csv(
            result_csv, sort_rows(list(rows.values()), sources, attacks, epsilons)
        )
        print("Starting a fresh full evaluation from the existing images.")

    device = torch.device(
        f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu"
    )
    completed_since_checkpoint = 0
    total_evaluations = len(cases) * len(PAPER_EVAL_MODELS)
    completed_evaluations = sum(
        has_numeric_result(row, model_name)
        for row in rows.values()
        for model_name in PAPER_EVAL_MODELS
    )
    print(
        f"Full evaluation inventory: {len(cases)} cases x "
        f"{len(PAPER_EVAL_MODELS)} targets = {total_evaluations} evaluations."
    )
    print(f"Already available in result CSV: {completed_evaluations}")
    print(f"Evaluation device: {device}")

    for model_name in PAPER_EVAL_MODELS:
        pending = [
            case
            for case in cases
            if not has_numeric_result(
                rows[case_key(case[0], case[1], case[2], case[4])], model_name
            )
        ]
        if not pending:
            print(f"Skipping completed target model: {model_name}")
            continue

        print("\n" + "=" * 80)
        print(f"Loading target model once: {model_name} ({len(pending)} pending cases)")
        model = load_one_evaluation_model(model_name)
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        for threat, source, eps_label, eps_value, attack in pending:
            key = case_key(threat, source, eps_label, attack)
            case_dir = case_output_dir(args, threat, source, eps_label, attack)
            print(
                f"Evaluating {model_name}: "
                f"{threat}/{source}/{eps_label}/{attack}"
            )
            dataset = AdvDataset(
                input_dir=args.input_dir,
                output_dir=str(case_dir),
                targeted=(threat == "targeted"),
                eval=True,
            )
            loader = make_loader(dataset, resolve_eval_batchsize(args), args)
            rows[key][model_name] = eval_asr(
                model,
                loader,
                targeted=(threat == "targeted"),
                device=device,
            )
            completed_evaluations += 1
            completed_since_checkpoint += 1
            print(
                f"Progress: {completed_evaluations}/{total_evaluations}; "
                f"ASR={float(rows[key][model_name]):.2f}%"
            )
            if completed_since_checkpoint >= args.eval_checkpoint_every:
                checkpoint_rows = sort_rows(
                    list(rows.values()), sources, attacks, epsilons
                )
                write_result_csv(result_csv, checkpoint_rows)
                completed_since_checkpoint = 0

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoint_rows = sort_rows(list(rows.values()), sources, attacks, epsilons)
        write_result_csv(result_csv, checkpoint_rows)

    incomplete = [
        f"{row['threat']}/{row['source']}/{row['epsilon_label']}/{row['attack']}"
        for row in rows.values()
        if any(not has_numeric_result(row, name) for name in PAPER_EVAL_MODELS)
    ]
    if incomplete:
        raise RuntimeError(
            f"Evaluation stopped with {len(incomplete)} incomplete cases. "
            "Rerun full_eval with --resume_full_eval to resume."
        )

    finalized_rows = [finalize_result_row(row) for row in rows.values()]
    ordered_rows = sort_rows(finalized_rows, sources, attacks, epsilons)
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows, sources, attacks, epsilons)
    write_latex_table_rows(summary_tex, ordered_rows, sources, attacks, epsilons)
    write_full_latex_table(full_table_tex, ordered_rows, sources, attacks, epsilons)
    print("\nFull evaluation completed.")
    print(f"Result CSV: {result_csv}")
    print(f"LaTeX rows: {summary_tex}")
    print(f"Complete LaTeX table: {full_table_tex}")
    print(f"Analysis: {analysis_file}")


def evaluate_cases(
    args: argparse.Namespace,
    threat: str,
    sources: List[str],
    attacks: List[str],
    epsilons: List[Tuple[str, float]],
    report_attacks: Optional[List[str]] = None,
) -> None:
    report_attacks = report_attacks or attacks
    result_csv, summary_tex, full_table_tex, analysis_file = output_paths(args)
    existing = load_existing_rows(result_csv)
    rows: Dict[Tuple[str, str, str, str], Dict[str, object]] = dict(existing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for source in sources:
        for eps_label, eps_value in epsilons:
            for attack in attacks:
                key = case_key(threat, source, eps_label, attack)
                if args.reuse_existing and key in rows:
                    print(f"Skipping existing evaluation: {threat}/{source}/{eps_label}/{attack}")
                    continue
                print("\n" + "=" * 80)
                print(
                    f"Evaluating threat={threat}, source={source}, "
                    f"epsilon={eps_label}, attack={attack}"
                )
                rows[key] = evaluate_one_case(
                    args=args,
                    threat=threat,
                    source=source,
                    eps_label=eps_label,
                    eps_value=eps_value,
                    attack=attack,
                    device=device,
                )
                ordered_rows = sort_rows(
                    list(rows.values()), sources, report_attacks, epsilons
                )
                write_result_csv(result_csv, ordered_rows)
                write_analysis(
                    analysis_file, ordered_rows, sources, report_attacks, epsilons
                )
                write_latex_table_rows(
                    summary_tex, ordered_rows, sources, report_attacks, epsilons
                )
                write_full_latex_table(
                    full_table_tex, ordered_rows, sources, report_attacks, epsilons
                )

    ordered_rows = sort_rows(list(rows.values()), sources, report_attacks, epsilons)
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows, sources, report_attacks, epsilons)
    write_latex_table_rows(
        summary_tex, ordered_rows, sources, report_attacks, epsilons
    )
    write_full_latex_table(
        full_table_tex, ordered_rows, sources, report_attacks, epsilons
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate budget sensitivity plus targeted/untargeted "
            "transferability table entries."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "audit",
            "full_eval",
            "untargeted_generate",
            "untargeted_eval",
            "untargeted_both",
            "targeted_generate",
            "targeted_eval",
            "targeted_both",
        ],
        default="full_eval",
        help=(
            "Use full_eval to evaluate all existing untargeted and targeted "
            "samples for the six paper attacks without regenerating images; "
            "it starts a fresh ASR evaluation unless --resume_full_eval is set. "
            "Use audit to validate the complete generation inventory only."
        ),
    )
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--epsilons", default=DEFAULT_EPSILONS)
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "budget_targeted_sensitivity"),
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
        "--reuse_existing",
        action="store_true",
        help="Generation modes only: reuse compatible images and generate missing ones.",
    )
    parser.add_argument(
        "--resume_full_eval",
        action="store_true",
        help="Resume full_eval from model-level ASR values in the existing result CSV.",
    )
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--decay", default=1.0, type=float)
    parser.add_argument("--norm", default="linfty")
    parser.add_argument("--loss", default="crossentropy")
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
            "their method-specific settings."
        ),
    )

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--summary_tex", default="")
    parser.add_argument("--full_table_tex", default="")
    parser.add_argument("--analysis_file", default="")
    parser.add_argument("--audit_csv", default="")
    parser.add_argument(
        "--eval_checkpoint_every",
        default=8,
        type=int,
        help="Write a resumable evaluation CSV after this many model-case evaluations.",
    )
    return parser


def main() -> None:
    global GLOBAL_SEED
    parser = get_parser()
    args = parser.parse_args()
    GLOBAL_SEED = args.seed
    set_seed(args.seed)

    sources = parse_list(args.sources)
    attacks = parse_attacks(args.attacks)
    epsilons = parse_epsilons(args.epsilons)

    if args.eval_checkpoint_every < 1:
        parser.error("--eval_checkpoint_every must be at least 1.")

    if args.mode in {"audit", "full_eval"}:
        paper_attacks = parse_attacks(DEFAULT_ATTACKS)
        threats = ["untargeted", "targeted"]
        print(
            "Complete-table protocol: evaluating all six paper attacks "
            f"({', '.join(display_attack(item) for item in paper_attacks)})."
        )
        if args.mode == "audit":
            audit_generation_inventory(
                args, threats, sources, paper_attacks, epsilons
            )
        else:
            evaluate_full_table(args, sources, paper_attacks, epsilons)
        return

    threat = threat_from_mode(args.mode)
    if mode_runs_generate(args.mode):
        generate_cases(args, threat, sources, attacks, epsilons)
    if mode_runs_eval(args.mode):
        evaluate_cases(args, threat, sources, attacks, epsilons)


if __name__ == "__main__":
    main()
