import argparse
import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import transferattack
from transferattack.utils import (
    AdvDataset,
    cnn_model_paper,
    load_pretrained_model,
    save_images,
    vit_model_paper,
    wrap_model,
)


GLOBAL_SEED = 42
DEFAULT_SOURCES ="resnet18,inception_v3,inception_v4,inception_resnet_v2"
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
    "resnet18": {"l2t": 2, "bsr": 32, "decowa": 32, "ops": 32, "sid": 32, "eda": 32},
    "inception_v3": {"l2t": 1, "bsr": 8, "decowa": 32, "ops": 32, "sid": 32, "eda": 32},
    "inception_v4": {"l2t": 1, "bsr": 8, "decowa": 32, "ops": 32, "sid": 32, "eda": 32},
    "inception_resnet_v2": {"l2t": 1, "bsr": 8, "decowa": 32, "ops": 32, "sid": 32, "eda": 32},
}

SOURCE_DISPLAY = {
    "resnet18": "RN-18",
    "inception_v3": "Inc-v3",
    "inception_v4": "Inc-v4",
    "inception_resnet_v2": "IncRes-v2",
    "vit_base_patch16_224": "ViT-B",
    "levit_256": "LeViT",
}
ATTACK_DISPLAY = {
    "l2t": "L2T",
    "bsr": "BSR",
    "decowa": "DeCoWA",
    "ops": "OPS",
    "sid": "SID",
    "eda": "EDA",
}
GENERATION_PROTOCOL_VERSION = 2
DECOWA_ORIGINAL_MESH_WIDTH = 3
DECOWA_ORIGINAL_MESH_HEIGHT = 3
DECOWA_ORIGINAL_NOISE_SCALE = 2.0

CNN_EXTRA_FROM_TIMM = ["inception_v4", "inception_resnet_v2"]
PAPER_CNN_MODELS = cnn_model_paper + [name for name in CNN_EXTRA_FROM_TIMM if name in vit_model_paper]
PAPER_VIT_MODELS = [name for name in vit_model_paper if name not in CNN_EXTRA_FROM_TIMM]
EVAL_CNN_LOAD_LIST = cnn_model_paper
EVAL_VIT_LOAD_LIST = CNN_EXTRA_FROM_TIMM + PAPER_VIT_MODELS
PAPER_EVAL_MODELS = PAPER_CNN_MODELS + PAPER_VIT_MODELS


@dataclass(frozen=True)
class DatasetDefaults:
    dataset_name: str
    input_dir: str
    output_dir: str
    result_prefix: str
    expected_images: int


IMAGENET_VAL_DEFAULTS = DatasetDefaults(
    dataset_name="ImageNet-1K validation",
    input_dir="./data_imagenet_val",
    output_dir="./imagenet_val_generalization",
    result_prefix="imagenet_val",
    expected_images=50000,
)


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
        raise ValueError("At least one value must be provided.")
    return values


def parse_sources(raw_sources: str) -> List[str]:
    return parse_list(raw_sources)


def parse_attacks(raw_attacks: str) -> List[str]:
    attacks = [attack.lower() for attack in parse_list(raw_attacks)]
    for attack in attacks:
        if attack not in transferattack.attack_zoo:
            raise ValueError(f"Unsupported attack method: {attack}")
    return attacks


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_attack(attack: str) -> str:
    return ATTACK_DISPLAY.get(attack, attack.upper())


def source_model_names(source: str) -> set:
    return {name.strip() for name in str(source).split(",") if name.strip()}


def read_label_rows(input_dir: str) -> List[Dict[str, str]]:
    labels_path = Path(input_dir) / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "label", "targeted_label"}
    if rows and required.difference(rows[0].keys()):
        raise ValueError(f"labels.csv must contain columns: {sorted(required)}")
    return rows


def expected_filenames(input_dir: str) -> List[str]:
    return [str(row["filename"]) for row in read_label_rows(input_dir)]


def validate_dataset(input_dir: str, expected_count: int) -> None:
    rows = read_label_rows(input_dir)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rows in {Path(input_dir) / 'labels.csv'}, got {len(rows)}."
        )
    image_dir = Path(input_dir) / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    labels = [int(row["label"]) for row in rows]
    if min(labels) < 0 or max(labels) > 999:
        raise ValueError("Labels must be 0-999 for PyTorch/timm ImageNet models.")
    print(f"Dataset check passed: {input_dir}")
    print(f"  rows: {len(rows)}")
    print(f"  label range: {min(labels)}-{max(labels)}")


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
        raise ValueError(f"Some filenames are missing from labels.csv: {missing[:5]}")
    return Subset(dataset, indices)


def make_loader(dataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def resolve_generation_batchsize(args: argparse.Namespace, source: str, attack: str) -> int:
    if args.batchsize > 0:
        return int(args.batchsize)
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def case_output_dir(args: argparse.Namespace, source: str, attack: str) -> Path:
    return Path(args.output_dir) / source / attack


def load_case_metadata(case_dir: Path) -> Dict[str, object]:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_float(metadata: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def attack_protocol_metadata(args: argparse.Namespace, attack: str) -> Dict[str, object]:
    """Record the effective sampling protocol for each paper attack."""
    if attack == "l2t":
        return {"sample_parameter": "original_method_setting", "num_scale": 2}
    if attack == "bsr":
        return {"sample_parameter": "num_scale", "num_scale": args.num_warping}
    if attack == "decowa":
        return {
            "sample_parameter": "num_warping",
            "num_warping": args.num_warping,
            "mesh_width": DECOWA_ORIGINAL_MESH_WIDTH,
            "mesh_height": DECOWA_ORIGINAL_MESH_HEIGHT,
            "noise_scale": DECOWA_ORIGINAL_NOISE_SCALE,
        }
    if attack == "ops":
        return {
            "sample_parameter": "original_method_setting",
            "num_sample_operator": 5,
            "num_sample_neighbor": 5,
        }
    if attack == "sid":
        return {"sample_parameter": "num_scale", "num_scale": args.num_warping}
    if attack == "eda":
        return {
            "sample_parameter": "num_warping",
            "num_warping": args.num_warping,
            "mesh_width": args.mesh_width,
            "mesh_height": args.mesh_height,
            "noise_scale": args.noise_scale,
        }
    return {"sample_parameter": "original_method_setting"}


def metadata_values_match(existing: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(existing) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return existing == expected


def validate_reuse_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    attack: str,
) -> None:
    """Reject outputs generated under an incompatible attack protocol."""
    metadata = load_case_metadata(case_dir)
    if not metadata:
        raise RuntimeError(
            f"Cannot reuse {case_dir}: case_meta.json is missing. "
            "Regenerate this case without --reuse_existing."
        )
    expected = {
        "dataset": args.dataset_name,
        "source": source,
        "attack": attack,
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": False,
    }
    protocol_version = int(metadata_float(metadata, "protocol_version", 0))
    if protocol_version >= GENERATION_PROTOCOL_VERSION:
        expected.update(attack_protocol_metadata(args, attack))
    elif attack in {"bsr", "sid", "decowa"}:
        raise RuntimeError(
            f"Cannot reuse legacy outputs in {case_dir}: the metadata does not "
            f"verify the corrected protocol for {display_attack(attack)}. "
            "Regenerate this case without --reuse_existing."
        )
    elif attack == "eda":
        legacy_eda = attack_protocol_metadata(args, attack)
        legacy_eda.pop("sample_parameter", None)
        expected.update(legacy_eda)

    mismatches = [
        f"{key}: existing={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if not metadata_values_match(metadata.get(key), value)
    ]
    if mismatches:
        raise RuntimeError(
            f"Cannot reuse incompatible outputs in {case_dir}:\n- "
            + "\n- ".join(mismatches)
            + "\nRegenerate this case without --reuse_existing."
        )


def append_progress(progress_csv: Path, row: Dict[str, object]) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "dataset",
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
    write_header = not progress_csv.is_file()
    with progress_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_case_metadata(
    case_dir: Path,
    args: argparse.Namespace,
    source: str,
    attack: str,
    elapsed_seconds_total: float,
    elapsed_seconds_this_run: float,
    batch_size: int,
    completed_images: int,
    expected_images: int,
    resume_runs: int,
) -> None:
    metadata = {
        "protocol_version": GENERATION_PROTOCOL_VERSION,
        "dataset": args.dataset_name,
        "input_dir": args.input_dir,
        "source": source,
        "source_label": display_source(source),
        "attack": attack,
        "attack_label": display_attack(attack),
        "epsilon": args.eps,
        "alpha": args.alpha,
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": False,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds_total,
        "elapsed_seconds_total": elapsed_seconds_total,
        "elapsed_seconds_this_run": elapsed_seconds_this_run,
        "completed_images": completed_images,
        "expected_images": expected_images,
        "complete": completed_images >= expected_images,
        "resume_runs": resume_runs,
    }
    metadata.update(attack_protocol_metadata(args, attack))
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_attacker(args: argparse.Namespace, source: str, attack: str):
    attack_class = transferattack.load_attack_class(attack)
    kwargs = dict(
        model_name=source,
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=False,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
        seed=args.seed,
    )
    if attack == "bsr":
        kwargs.update(num_scale=args.num_warping)
    elif attack == "decowa":
        kwargs.update(
            num_warping=args.num_warping,
            mesh_width=DECOWA_ORIGINAL_MESH_WIDTH,
            mesh_height=DECOWA_ORIGINAL_MESH_HEIGHT,
            noise_scale=DECOWA_ORIGINAL_NOISE_SCALE,
        )
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


def generate_cases(args: argparse.Namespace, sources: List[str], attacks: List[str]) -> None:
    validate_dataset(args.input_dir, args.expected_images)
    expected = expected_filenames(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for source in sources:
        for attack in attacks:
            set_seed(args.seed)
            case_dir = case_output_dir(args, source, attack)
            existing = existing_expected_images(case_dir, expected)
            missing = missing_expected_images(case_dir, expected)
            if args.reuse_existing and existing:
                validate_reuse_metadata(case_dir, args, source, attack)
            if args.reuse_existing and not missing:
                print(f"Skipping complete case: source={source}, attack={attack}")
                continue
            if case_dir.exists() and not args.reuse_existing:
                shutil.rmtree(case_dir)
                existing = []
                missing = list(expected)
            case_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Generating dataset={args.dataset_name}, source={source}, attack={attack}")
            print(f"Output: {case_dir}")
            if args.reuse_existing and existing:
                print(
                    f"Resume mode: found {len(existing)}/{len(expected)} images; "
                    f"generating {len(missing)} missing images."
                )

            attacker = build_attacker(args, source, attack)
            dataset = AdvDataset(input_dir=args.input_dir, output_dir=str(case_dir), targeted=False, eval=False)
            if args.reuse_existing and existing:
                dataset = filter_dataset_by_filenames(dataset, missing)
            batch_size = resolve_generation_batchsize(args, source, attack)
            loader = make_loader(dataset, batch_size, args)
            print(f"Generation batchsize: {batch_size}")

            previous_metadata = load_case_metadata(case_dir)
            previous_elapsed = metadata_float(
                previous_metadata,
                "elapsed_seconds_total",
                metadata_float(previous_metadata, "elapsed_seconds"),
            )
            previous_resume_runs = int(metadata_float(previous_metadata, "resume_runs"))
            progress_csv = case_dir / "generation_progress.csv"
            completed_images = len(existing)
            run_id = time.strftime("%Y%m%d_%H%M%S")
            start_time = time.perf_counter()
            for images, labels, filenames in tqdm(loader, desc=f"{source}/{attack}"):
                batch_start = time.perf_counter()
                perturbations = attacker(images, labels)
                save_images(str(case_dir), images + perturbations.cpu(), filenames)
                batch_elapsed = time.perf_counter() - batch_start
                completed_images += len(filenames)
                append_progress(
                    progress_csv,
                    {
                        "run_id": run_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset": args.dataset_name,
                        "source": source,
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

            elapsed_this_run = time.perf_counter() - start_time
            elapsed_total = previous_elapsed + elapsed_this_run
            final_completed = len(existing_expected_images(case_dir, expected))
            save_case_metadata(
                case_dir=case_dir,
                args=args,
                source=source,
                attack=attack,
                elapsed_seconds_total=elapsed_total,
                elapsed_seconds_this_run=elapsed_this_run,
                batch_size=batch_size,
                completed_images=final_completed,
                expected_images=len(expected),
                resume_runs=previous_resume_runs + 1,
            )
            print(
                f"Generation time this run: {elapsed_this_run:.2f}s; "
                f"cumulative: {elapsed_total:.2f}s; "
                f"completed: {final_completed}/{len(expected)}"
            )
            del attacker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def prediction_cache_path(args: argparse.Namespace, model_name: str) -> Path:
    safe_name = model_name.replace("/", "_")
    return Path(args.output_dir) / "clean_predictions" / f"{safe_name}_clean_predictions.npz"


def load_clean_predictions(path: Path):
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "filenames": data["filenames"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "predictions": data["predictions"].astype(np.int64),
    }


def predict_dataset(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filenames_all: List[str] = []
    labels_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels, filenames in tqdm(loader):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            labels_np = labels.numpy().astype(np.int64)
            filenames_all.extend([str(name) for name in filenames])
            labels_all.append(labels_np)
            preds_all.append(predictions)
    return (
        np.array(filenames_all),
        np.concatenate(labels_all, axis=0),
        np.concatenate(preds_all, axis=0),
    )


def get_clean_predictions(args: argparse.Namespace, model_name: str, model, device: torch.device):
    cache_path = prediction_cache_path(args, model_name)
    cached = load_clean_predictions(cache_path)
    if cached is not None and len(cached["filenames"]) == args.expected_images:
        print(f"Loaded cached clean predictions for {model_name}: {cache_path}")
        return cached

    print(f"Computing clean predictions for {model_name}")
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=str(Path(args.output_dir) / "_clean_dummy"), targeted=False, eval=False)
    loader = make_loader(dataset, args.eval_batchsize, args)
    filenames, labels, predictions = predict_dataset(model, loader, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        filenames=filenames,
        labels=labels,
        predictions=predictions,
    )
    clean_acc = 100.0 * float(np.mean(predictions == labels))
    print(f"Clean accuracy for {model_name}: {clean_acc:.2f}%")
    print(f"Saved clean predictions to: {cache_path}")
    return {"filenames": filenames, "labels": labels, "predictions": predictions}


def load_generation_seconds(case_dir: Path) -> float:
    metadata = load_case_metadata(case_dir)
    return metadata_float(metadata, "elapsed_seconds_total", metadata_float(metadata, "elapsed_seconds", float("nan")))


def metric_prefix(model_name: str, suffix: str) -> str:
    return f"{model_name}_{suffix}"


def evaluate_one_case(
    args: argparse.Namespace,
    source: str,
    attack: str,
    clean_cache_by_model: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, object]:
    case_dir = case_output_dir(args, source, attack)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing adversarial examples: {case_dir}")

    row: Dict[str, object] = {
        "dataset": args.dataset_name,
        "input_dir": args.input_dir,
        "source": source,
        "source_label": display_source(source),
        "attack": attack,
        "attack_label": display_attack(attack),
        "generation_seconds": load_generation_seconds(case_dir),
    }

    raw_results: Dict[str, float] = {}
    clean_acc_results: Dict[str, float] = {}
    clean_correct_results: Dict[str, float] = {}

    for model_name, model in load_pretrained_model(EVAL_CNN_LOAD_LIST, EVAL_VIT_LOAD_LIST):
        print(f"Evaluating dataset={args.dataset_name}, source={source}, attack={attack}, target={model_name}")
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        if model_name not in clean_cache_by_model:
            clean_cache_by_model[model_name] = get_clean_predictions(args, model_name, model, device)
        clean_cache = clean_cache_by_model[model_name]

        adv_dataset = AdvDataset(input_dir=args.input_dir, output_dir=str(case_dir), targeted=False, eval=True)
        adv_loader = make_loader(adv_dataset, args.eval_batchsize, args)
        adv_filenames, labels, adv_predictions = predict_dataset(model, adv_loader, device)

        if not np.array_equal(adv_filenames.astype(str), clean_cache["filenames"].astype(str)):
            raise ValueError(f"Filename order mismatch for target model {model_name}.")
        if not np.array_equal(labels.astype(np.int64), clean_cache["labels"].astype(np.int64)):
            raise ValueError(f"Label order mismatch for target model {model_name}.")

        clean_predictions = clean_cache["predictions"].astype(np.int64)
        labels_np = labels.astype(np.int64)
        clean_correct_mask = clean_predictions == labels_np
        raw_asr = 100.0 * float(np.mean(adv_predictions != labels_np))
        clean_acc = 100.0 * float(np.mean(clean_correct_mask))
        if clean_correct_mask.any():
            clean_correct_asr = 100.0 * float(
                np.mean(adv_predictions[clean_correct_mask] != labels_np[clean_correct_mask])
            )
        else:
            clean_correct_asr = float("nan")

        raw_results[model_name] = raw_asr
        clean_acc_results[model_name] = clean_acc
        clean_correct_results[model_name] = clean_correct_asr
        row[metric_prefix(model_name, "raw_asr")] = raw_asr
        row[metric_prefix(model_name, "clean_acc")] = clean_acc
        row[metric_prefix(model_name, "clean_correct_asr")] = clean_correct_asr
        print(
            f"{source}/{attack} -> {model_name}: "
            f"raw ASR={raw_asr:.2f}%, clean acc={clean_acc:.2f}%, "
            f"clean-correct ASR={clean_correct_asr:.2f}%"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(source)
    for suffix, results in [
        ("raw_asr", raw_results),
        ("clean_correct_asr", clean_correct_results),
        ("clean_acc", clean_acc_results),
    ]:
        cnn_all = [results[name] for name in PAPER_CNN_MODELS if name in results]
        vit_all = [results[name] for name in PAPER_VIT_MODELS if name in results]
        cnn_excl = [results[name] for name in PAPER_CNN_MODELS if name in results and name not in source_models]
        vit_excl = [results[name] for name in PAPER_VIT_MODELS if name in results and name not in source_models]
        transfer_excl = [results[name] for name in PAPER_EVAL_MODELS if name in results and name not in source_models]
        row[f"cnn_avg_{suffix}_incl_source"] = float(np.mean(cnn_all)) if cnn_all else float("nan")
        row[f"vit_avg_{suffix}_incl_source"] = float(np.mean(vit_all)) if vit_all else float("nan")
        row[f"cnn_avg_{suffix}_excl_source"] = float(np.mean(cnn_excl)) if cnn_excl else float("nan")
        row[f"vit_avg_{suffix}_excl_source"] = float(np.mean(vit_excl)) if vit_excl else float("nan")
        row[f"target_weighted_avg_{suffix}_excl_source"] = (
            float(np.mean(transfer_excl)) if transfer_excl else float("nan")
        )
        row[f"group_avg_{suffix}_excl_source"] = (
            float((np.mean(cnn_excl) + np.mean(vit_excl)) / 2.0)
            if cnn_excl and vit_excl
            else float("nan")
        )
    return row


def result_fieldnames() -> List[str]:
    base = [
        "dataset",
        "input_dir",
        "source",
        "source_label",
        "attack",
        "attack_label",
        "generation_seconds",
    ]
    per_model = []
    for model_name in PAPER_EVAL_MODELS:
        per_model.extend(
            [
                metric_prefix(model_name, "raw_asr"),
                metric_prefix(model_name, "clean_acc"),
                metric_prefix(model_name, "clean_correct_asr"),
            ]
        )
    summary = []
    for suffix in ["raw_asr", "clean_correct_asr", "clean_acc"]:
        summary.extend(
            [
                f"cnn_avg_{suffix}_incl_source",
                f"vit_avg_{suffix}_incl_source",
                f"cnn_avg_{suffix}_excl_source",
                f"vit_avg_{suffix}_excl_source",
                f"target_weighted_avg_{suffix}_excl_source",
                f"group_avg_{suffix}_excl_source",
            ]
        )
    return base + per_model + summary


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / f"{args.result_prefix}_generalization_results.csv"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / f"{args.result_prefix}_generalization_analysis.txt"
    )
    summary_tex = (
        Path(args.summary_tex)
        if args.summary_tex
        else output_dir / f"{args.result_prefix}_generalization_table_rows.tex"
    )
    full_table_tex = output_dir / f"{args.result_prefix}_generalization_table.tex"
    return result_csv, analysis_file, summary_tex, full_table_tex


def write_result_csv(result_csv: Path, rows: List[Dict[str, object]]) -> None:
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames()
    with result_csv.open("w", encoding="utf-8", newline="") as f:
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


def load_existing_rows(result_csv: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not result_csv.is_file():
        return {}
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["source"], row["attack"]): row for row in rows}


def sort_rows(rows: List[Dict[str, object]], sources: Sequence[str], attacks: Sequence[str]) -> List[Dict[str, object]]:
    source_rank = {source: idx for idx, source in enumerate(sources)}
    attack_rank = {attack: idx for idx, attack in enumerate(attacks)}
    return sorted(
        rows,
        key=lambda row: (
            source_rank.get(str(row.get("source", "")), 10_000),
            attack_rank.get(str(row.get("attack", "")), 10_000),
        ),
    )


def summary_entry(row: Dict[str, object], metric: str) -> str:
    cnn = as_float(row, f"cnn_avg_{metric}_excl_source")
    vit = as_float(row, f"vit_avg_{metric}_excl_source")
    return f"{cnn:.1f}/{vit:.1f}"


def write_analysis(path: Path, rows: List[Dict[str, object]], sources: Sequence[str], attacks: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{rows[0]['dataset'] if rows else 'Dataset'} generalization analysis\n")
        f.write("=" * 72 + "\n")
        f.write("CNN and ViT averages exclude source-identical targets whenever applicable.\n")
        f.write("raw ASR counts every target-model misclassification after attack.\n")
        f.write("clean-correct ASR is computed only on samples correctly classified before attack.\n\n")
        for source in sources:
            f.write(f"Source: {display_source(source)} ({source})\n")
            for attack in attacks:
                matches = [row for row in rows if row.get("source") == source and row.get("attack") == attack]
                if not matches:
                    continue
                row = matches[0]
                f.write(
                    f"  {display_attack(attack)}: "
                    f"raw CNN/ViT={summary_entry(row, 'raw_asr')}, "
                    f"clean-correct CNN/ViT={summary_entry(row, 'clean_correct_asr')}, "
                    f"clean acc CNN/ViT={summary_entry(row, 'clean_acc')}, "
                    f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
                )
            f.write("\n")


def write_summary_tex(path: Path, rows: List[Dict[str, object]], sources: Sequence[str], attacks: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by dataset generalization script.\n")
        f.write("% Entries are CNN/ViT averages; CC-ASR means clean-correct ASR.\n")
        first_source = True
        for source in sources:
            f.write(r"\midrule" + "\n")
            first_source = False
            for idx, attack in enumerate(attacks):
                matches = [row for row in rows if row.get("source") == source and row.get("attack") == attack]
                if matches:
                    row = matches[0]
                    raw = summary_entry(row, "raw_asr")
                    cc = summary_entry(row, "clean_correct_asr")
                else:
                    raw = "--/--"
                    cc = "--/--"
                source_cell = rf"\multirow{{{len(attacks)}}}{{*}}{{{display_source(source)}}}" if idx == 0 else ""
                f.write(f"{source_cell} & {display_attack(attack)} & {raw} & {cc} \\\\\n")


def write_full_table_tex(path: Path, rows: List[Dict[str, object]], sources: Sequence[str], attacks: Sequence[str]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table*}[!htb]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(
            rf"\caption{{Generalization evaluation on {rows[0]['dataset']}. "
            r"Each entry reports CNN/ViT averages (\%). CC-ASR is computed on clean-correct samples.}}"
            "\n"
        )
        f.write(r"\label{tab:dataset_generalization_" + str(rows[0]["dataset"]).lower().replace(" ", "_").replace("-", "_") + "}\n")
        f.write(r"\resizebox{0.72\textwidth}{!}{" + "\n")
        f.write(r"\begin{tabular}{c c c c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Source & Attack & Raw ASR & CC-ASR \\" + "\n")
        f.write(r"\midrule" + "\n")
        first_source = True
        for source in sources:
            if not first_source:
                f.write(r"\midrule" + "\n")
            first_source = False
            for idx, attack in enumerate(attacks):
                matches = [row for row in rows if row.get("source") == source and row.get("attack") == attack]
                if matches:
                    row = matches[0]
                    raw = summary_entry(row, "raw_asr")
                    cc = summary_entry(row, "clean_correct_asr")
                else:
                    raw = "--/--"
                    cc = "--/--"
                source_cell = rf"\multirow{{{len(attacks)}}}{{*}}{{{display_source(source)}}}" if idx == 0 else ""
                f.write(f"{source_cell} & {display_attack(attack)} & {raw} & {cc} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write("}\n")
        f.write(r"\end{table*}" + "\n")


def evaluate_cases(args: argparse.Namespace, sources: List[str], attacks: List[str]) -> None:
    validate_dataset(args.input_dir, args.expected_images)
    result_csv, analysis_file, summary_tex, full_table_tex = output_paths(args)
    existing_rows = load_existing_rows(result_csv)
    rows = dict(existing_rows)
    clean_cache_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")

    for source in sources:
        for attack in attacks:
            key = (source, attack)
            if args.reuse_existing and key in rows:
                print(f"Skipping existing evaluation: source={source}, attack={attack}")
                continue
            rows[key] = evaluate_one_case(args, source, attack, clean_cache_by_model, device)
            ordered_rows = sort_rows(list(rows.values()), sources, attacks)
            write_result_csv(result_csv, ordered_rows)
            write_analysis(analysis_file, ordered_rows, sources, attacks)
            write_summary_tex(summary_tex, ordered_rows, sources, attacks)
            write_full_table_tex(full_table_tex, ordered_rows, sources, attacks)

    ordered_rows = sort_rows(list(rows.values()), sources, attacks)
    write_result_csv(result_csv, ordered_rows)
    write_analysis(analysis_file, ordered_rows, sources, attacks)
    write_summary_tex(summary_tex, ordered_rows, sources, attacks)
    write_full_table_tex(full_table_tex, ordered_rows, sources, attacks)
    print(f"Saved result CSV to: {result_csv}")
    print(f"Saved analysis to: {analysis_file}")
    print(f"Saved table rows to: {summary_tex}")
    print(f"Saved full table to: {full_table_tex}")


def get_parser(defaults: DatasetDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Generate and evaluate transfer attacks on {defaults.dataset_name}."
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--dataset_name", default=defaults.dataset_name)
    parser.add_argument("--input_dir", default=defaults.input_dir)
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--result_prefix", default=defaults.result_prefix)
    parser.add_argument("--expected_images", type=int, default=defaults.expected_images)
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--batchsize", type=int, default=0, help="Use 0 for source/attack-specific defaults.")
    parser.add_argument("--eval_batchsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--reuse_existing", action="store_true")

    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--alpha", type=float, default=1.6 / 255)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--norm", default="linfty")
    parser.add_argument("--loss", default="crossentropy")
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--mesh_width", type=int, default=3, help="EDA mesh width.")
    parser.add_argument("--mesh_height", type=int, default=3, help="EDA mesh height.")
    parser.add_argument("--noise_scale", type=float, default=0.45, help="EDA deformation scale.")
    parser.add_argument(
        "--num_warping",
        type=int,
        default=25,
        help=(
            "Matched transformed-sample count N: mapped to num_scale for "
            "BSR/SID and num_warping for DeCoWA/EDA. OPS and L2T retain "
            "their original method-specific settings."
        ),
    )

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--analysis_file", default="")
    parser.add_argument("--summary_tex", default="")
    return parser


def main(defaults: DatasetDefaults = IMAGENET_VAL_DEFAULTS) -> None:
    parser = get_parser(defaults)
    args = parser.parse_args()
    sources = parse_sources(args.sources)
    attacks = parse_attacks(args.attacks)
    set_seed(args.seed)
    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, attacks)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources, attacks)


if __name__ == "__main__":
    main()
