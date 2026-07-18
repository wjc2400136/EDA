import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


GLOBAL_SEED = 42
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
DEFAULT_EPSILON = "16/255"
GENERATION_PROTOCOL_VERSION = 2
DECOWA_ORIGINAL_MESH_WIDTH = 3
DECOWA_ORIGINAL_MESH_HEIGHT = 3
DECOWA_ORIGINAL_NOISE_SCALE = 2.0
DEFAULT_ROBUST_MODELS = (
    "Singh2023Revisiting_ConvNeXt-T-ConvStem,"
    "Singh2023Revisiting_ViT-B-ConvStem,"
    "Liu2023Comprehensive_Swin-B,"
    "Chen2024Data_WRN_50_2,"
    "Peng2023Robust,"
    "Xu2024MIMIR_Swin-B,"
    "Liu2023Comprehensive_ConvNeXt-L,"
    "Amini2024MeanSparse_ConvNeXt-L"
)
DEFAULT_GENERATION_BATCHSIZE = 32

ATTACK_DEFAULT_BATCHSIZE = {
    "l2t": 1,
    "bsr": 8,
    "decowa": 32,
    "decowa16": 32,
    "ops": 32,
    "sid": 32,
    "eda": 32,
}
SOURCE_ATTACK_BATCHSIZE = {
    "resnet18": {
        "l2t": 2,
        "bsr": 32,
        "decowa": 32,
        "decowa16": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "densenet121": {
        "l2t": 1,
        "bsr": 4,
        "decowa": 4,
        "decowa16": 4,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "vgg19": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 8,
        "decowa16": 8,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_v3": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "decowa16": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_v4": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "decowa16": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
    "inception_resnet_v2": {
        "l2t": 1,
        "bsr": 8,
        "decowa": 32,
        "decowa16": 32,
        "ops": 32,
        "sid": 32,
        "eda": 32,
    },
}

ATTACK_DISPLAY = {
    "l2t": "L2T",
    "bsr": "BSR",
    "decowa": "DeCoWA",
    "decowa16": "DeCoWA",
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
ROBUST_MODEL_DISPLAY = {
    "Singh2023Revisiting_ConvNeXt-T-ConvStem": "ConvNeXt-T",
    "Singh2023Revisiting_ViT-B-ConvStem": "ViT-B",
    "Liu2023Comprehensive_Swin-B": "Swin-B",
    "Chen2024Data_WRN_50_2": "WRN-50-2",
    "Peng2023Robust": "RobustArch",
    "Xu2024MIMIR_Swin-B": "MIMIR-Swin-B",
    "Liu2023Comprehensive_ConvNeXt-L": "ConvNeXt-L",
    "Amini2024MeanSparse_ConvNeXt-L": "MeanSparse-CNX-L",
    "Bai2024MixedNUTS": "MixedNUTS",
}

IMAGE_SIZE = 224
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir: str, targeted: bool = False, target_class: Optional[int] = None) -> None:
        self.input_dir = Path(input_dir)
        self.image_dir = self.input_dir / "images"
        self.targeted = targeted
        self.target_class = target_class
        self.labels = self.load_labels(self.input_dir / "labels.csv")
        self.filenames = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = Image.open(self.image_dir / filename).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        image = np.asarray(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, self.labels[filename], filename

    def load_labels(self, label_file: Path) -> Dict[str, object]:
        dev = pd.read_csv(label_file)
        if self.targeted:
            if self.target_class is not None:
                return {
                    dev.iloc[i]["filename"]: [int(dev.iloc[i]["label"]), int(self.target_class)]
                    for i in range(len(dev))
                }
            return {
                dev.iloc[i]["filename"]: [int(dev.iloc[i]["label"]), int(dev.iloc[i]["targeted_label"])]
                for i in range(len(dev))
            }
        return {dev.iloc[i]["filename"]: int(dev.iloc[i]["label"]) for i in range(len(dev))}


class RobustBenchEvalDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir: str, adv_dir: Path, targeted: bool = False) -> None:
        self.input_dir = Path(input_dir)
        self.adv_dir = Path(adv_dir)
        self.targeted = targeted
        self.labels = self.load_labels(self.input_dir / "labels.csv")
        self.filenames = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = Image.open(self.adv_dir / filename).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        image = np.asarray(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.labels[filename]
        return image, label, filename

    def load_labels(self, label_file: Path) -> Dict[str, int]:
        dev = pd.read_csv(label_file)
        column = "targeted_label" if self.targeted else "label"
        return {dev.iloc[i]["filename"]: int(dev.iloc[i][column]) for i in range(len(dev))}


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


def display_robust_model(model_name: str) -> str:
    return ROBUST_MODEL_DISPLAY.get(model_name, model_name)


def safe_column_name(model_name: str) -> str:
    return "robust_" + "".join(char if char.isalnum() else "_" for char in model_name).strip("_")


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def list_image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    expected = set(filenames)
    present = {path.name for path in list_image_files(directory)}
    return expected.issubset(present)


def save_images(output_dir: Path, adversaries: torch.Tensor, filenames: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = (adversaries.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for idx, filename in enumerate(filenames):
        Image.fromarray(images[idx]).save(output_dir / filename)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.GPU_ID == "":
        return torch.device("cuda")
    torch.cuda.set_device(int(args.GPU_ID))
    return torch.device(f"cuda:{int(args.GPU_ID)}")


def resolve_generation_batchsize(args: argparse.Namespace, source: str, attack: str) -> int:
    if args.batchsize > 0:
        return args.batchsize
    source_rules = SOURCE_ATTACK_BATCHSIZE.get(source, {})
    if attack in source_rules:
        return int(source_rules[attack])
    return int(ATTACK_DEFAULT_BATCHSIZE.get(attack, DEFAULT_GENERATION_BATCHSIZE))


def make_loader(dataset: torch.utils.data.Dataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def case_output_dir(args: argparse.Namespace, source: str, attack: str) -> Path:
    eps_label, _ = parse_epsilon_token(args.eps)
    eps_dir = eps_dir_name(eps_label)
    if args.adv_dir_template:
        return Path(
            args.adv_dir_template.format(
                source=source,
                source_label=display_source(source),
                attack=attack,
                attack_label=display_attack(attack),
                eps_label=eps_label,
                eps_dir=eps_dir,
                output_dir=args.output_dir,
            )
        ).resolve()
    return (Path(args.output_dir) / "adversarial_examples" / source / eps_dir / attack).resolve()


def case_meta_path(case_dir: Path) -> Path:
    return case_dir / "case_meta.json"


def load_generation_seconds(case_dir: Path) -> float:
    meta_path = case_meta_path(case_dir)
    if not meta_path.is_file():
        return float("nan")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("elapsed_seconds", float("nan")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return float("nan")


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "modern_robustbench_results.csv"
    tex_rows = Path(args.tex_rows) if args.tex_rows else output_dir / "modern_robustbench_table_rows.tex"
    full_tex = Path(args.full_tex) if args.full_tex else output_dir / "modern_robustbench_table.tex"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "modern_robustbench_analysis.txt"
    )
    return result_csv, tex_rows, full_tex, analysis_file


def build_attacker(args: argparse.Namespace, source: str, attack: str, eps_value: float):
    import timm

    if not hasattr(timm.models, "hub"):
        class _TimmHubCompat:
            HUB_SERVER = None

        timm.models.hub = _TimmHubCompat()

    import transferattack

    if attack not in transferattack.attack_zoo:
        raise ValueError(f"Unsupported attack method: {attack}")
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
        common_kwargs.update(
            num_warping=args.num_warping,
            mesh_width=DECOWA_ORIGINAL_MESH_WIDTH,
            mesh_height=DECOWA_ORIGINAL_MESH_HEIGHT,
            noise_scale=DECOWA_ORIGINAL_NOISE_SCALE,
        )
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


def attack_protocol_metadata(args: argparse.Namespace, attack: str) -> Dict[str, object]:
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


def load_case_metadata(case_dir: Path) -> Dict[str, object]:
    meta_path = case_meta_path(case_dir)
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    eps_label: str,
    eps_value: float,
) -> None:
    metadata = load_case_metadata(case_dir)
    if not metadata:
        raise RuntimeError(
            f"Cannot reuse {case_dir}: case_meta.json is missing. "
            "Regenerate this case without --reuse_existing."
        )
    alpha = args.alpha if args.alpha is not None else eps_value / float(args.epoch)
    expected = {
        "source": source,
        "attack": attack,
        "epsilon_label": eps_label,
        "epsilon": eps_value,
        "alpha": alpha,
        "epoch": args.epoch,
        "seed": args.seed,
        "targeted": args.targeted,
    }
    protocol_version = int(metadata.get("protocol_version", 0) or 0)
    if protocol_version >= GENERATION_PROTOCOL_VERSION:
        expected.update(attack_protocol_metadata(args, attack))
    elif attack in {"bsr", "decowa", "sid"}:
        raise RuntimeError(
            f"Cannot reuse legacy {display_attack(attack)} outputs in {case_dir}: "
            "their transformed-sample count is not verified. Regenerate this "
            "case without --reuse_existing."
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
        "protocol_version": GENERATION_PROTOCOL_VERSION,
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
    }
    metadata.update(attack_protocol_metadata(args, attack))
    with open(case_meta_path(case_dir), "w", encoding="utf-8") as f:
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
                validate_reuse_metadata(
                    case_dir, args, source, attack, eps_label, eps_value
                )
                print(f"Skipping existing adversarial examples: source={source}, attack={attack}")
                continue
            if case_dir.exists() and not args.reuse_existing:
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Generating source={source}, attack={attack}, epsilon={eps_label}")
            print(f"Output: {case_dir}")

            attacker = build_attacker(args, source, attack, eps_value)
            dataset = GenerationDataset(
                input_dir=args.input_dir,
                targeted=args.targeted,
                target_class=args.target_class,
            )
            batch_size = resolve_generation_batchsize(args, source, attack)
            print(f"Generation batchsize: {batch_size}")
            loader = make_loader(dataset, batch_size, args)

            start_time = time.perf_counter()
            for images, labels, filenames in tqdm(loader, desc=f"generate/{source}/{attack}"):
                perturbations = attacker(images, labels)
                save_images(case_dir, images + perturbations.cpu(), filenames)
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


def robustbench_checkpoint_path(args: argparse.Namespace, model_name: str) -> Path:
    return Path(args.model_dir) / args.dataset / args.threat_model / f"{model_name}.pt"


def load_robustbench_model(args: argparse.Namespace, model_name: str, device: torch.device):
    if not args.allow_download:
        checkpoint = robustbench_checkpoint_path(args, model_name)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Missing local RobustBench checkpoint:\n"
                f"{checkpoint}\n"
                "Download it manually or rerun with --allow_download if the server can access Google Drive."
            )
    from robustbench.utils import load_model

    model = load_model(
        model_name=model_name,
        dataset=args.dataset,
        threat_model=args.threat_model,
        model_dir=args.model_dir,
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def eval_asr(model, loader: DataLoader, targeted: bool, device: torch.device) -> float:
    success = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            if targeted:
                success += (predictions == labels).sum().item()
            else:
                success += (predictions != labels).sum().item()
            total += labels.numel()
    if total == 0:
        return float("nan")
    return 100.0 * float(success) / float(total)


def make_initial_rows(args: argparse.Namespace, sources: List[str], attacks: List[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
    eps_label, eps_value = parse_epsilon_token(args.eps)
    rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    for source in sources:
        for attack in attacks:
            case_dir = case_output_dir(args, source, attack)
            rows[(source, attack)] = {
                "source": source,
                "source_label": display_source(source),
                "attack": attack,
                "attack_label": display_attack(attack),
                "epsilon_label": eps_label,
                "epsilon": eps_value,
                "targeted": args.targeted,
                "generation_seconds": load_generation_seconds(case_dir),
            }
    return rows


def evaluate_cases(
    args: argparse.Namespace,
    sources: List[str],
    attacks: List[str],
    robust_models: List[str],
    device: torch.device,
) -> List[Dict[str, object]]:
    expected = expected_filenames(args.input_dir)
    rows = make_initial_rows(args, sources, attacks)

    for source in sources:
        for attack in attacks:
            case_dir = case_output_dir(args, source, attack)
            if not has_expected_images(case_dir, expected):
                missing = [name for name in expected if not (case_dir / name).is_file()]
                raise FileNotFoundError(
                    f"Missing adversarial examples for source={source}, attack={attack}.\n"
                    f"Directory: {case_dir}\n"
                    f"First missing file: {missing[0] if missing else 'unknown'}"
                )

    for robust_model_name in robust_models:
        column = safe_column_name(robust_model_name)
        print("\n" + "=" * 80)
        print(f"Loading RobustBench model: {robust_model_name}")
        print(f"Column: {column}")
        model = load_robustbench_model(args, robust_model_name, device)

        for source in sources:
            for attack in attacks:
                case_dir = case_output_dir(args, source, attack)
                dataset = RobustBenchEvalDataset(
                    input_dir=args.input_dir,
                    adv_dir=case_dir,
                    targeted=args.targeted,
                )
                loader = make_loader(dataset, args.eval_batchsize, args)
                print(
                    f"Evaluating source={source}, attack={attack}, "
                    f"target={robust_model_name}, images={case_dir}"
                )
                asr = eval_asr(model, loader, args.targeted, device)
                rows[(source, attack)][column] = asr
                print(f"{source} | {attack} -> {display_robust_model(robust_model_name)}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    robust_columns = [safe_column_name(name) for name in robust_models]
    final_rows: List[Dict[str, object]] = []
    for source in sources:
        for attack in attacks:
            row = rows[(source, attack)]
            values = [float(row[col]) for col in robust_columns if col in row]
            row["avg"] = float(np.mean(values)) if values else float("nan")
            final_rows.append(row)
    return final_rows


def result_fieldnames(robust_models: List[str]) -> List[str]:
    return [
        "source",
        "source_label",
        "attack",
        "attack_label",
        "epsilon_label",
        "epsilon",
        "targeted",
        "generation_seconds",
    ] + [safe_column_name(model_name) for model_name in robust_models] + ["avg"]


def write_result_csv(path: Path, rows: List[Dict[str, object]], robust_models: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames(robust_models)
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def best_second_by_source(
    rows: List[Dict[str, object]],
    sources: List[str],
    columns: List[str],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for source in sources:
        source_rows = [row for row in rows if row["source"] == source]
        for column in columns:
            values = sorted(
                [as_float(row, column) for row in source_rows if np.isfinite(as_float(row, column))],
                reverse=True,
            )
            best = values[0] if values else float("nan")
            second = values[1] if len(values) > 1 else float("nan")
            lookup[(source, column)] = (best, second)
    return lookup


def format_latex_value(value: float, best: float, second: float) -> str:
    text = f"{value:.1f}"
    if np.isfinite(best) and abs(value - best) <= 1e-9:
        return f"\\textbf{{{text}}}"
    if np.isfinite(second) and abs(value - second) <= 1e-9:
        return f"\\underline{{{text}}}"
    return text


def write_latex_outputs(
    tex_rows_path: Path,
    full_tex_path: Path,
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    robust_models: List[str],
) -> None:
    tex_rows_path.parent.mkdir(parents=True, exist_ok=True)
    robust_columns = [safe_column_name(name) for name in robust_models]
    display_columns = [display_robust_model(name) for name in robust_models]
    score_columns = robust_columns + ["avg"]
    best_lookup = best_second_by_source(rows, sources, score_columns)
    row_map = {(row["source"], row["attack"]): row for row in rows}

    lines: List[str] = []
    for source in sources:
        for attack in attacks:
            row = row_map[(source, attack)]
            cells = [display_source(source), display_attack(attack)]
            for column in robust_columns:
                cells.append(
                    format_latex_value(
                        as_float(row, column),
                        *best_lookup[(source, column)],
                    )
                )
            cells.append(
                format_latex_value(
                    as_float(row, "avg"),
                    *best_lookup[(source, "avg")],
                )
            )
            lines.append(" & ".join(cells) + r" \\")

    with open(tex_rows_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    column_spec = "c c " + " ".join(["c"] * len(robust_models)) + " c"
    header = "Source & Attack & " + " & ".join(display_columns) + " & Avg. \\\\"
    full_table = rf"""\begin{{table*}}[!htb]
\centering
\caption{{Attack success rates (\%) against modern RobustBench ImageNet $\ell_\infty$ robust models. The robust targets are loaded from RobustBench Model Zoo with official checkpoints. Avg. is computed over the listed robust targets.}}
\label{{tab:modern_robust_models}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{{column_spec}}}
\toprule
{header}
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    with open(full_tex_path, "w", encoding="utf-8") as f:
        f.write(full_table)


def write_analysis(
    analysis_file: Path,
    rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
    robust_models: List[str],
    args: argparse.Namespace,
) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    robust_columns = [safe_column_name(name) for name in robust_models]
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Modern RobustBench ImageNet robust-model transfer evaluation\n")
        f.write("=" * 68 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Threat model of robust targets: {args.threat_model}\n")
        f.write(f"Attack budget for generated examples: {args.eps}\n")
        f.write(f"Targeted: {args.targeted}\n")
        f.write(f"Sources: {', '.join(sources)}\n")
        f.write(f"Attacks: {', '.join(attacks)}\n")
        f.write("RobustBench model IDs:\n")
        for model_name in robust_models:
            f.write(f"  {display_robust_model(model_name)}: {model_name}\n")
        f.write("\nResults:\n")
        for row in rows:
            target_values = ", ".join(
                f"{display_robust_model(model)}={as_float(row, safe_column_name(model)):.2f}%"
                for model in robust_models
            )
            f.write(
                f"{row['source_label']} | {row['attack_label']}: "
                f"{target_values}, Avg={as_float(row, 'avg'):.2f}%, "
                f"generation={as_float(row, 'generation_seconds'):.2f}s\n"
            )


def list_available_robustbench_models(args: argparse.Namespace) -> None:
    from robustbench.model_zoo import model_dicts

    models = model_dicts[args.dataset][args.threat_model]
    print(f"Available RobustBench models for dataset={args.dataset}, threat_model={args.threat_model}:")
    for model_name in sorted(models.keys()):
        print(model_name)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adversarial examples and evaluate transfer ASR against modern "
            "RobustBench ImageNet robust models."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--models", default=DEFAULT_ROBUST_MODELS)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./modern_robustbench_eval")
    parser.add_argument(
        "--adv_dir_template",
        default="",
        help=(
            "Optional adversarial-image directory template. Available fields: "
            "{source}, {source_label}, {attack}, {attack_label}, {eps_label}, {eps_dir}, {output_dir}. "
            "Example: './results/eda/{source}-{attack}'"
        ),
    )
    parser.add_argument("--model_dir", default="./robustbench_models")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--threat_model", default="Linf")
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow RobustBench to download missing checkpoints. By default, local checkpoint files are required.",
    )
    parser.add_argument("--list_models", action="store_true")

    parser.add_argument(
        "--batchsize",
        default=0,
        type=int,
        help="Generation batch size. Use 0 for automatic source/attack-specific values.",
    )
    parser.add_argument("--eval_batchsize", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--GPU_ID",
        default="0",
        help="Logical CUDA device id. If CUDA_VISIBLE_DEVICES is set externally, usually use --GPU_ID 0.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)
    parser.add_argument("--reuse_existing", action="store_true")

    parser.add_argument("--eps", default=DEFAULT_EPSILON)
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--decay", default=1.0, type=float)
    parser.add_argument("--norm", default="linfty")
    parser.add_argument("--loss", default="crossentropy")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--target_class", default=None, type=int)
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--mesh_width", default=3, type=int)
    parser.add_argument("--mesh_height", default=3, type=int)
    parser.add_argument("--noise_scale", default=0.45, type=float)
    parser.add_argument(
        "--num_warping",
        default=25,
        type=int,
        help=(
            "Matched transformed-sample count N: num_scale for BSR/SID and "
            "num_warping for DeCoWA/EDA. L2T and OPS keep their original settings."
        ),
    )

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--tex_rows", default="")
    parser.add_argument("--full_tex", default="")
    parser.add_argument("--analysis_file", default="")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    set_seed(args.seed)
    sources = parse_list(args.sources)
    attacks = parse_list(args.attacks)
    robust_models = parse_list(args.models)
    device = resolve_device(args)
    print(f"Using device: {device}")

    if args.list_models:
        list_available_robustbench_models(args)
        return

    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, attacks)

    if args.mode in {"eval", "both"}:
        rows = evaluate_cases(args, sources, attacks, robust_models, device)
        result_csv, tex_rows, full_tex, analysis_file = output_paths(args)
        write_result_csv(result_csv, rows, robust_models)
        write_latex_outputs(tex_rows, full_tex, rows, sources, attacks, robust_models)
        write_analysis(analysis_file, rows, sources, attacks, robust_models, args)
        print("\nSaved:")
        print(f"  CSV:       {result_csv}")
        print(f"  TeX rows:  {tex_rows}")
        print(f"  TeX table: {full_tex}")
        print(f"  Analysis:  {analysis_file}")


if __name__ == "__main__":
    main()
