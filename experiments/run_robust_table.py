import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = "resnet18,inception_v3,inception_v4,inception_resnet_v2"
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}

ROBUST_COLUMNS = [
    "adv-RN-50(l2)",
    "adv-RN-50(linf)",
    "FAT(2px)",
    "FAT(4px)",
    "Inc-v3(adv)",
    "IncRes-v2(adv)",
    "HGD",
    "NRP(RN-101)",
    "NRP(ViT-B)",
    "RS",
]

DIRECT_MODEL_COLUMNS = {
    "adv-RN-50(l2)": ("resnet50_l2_eps1", 0),
    "adv-RN-50(linf)": ("resnet50_linf_eps4", 0),
    "Inc-v3(adv)": ("adv_inception_v3", 0),
    "IncRes-v2(adv)": ("ens_adv_inception_resnet_v2", 0),
    "NRP(RN-101)": ("resnet101", 0),
    "NRP(ViT-B)": ("vit_base_patch16_224", 0),
}

TIMM_MODEL_NAMES = {
    "adv_inception_v3",
    "ens_adv_inception_resnet_v2",
    "inception_v3.tf_adv_in1k",
    "inception_resnet_v2.tf_ens_adv_in1k",
    "vit_base_patch16_224",
}


class PreprocessingModel(nn.Module):
    def __init__(self, resize: int, mean: Iterable[float], std: Iterable[float]) -> None:
        super().__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.resize(x))


class RobustEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        targeted: bool = False,
    ) -> None:
        self.targeted = targeted
        self.data_dir = output_dir
        self.f2l = load_dataset_labels(input_dir / "labels.csv", targeted=targeted)

    def __len__(self) -> int:
        return len(self.f2l)

    def __getitem__(self, idx: int):
        filename = list(self.f2l.keys())[idx]
        filepath = self.data_dir / filename
        image = Image.open(filepath).resize((IMG_HEIGHT, IMG_WIDTH)).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]
        return image, label, filename


def wrap_model(model: nn.Module) -> nn.Module:
    model_name = model.__class__.__name__
    resize = 224
    if hasattr(model, "default_cfg"):
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
    elif "Inc" in model_name:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        resize = 299
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return nn.Sequential(PreprocessingModel(resize, mean, std), model)


def parse_sources(raw_sources: str) -> List[str]:
    sources = [item.strip() for item in raw_sources.split(",") if item.strip()]
    if not sources:
        raise ValueError("At least one source model is required.")
    return sources


def parse_columns(raw_columns: str) -> List[str]:
    if raw_columns.strip().lower() == "all":
        return ROBUST_COLUMNS
    columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
    unknown = [item for item in columns if item not in ROBUST_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown robust table columns: {unknown}")
    return columns


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def source_adv_dir(args: argparse.Namespace, source: str) -> Path:
    if args.adv_dir_template:
        value = args.adv_dir_template.format(
            source=source,
            attack=args.attack,
            run_name=args.run_name,
        )
        return Path(value).resolve()
    adv_root = Path(args.adv_root)
    if args.run_name:
        return (adv_root / args.run_name / f"{source}-{args.attack}").resolve()
    return (adv_root / f"{source}-{args.attack}").resolve()


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def ensure_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def list_image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def image_dir_has_expected_files(directory: Path, expected_names: Iterable[str]) -> bool:
    expected = set(expected_names)
    present = {path.name for path in list_image_files(directory)}
    return expected.issubset(present)


def link_or_copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.symlink(source, destination)
    except OSError:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def prepare_nrp_image_input_dir(
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> Path:
    label_file = Path(args.input_dir).resolve() / "labels.csv"
    expected_names = set(load_labels_map(label_file, targeted=args.targeted).keys())
    input_dir = (
        result_root
        / "nrp_inputs"
        / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_images"
    ).resolve()

    if args.reuse_existing and image_dir_has_expected_files(input_dir, expected_names):
        print(f"Reusing image-only NRP input: {input_dir}")
        return input_dir

    input_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in list(input_dir.iterdir()):
        if stale_file.is_file() or stale_file.is_symlink():
            stale_file.unlink()

    image_files = list_image_files(adv_dir)
    by_name = {path.name: path for path in image_files}
    missing = sorted(expected_names - set(by_name))
    if missing:
        raise FileNotFoundError(
            f"NRP input is missing {len(missing)} labeled image files in {adv_dir}. "
            f"First missing file: {missing[0]}"
        )

    for filename in sorted(expected_names):
        link_or_copy_file(by_name[filename], input_dir / filename)
    return input_dir


def prepare_rs_image_input_dir(
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> Path:
    label_file = Path(args.input_dir).resolve() / "labels.csv"
    expected_names = set(load_labels_map(label_file, targeted=args.targeted).keys())
    input_dir = (
        result_root
        / "rs_inputs"
        / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_images"
    ).resolve()

    if args.reuse_existing and image_dir_has_expected_files(input_dir, expected_names):
        print(f"Reusing image-only RS input: {input_dir}")
        return input_dir

    input_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in list(input_dir.iterdir()):
        if stale_file.is_file() or stale_file.is_symlink():
            stale_file.unlink()

    image_files = list_image_files(adv_dir)
    by_name = {path.name: path for path in image_files}
    missing = sorted(expected_names - set(by_name))
    if missing:
        raise FileNotFoundError(
            f"RS input is missing {len(missing)} labeled image files in {adv_dir}. "
            f"First missing file: {missing[0]}"
        )

    for filename in sorted(expected_names):
        link_or_copy_file(by_name[filename], input_dir / filename)
    return input_dir


def load_labels_map(label_file: Path, targeted: bool = False) -> Dict[str, int]:
    dev = pd.read_csv(label_file)
    if targeted:
        dev = dev.iloc[:, [0, 2]]
    else:
        dev = dev.iloc[:, [0, 1]]
    dev.columns = ["filename", "label"]
    return {
        str(dev.iloc[i]["filename"]): int(dev.iloc[i]["label"])
        for i in range(len(dev))
    }


def load_dataset_labels(label_file: Path, targeted: bool = False) -> Dict[str, object]:
    dev = pd.read_csv(label_file)
    if targeted:
        return {
            str(dev.iloc[i]["filename"]): [
                int(dev.iloc[i]["label"]),
                int(dev.iloc[i]["targeted_label"]),
            ]
            for i in range(len(dev))
        }
    return {
        str(dev.iloc[i]["filename"]): int(dev.iloc[i]["label"])
        for i in range(len(dev))
    }


def load_output_map(output_file: Path) -> Dict[str, int]:
    dev = pd.read_csv(output_file, header=None)
    dev = dev.iloc[:, [0, 1]]
    dev.columns = ["filename", "label"]
    if len(dev) and str(dev.iloc[0]["filename"]) == "filename":
        dev = dev.iloc[1:]
    return {
        str(dev.iloc[i]["filename"]): int(dev.iloc[i]["label"])
        for i in range(len(dev))
    }


def compute_asr_from_one_based_output(
    label_file: Path,
    output_file: Path,
    targeted: bool,
) -> float:
    labels = load_labels_map(label_file, targeted=targeted)
    outputs = load_output_map(output_file)
    missing = sorted(set(labels) - set(outputs))
    extra = sorted(set(outputs) - set(labels))
    if missing or extra:
        raise ValueError(
            f"Output/label filename mismatch for {output_file}. "
            f"Missing={len(missing)}, extra={len(extra)}"
        )

    success = 0
    for filename, label in labels.items():
        predicted = outputs[filename]
        expected = label + 1
        if targeted:
            success += int(predicted == expected)
        else:
            success += int(predicted != expected)
    return 100.0 * success / len(labels)


def make_loader(
    input_dir: Path,
    adv_dir: Path,
    args: argparse.Namespace,
) -> DataLoader:
    dataset = RobustEvalDataset(
        input_dir=input_dir,
        output_dir=adv_dir,
        targeted=args.targeted,
    )
    return DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def eval_model_asr(
    model: nn.Module,
    loader: DataLoader,
    targeted: bool,
    label_offset: int = 0,
) -> float:
    model.eval()
    success = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            if targeted:
                labels = labels[1]
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True) + label_offset
            predictions = model(images).argmax(dim=1)
            if targeted:
                success += (predictions == labels).sum().item()
            else:
                success += (predictions != labels).sum().item()
            total += labels.shape[0]
    return 100.0 * success / total


def load_torch_or_timm_model(model_name: str) -> nn.Module:
    if model_name == "resnet50_l2_eps1":
        try:
            from transferattack.MadryLab import model_utils as madry_model_utils
            from transferattack.MadryLab.datasets import ImageNet as MadryImageNet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Madry robust model loading needs the missing dependency "
                f"'{exc.name}'. Install it in this environment, then rerun."
            ) from exc
        model, _ = madry_model_utils.make_and_restore_model(
            arch="resnet50",
            dataset=MadryImageNet(""),
            resume_path=str(REPO_ROOT / "robust-imagenet-models" / "resnet50_l2_eps1.ckpt"),
        )
        model = model.model
    elif model_name == "resnet50_linf_eps4":
        try:
            from transferattack.MadryLab import model_utils as madry_model_utils
            from transferattack.MadryLab.datasets import ImageNet as MadryImageNet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Madry robust model loading needs the missing dependency "
                f"'{exc.name}'. Install it in this environment, then rerun."
            ) from exc
        model, _ = madry_model_utils.make_and_restore_model(
            arch="resnet50",
            dataset=MadryImageNet(""),
            resume_path=str(REPO_ROOT / "robust-imagenet-models" / "resnet50_linf_eps4.0.ckpt"),
        )
        model = model.model
    elif model_name in TIMM_MODEL_NAMES:
        model = timm.create_model(model_name, pretrained=True)
    else:
        try:
            model = models.__dict__[model_name](weights="DEFAULT")
        except TypeError:
            model = models.__dict__[model_name](pretrained=True)
    model = wrap_model(model.eval().cuda())
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def evaluate_direct_column(
    column: str,
    adv_dir: Path,
    args: argparse.Namespace,
) -> float:
    model_name, label_offset = DIRECT_MODEL_COLUMNS[column]
    print(f"Evaluating {column} on {adv_dir}")
    model = load_torch_or_timm_model(model_name)
    loader = make_loader(Path(args.input_dir), adv_dir, args)
    asr = eval_model_asr(model, loader, args.targeted, label_offset=label_offset)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return asr


def run_command(command: List[str], cwd: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command))
    print("Log:", log_file)
    with open(log_file, "w", encoding="utf-8") as log:
        try:
            subprocess.run(
                command,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"Command failed. Last log lines from {log_file}:")
            print(read_log_tail(log_file))
            raise


def read_log_tail(log_file: Path, max_lines: int = 80) -> str:
    if not log_file.is_file():
        return "<log file was not created>"
    text = log_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "<log file is empty>"


def maybe_skip_existing(
    output_file: Path,
    label_file: Path,
    targeted: bool,
    reuse_existing: bool,
) -> Optional[float]:
    if reuse_existing and output_file.is_file():
        print(f"Reusing existing output: {output_file}")
        return compute_asr_from_one_based_output(label_file, output_file, targeted)
    return None


def run_fat_column(
    pixels: int,
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> float:
    label_file = Path(args.input_dir).resolve() / "labels.csv"
    output_rel = Path("at_results") / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_fat{pixels}px.txt"
    output_rel_posix = output_rel.as_posix()
    output_file = REPO_ROOT / "defense" / "at" / output_rel
    reused = maybe_skip_existing(output_file, label_file, args.targeted, args.reuse_existing)
    if reused is not None:
        return reused

    at_dir = REPO_ROOT / "defense" / "at"
    config = Path("configs") / f"configs_fast_{pixels}px_evaluate.yml"
    checkpoint = REPO_ROOT / "defense" / "models" / f"imagenet_model_weights_{pixels}px.pth.tar"
    ensure_file(at_dir / config, f"FAT {pixels}px config")
    ensure_file(checkpoint, f"FAT {pixels}px checkpoint")

    command = [
        sys.executable,
        "main_fast.py",
        "--data",
        str(adv_dir),
        "--config",
        config.as_posix(),
        "--output_prefix",
        output_rel_posix,
        "--resume",
        str(checkpoint),
        "--evaluate",
        "--restarts",
        str(args.fat_restarts),
        "--batchsize",
        str(args.defense_batchsize),
        "--GPU_ID",
        args.GPU_ID,
    ]
    log_file = result_root / "logs" / f"{slug(source)}_{slug(args.attack)}_fat{pixels}px.log"
    run_command(command, cwd=at_dir, log_file=log_file)
    return compute_asr_from_one_based_output(label_file, output_file, args.targeted)


def run_hgd_column(
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> float:
    label_file = Path(args.input_dir).resolve() / "labels.csv"
    output_rel = Path("hgd_results") / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_hgd.txt"
    output_rel_posix = output_rel.as_posix()
    output_file = REPO_ROOT / "defense" / "hgd" / output_rel
    reused = maybe_skip_existing(output_file, label_file, args.targeted, args.reuse_existing)
    if reused is not None:
        return reused

    hgd_dir = REPO_ROOT / "defense" / "hgd"
    checkpoint_dir = hgd_dir / "checkpoint"
    for name in [
        "denoise_res_015.ckpt",
        "denoise_inres_014.ckpt",
        "denoise_incepv3_012.ckpt",
        "denoise_rex_001.ckpt",
    ]:
        ensure_file(checkpoint_dir / name, f"HGD checkpoint {name}")

    command = [
        sys.executable,
        "defense.py",
        "--input_dir",
        str(adv_dir),
        "--output_file",
        output_rel_posix,
        "--checkpoint_dir_path",
        "checkpoint",
        "--batch-size",
        str(args.defense_batchsize),
        "--GPU_ID",
        args.GPU_ID,
    ]
    log_file = result_root / "logs" / f"{slug(source)}_{slug(args.attack)}_hgd.log"
    run_command(command, cwd=hgd_dir, log_file=log_file)
    return compute_asr_from_one_based_output(label_file, output_file, args.targeted)


def purify_with_nrp(
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> Path:
    output_dir = (
        result_root
        / "purified_data"
        / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_nrp"
    ).resolve()
    expected_names = set(
        load_labels_map(Path(args.input_dir).resolve() / "labels.csv", targeted=args.targeted).keys()
    )
    if args.reuse_existing and image_dir_has_expected_files(output_dir, expected_names):
        print(f"Reusing existing NRP purified data: {output_dir}")
        return output_dir

    nrp_script = REPO_ROOT / "defense" / "nrp" / "purify.py"
    model_path = REPO_ROOT / "defense" / "models" / "NRP.pth"
    ensure_file(nrp_script, "NRP purification script")
    ensure_file(model_path, "NRP checkpoint")
    nrp_input_dir = prepare_nrp_image_input_dir(source, adv_dir, args, result_root)

    base_command = [
        sys.executable,
        str(nrp_script),
        "--dir",
        str(nrp_input_dir),
        "--output",
        str(output_dir),
        "--purifier",
        "NRP",
        "--model_pth",
        str(model_path),
        "--GPU_ID",
        args.GPU_ID,
    ]
    log_file = result_root / "logs" / f"{slug(source)}_{slug(args.attack)}_nrp.log"

    if args.nrp_dynamic_mode == "value":
        command = base_command[:-2] + ["--dynamic", "True"] + base_command[-2:]
        run_command(command, cwd=REPO_ROOT / "defense", log_file=log_file)
    elif args.nrp_dynamic_mode == "flag":
        command = base_command[:-2] + ["--dynamic"] + base_command[-2:]
        run_command(command, cwd=REPO_ROOT / "defense", log_file=log_file)
    elif args.nrp_dynamic_mode == "none":
        run_command(base_command, cwd=REPO_ROOT / "defense", log_file=log_file)
    else:
        command = base_command[:-2] + ["--dynamic", "True"] + base_command[-2:]
        try:
            run_command(command, cwd=REPO_ROOT / "defense", log_file=log_file)
        except subprocess.CalledProcessError:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            if "unrecognized arguments: True" not in log_text:
                raise
            print("Retrying NRP with flag-style --dynamic.")
            command = base_command[:-2] + ["--dynamic"] + base_command[-2:]
            run_command(command, cwd=REPO_ROOT / "defense", log_file=log_file)

    ensure_dir(output_dir, "NRP purified output")
    return output_dir


def run_rs_column(
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
) -> float:
    label_file = Path(args.input_dir).resolve() / "labels.csv"
    log_file = result_root / "logs" / f"{slug(source)}_{slug(args.attack)}_rs.log"
    results_file = (
        result_root
        / "defense_outputs"
        / f"{slug(args.run_name)}_{slug(source)}_{slug(args.attack)}_rs_results.txt"
    )
    if args.reuse_existing and results_file.is_file():
        text = results_file.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"Final Attack Success Rate:\s*([0-9.]+)%", text)
        if matches:
            print(f"Reusing existing RS result: {results_file}")
            return float(matches[-1])
    if args.reuse_existing and log_file.is_file():
        recovered = recover_rs_result_from_log(log_file, results_file, label_file)
        if recovered is not None:
            print(f"Recovered completed RS result from previous log: {log_file}")
            return recovered

    rs_dir = REPO_ROOT / "defense" / "rs"
    checkpoint = rs_dir / "RS_noise_0.50" / "checkpoint.pth.tar"
    ensure_file(checkpoint, "randomized smoothing checkpoint")
    rs_input_dir = prepare_rs_image_input_dir(source, adv_dir, args, result_root)

    command = [
        sys.executable,
        "predict.py",
        "--input",
        str(rs_input_dir),
        "--label_file",
        str(label_file),
        "--base_classifier",
        str(checkpoint),
        "--sigma",
        str(args.rs_sigma),
        "--outfile",
        str(result_root / "defense_outputs" / f"{slug(source)}_{slug(args.attack)}_rs_predictions.txt"),
        "--alpha",
        str(args.rs_alpha),
        "--N",
        str(args.rs_N),
        "--skip",
        str(args.rs_skip),
        "--batch",
        str(args.rs_batch),
        "--GPU_ID",
        args.GPU_ID,
        "--results_file",
        str(results_file),
    ]
    if args.targeted:
        command.append("--targeted")
    run_command(command, cwd=rs_dir, log_file=log_file)

    text = results_file.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Final Attack Success Rate:\s*([0-9.]+)%", text)
    if not matches:
        raise RuntimeError(f"Could not parse RS ASR from {results_file}")
    return float(matches[-1])


def recover_rs_result_from_log(
    log_file: Path,
    results_file: Path,
    label_file: Path,
) -> Optional[float]:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    final_matches = re.findall(r"Final Attack Success Rate:\s*([0-9.]+)%", text)
    if final_matches:
        return float(final_matches[-1])

    expected_total = len(load_labels_map(label_file, targeted=False))
    progress_matches = [int(item) for item in re.findall(r"(\d+)it\s*\[", text)]
    rate_matches = re.findall(r"Attack Success Rate:\s*([0-9.]+)%", text)
    if not progress_matches or not rate_matches:
        return None
    if max(progress_matches) < expected_total:
        return None

    recovered = float(rate_matches[-1])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"Recovered from log: {log_file}\n")
        f.write(f"Total samples: {expected_total}\n")
        f.write(f"Final Attack Success Rate: {recovered:.2f}%\n")
        f.write("-" * 50 + "\n")
    return recovered


def evaluate_column(
    column: str,
    source: str,
    adv_dir: Path,
    args: argparse.Namespace,
    result_root: Path,
    nrp_cache: Dict[str, Path],
) -> float:
    if column in {"adv-RN-50(l2)", "adv-RN-50(linf)", "Inc-v3(adv)", "IncRes-v2(adv)"}:
        return evaluate_direct_column(column, adv_dir, args)
    if column == "FAT(2px)":
        return run_fat_column(2, source, adv_dir, args, result_root)
    if column == "FAT(4px)":
        return run_fat_column(4, source, adv_dir, args, result_root)
    if column == "HGD":
        return run_hgd_column(source, adv_dir, args, result_root)
    if column in {"NRP(RN-101)", "NRP(ViT-B)"}:
        if source not in nrp_cache:
            nrp_cache[source] = purify_with_nrp(source, adv_dir, args, result_root)
        return evaluate_direct_column(column, nrp_cache[source], args)
    if column == "RS":
        return run_rs_column(source, adv_dir, args, result_root)
    raise ValueError(f"Unsupported column: {column}")


def format_value(value: float) -> str:
    return f"{value:.1f}"


def write_outputs(
    rows: List[Dict[str, object]],
    columns: List[str],
    result_root: Path,
    output_tag: str,
) -> Tuple[Path, Path]:
    result_root.mkdir(parents=True, exist_ok=True)
    csv_path = result_root / f"robust_table_{slug(output_tag)}.csv"
    tex_path = result_root / f"robust_table_{slug(output_tag)}_latex_rows.txt"

    fieldnames = ["Source", "Attack"] + columns + ["Avg."]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(tex_path, "w", encoding="utf-8") as f:
        for row in rows:
            values = [str(row["Source"]), str(row["Attack"])]
            values.extend(str(row[column]) for column in columns)
            values.append(str(row["Avg."]))
            f.write(" & ".join(values) + r" \\" + "\n")

    return csv_path, tex_path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adversarial examples against the ten defense/robust "
            "targets used in the Robusts table."
        )
    )
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--attack", default="eda")
    parser.add_argument(
        "--attacks",
        default="",
        help="Optional comma-separated attacks. Overrides --attack when set.",
    )
    parser.add_argument("--adv_root", default="./results/eda")
    parser.add_argument("--run_name", default="")
    parser.add_argument(
        "--adv_dir_template",
        default="",
        help=(
            "Optional template for adversarial directories. Available fields: "
            "{source}, {attack}, {run_name}."
        ),
    )
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--result_root", default="./robust_table_results")
    parser.add_argument("--columns", default="all")
    parser.add_argument("--batchsize", default=32, type=int)
    parser.add_argument("--defense_batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse existing defense outputs/purified images when present.",
    )

    parser.add_argument("--fat_restarts", default=10, type=int)
    parser.add_argument("--rs_sigma", default=0.50, type=float)
    parser.add_argument("--rs_N", default=1000, type=int)
    parser.add_argument("--rs_alpha", default=0.001, type=float)
    parser.add_argument("--rs_skip", default=100, type=int)
    parser.add_argument("--rs_batch", default=1, type=int)
    parser.add_argument(
        "--nrp_dynamic_mode",
        choices=["auto", "value", "flag", "none"],
        default="auto",
        help=(
            "How to pass NRP dynamic inference. auto first tries "
            "'--dynamic True' and retries '--dynamic' if that parser rejects True."
        ),
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    set_seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    label_file = input_dir / "labels.csv"
    ensure_file(label_file, "label file")

    sources = parse_sources(args.sources)
    attacks = parse_sources(args.attacks) if args.attacks else [args.attack]
    columns = parse_columns(args.columns)
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    output_tag = attacks[0] if len(attacks) == 1 else "multi_attack"

    for source in sources:
        for attack in attacks:
            args.attack = attack
            nrp_cache: Dict[str, Path] = {}
            adv_dir = source_adv_dir(args, source)
            ensure_dir(adv_dir, f"adversarial image directory for source={source}")
            print("\n" + "=" * 80)
            print(f"Source: {source}")
            print(f"Attack: {args.attack}")
            print(f"Adversarial images: {adv_dir}")

            results: Dict[str, float] = {}
            for column in columns:
                asr = evaluate_column(column, source, adv_dir, args, result_root, nrp_cache)
                results[column] = asr
                print(f"{source} -> {column}: {asr:.2f}%")

            avg = float(np.mean([results[column] for column in columns]))
            attack_name = args.attack.upper() if args.attack.lower() == "eda" else args.attack
            row: Dict[str, object] = {"Source": source, "Attack": attack_name}
            for column in columns:
                row[column] = format_value(results[column])
            row["Avg."] = format_value(avg)
            rows.append(row)
            write_outputs(rows, columns, result_root, output_tag)

    csv_path, tex_path = write_outputs(rows, columns, result_root, output_tag)
    print("\nCompleted robust-table evaluation.")
    print(f"CSV: {csv_path}")
    print(f"LaTeX rows: {tex_path}")


if __name__ == "__main__":
    main()
