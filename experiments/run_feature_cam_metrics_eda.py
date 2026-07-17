import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm

if not hasattr(timm.models, "hub"):
    class _TimmHubCompat:
        HUB_SERVER = None

    timm.models.hub = _TimmHubCompat()

SCRIPT_DIR = Path(__file__).resolve().parent


def locate_repo_root(start_dir: Path) -> Path:
    for path in [start_dir, *start_dir.parents]:
        if (path / "transferattack" / "__init__.py").is_file():
            return path
    searched = "\n".join(str(path) for path in [start_dir, *start_dir.parents])
    raise FileNotFoundError(
        "Cannot find TransferAttack project root containing transferattack/__init__.py. "
        "Searched:\n" + searched
    )


REPO_ROOT = locate_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GRADCAM_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR / "pytorch-grad-cam",
    REPO_ROOT / "pytorch-grad-cam",
    REPO_ROOT / "pytorch-grad-cam" / "pytorch-grad-cam",
]
GRADCAM_ROOT = next(
    (
        path
        for path in GRADCAM_CANDIDATES
        if (path / "pytorch_grad_cam" / "__init__.py").is_file()
    ),
    None,
)
if GRADCAM_ROOT is None:
    searched = "\n".join(str(path) for path in GRADCAM_CANDIDATES)
    raise FileNotFoundError(
        "Cannot find local pytorch_grad_cam package. Searched:\n" + searched
    )
if str(GRADCAM_ROOT) not in sys.path:
    sys.path.insert(0, str(GRADCAM_ROOT))

from pytorch_grad_cam import GradCAM

import transferattack
from transferattack.utils import AdvDataset, save_images


GLOBAL_SEED = 42
DEFAULT_SOURCE = "resnet18"
DEFAULT_ATTACKS = "l2t,bsr,decowa,ops,sid,eda"
DEFAULT_EPSILON = "16/255"
DEFAULT_GENERATION_BATCHSIZE = 32
SWIN_TARGET_MODEL = "swin_tiny_patch4_window7_224"

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

PRIMARY_METRICS = [
    ("feature_l2", "max"),
    ("feature_kl", "max"),
    ("feature_cosine", "min"),
    ("cam_l2", "max"),
    ("cam_js", "max"),
    ("cam_corr", "min"),
]
ALL_METRICS = [
    "feature_l2",
    "feature_kl",
    "feature_cosine",
    "feature_js",
    "feature_corr",
    "cam_l2",
    "cam_kl",
    "cam_js",
    "cam_corr",
    "cam_cosine",
]


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


def eps_dir_name(eps_label: str) -> str:
    return "eps_" + eps_label.replace("/", "_").replace(".", "p")


def display_source(source: str) -> str:
    return SOURCE_DISPLAY.get(source, source)


def display_attack(attack: str) -> str:
    return ATTACK_DISPLAY.get(attack, attack.upper())


def expected_filenames(input_dir: str) -> List[str]:
    labels = pd.read_csv(Path(input_dir) / "labels.csv")
    return [str(filename) for filename in labels["filename"].tolist()]


def has_expected_images(directory: Path, filenames: Iterable[str]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


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


def make_loader(dataset: AdvDataset, batch_size: int, args: argparse.Namespace) -> DataLoader:
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
    return Path(args.output_dir) / display_source(source).replace("-", "") / eps_dir_name(eps_label) / attack


def case_meta_path(case_dir: Path) -> Path:
    return case_dir / "case_meta.json"


def load_generation_seconds(case_dir: Path) -> Optional[float]:
    meta_path = case_meta_path(case_dir)
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("elapsed_seconds"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    output_dir = Path(args.output_dir)
    per_image_csv = (
        Path(args.per_image_csv)
        if args.per_image_csv
        else output_dir / "feature_cam_metrics_per_image.csv"
    )
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_dir / "feature_cam_metrics_summary.csv"
    )
    tex_rows = (
        Path(args.tex_rows)
        if args.tex_rows
        else output_dir / "feature_cam_metrics_table_rows.tex"
    )
    full_tex = (
        Path(args.full_tex)
        if args.full_tex
        else output_dir / "feature_cam_metrics_table.tex"
    )
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "feature_cam_metrics_analysis.txt"
    )
    return per_image_csv, summary_csv, tex_rows, full_tex, analysis_file


def build_attacker(
    args: argparse.Namespace,
    source: str,
    attack: str,
    eps_value: float,
):
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
        common_kwargs.update(num_warping=args.num_warping)
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
        "final_adv_coordinate_system": "original image coordinate system",
    }
    if attack == "eda":
        metadata.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=args.noise_scale,
            num_warping=args.num_warping,
        )
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
                print(f"Skipping existing case: source={source}, attack={attack}")
                continue
            if case_dir.exists() and not args.reuse_existing:
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Generating source={source}, attack={attack}, epsilon={eps_label}")
            print(f"Output: {case_dir}")

            attacker = build_attacker(args, source, attack, eps_value)
            dataset = AdvDataset(
                input_dir=args.input_dir,
                output_dir=str(case_dir),
                targeted=args.targeted,
                eval=False,
            )
            batch_size = resolve_generation_batchsize(args, source, attack)
            print(f"Generation batchsize: {batch_size}")
            loader = make_loader(dataset, batch_size, args)

            start_time = time.perf_counter()
            for images, labels, filenames in tqdm(loader, desc=f"{source}/{attack}"):
                perturbations = attacker(images, labels)
                save_images(str(case_dir), images + perturbations.cpu(), filenames)
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


def swin_t_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_rgb_float(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).resize((image_size, image_size), Image.BILINEAR).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def preprocess_for_swin(rgb_img: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


def normalize_distribution(values: np.ndarray) -> Optional[np.ndarray]:
    arr = values.astype(np.float64)
    arr = arr - np.min(arr)
    total = float(np.sum(arr))
    if total <= 0.0 or not np.isfinite(total):
        return None
    arr = arr / total
    return arr + 1e-10


def calculate_l2_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    return float(torch.norm(tensor1 - tensor2, p=2).item())


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    return float(F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0)).item())


def calculate_kl_divergence(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    tensor1_softmax = F.softmax(tensor1.flatten().unsqueeze(0), dim=1) + 1e-10
    tensor2_softmax = F.softmax(tensor2.flatten().unsqueeze(0), dim=1) + 1e-10
    return float(F.kl_div(tensor1_softmax.log(), tensor2_softmax, reduction="batchmean").item())


def calculate_js_divergence(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    tensor1_softmax = F.softmax(tensor1.flatten().unsqueeze(0), dim=1) + 1e-10
    tensor2_softmax = F.softmax(tensor2.flatten().unsqueeze(0), dim=1) + 1e-10
    midpoint = 0.5 * (tensor1_softmax + tensor2_softmax)
    js = 0.5 * (
        F.kl_div(tensor1_softmax.log(), midpoint, reduction="batchmean")
        + F.kl_div(tensor2_softmax.log(), midpoint, reduction="batchmean")
    )
    return float(js.item())


def calculate_tensor_correlation(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    arr1 = tensor1.detach().flatten().cpu().numpy()
    arr2 = tensor2.detach().flatten().cpu().numpy()
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return 0.0
    value = float(np.corrcoef(arr1, arr2)[0, 1])
    return 0.0 if np.isnan(value) or np.isinf(value) else value


def calculate_cam_kl_divergence(cam1: np.ndarray, cam2: np.ndarray) -> float:
    dist1 = normalize_distribution(cam1)
    dist2 = normalize_distribution(cam2)
    if dist1 is None or dist2 is None:
        return 0.0
    value = float(np.sum(dist1 * np.log(dist1 / dist2)))
    return 0.0 if np.isnan(value) or np.isinf(value) else value


def calculate_cam_js_divergence(cam1: np.ndarray, cam2: np.ndarray) -> float:
    dist1 = normalize_distribution(cam1)
    dist2 = normalize_distribution(cam2)
    if dist1 is None or dist2 is None:
        return 0.0
    midpoint = 0.5 * (dist1 + dist2)
    value = 0.5 * (
        np.sum(dist1 * np.log(dist1 / midpoint))
        + np.sum(dist2 * np.log(dist2 / midpoint))
    )
    value = float(value)
    return 0.0 if np.isnan(value) or np.isinf(value) else value


def calculate_cam_correlation(cam1: np.ndarray, cam2: np.ndarray) -> float:
    arr1 = cam1.flatten()
    arr2 = cam2.flatten()
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return 0.0
    value = float(np.corrcoef(arr1, arr2)[0, 1])
    return 0.0 if np.isnan(value) or np.isinf(value) else value


def calculate_cam_cosine_similarity(cam1: np.ndarray, cam2: np.ndarray) -> float:
    tensor1 = torch.from_numpy(cam1.flatten()).float()
    tensor2 = torch.from_numpy(cam2.flatten()).float()
    return float(F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item())


def build_swin_metric_model(device: torch.device):
    model = timm.create_model(SWIN_TARGET_MODEL, pretrained=True).to(device).eval()
    target_layers = [model.layers[-1].blocks[-1].norm2]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=swin_t_reshape_transform,
    )
    return model, cam


def extract_features_and_cam(
    model,
    cam,
    image_path: Path,
    device: torch.device,
    image_size: int,
    aug_smooth: bool,
    eigen_smooth: bool,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    rgb_img = load_rgb_float(image_path, image_size)
    input_tensor = preprocess_for_swin(rgb_img, device)

    with torch.no_grad():
        features = model.forward_features(input_tensor).detach().cpu()

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth,
    )[0, :]
    grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)
    return features, grayscale_cam, rgb_img


def save_cam_overlay(rgb_img: np.ndarray, grayscale_cam: np.ndarray, output_path: Path) -> None:
    import cv2
    from pytorch_grad_cam.utils.image import show_cam_on_image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), cam_image)


def metric_row_for_pair(
    model,
    cam,
    clean_path: Path,
    adv_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    cam_output_root: Optional[Path],
    source: str,
    attack: str,
    filename: str,
) -> Dict[str, float]:
    clean_features, clean_cam, clean_rgb = extract_features_and_cam(
        model=model,
        cam=cam,
        image_path=clean_path,
        device=device,
        image_size=args.image_size,
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )
    adv_features, adv_cam, adv_rgb = extract_features_and_cam(
        model=model,
        cam=cam,
        image_path=adv_path,
        device=device,
        image_size=args.image_size,
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )

    if cam_output_root is not None:
        stem = Path(filename).stem
        save_cam_overlay(
            clean_rgb,
            clean_cam,
            cam_output_root / display_source(source).replace("-", "") / attack / "clean" / f"{stem}_gradcam.png",
        )
        save_cam_overlay(
            adv_rgb,
            adv_cam,
            cam_output_root / display_source(source).replace("-", "") / attack / "adv" / f"{stem}_gradcam.png",
        )

    return {
        "feature_l2": calculate_l2_distance(clean_features, adv_features),
        "feature_kl": calculate_kl_divergence(clean_features, adv_features),
        "feature_cosine": calculate_cosine_similarity(clean_features, adv_features),
        "feature_js": calculate_js_divergence(clean_features, adv_features),
        "feature_corr": calculate_tensor_correlation(clean_features, adv_features),
        "cam_l2": float(np.linalg.norm(clean_cam - adv_cam)),
        "cam_kl": calculate_cam_kl_divergence(clean_cam, adv_cam),
        "cam_js": calculate_cam_js_divergence(clean_cam, adv_cam),
        "cam_corr": calculate_cam_correlation(clean_cam, adv_cam),
        "cam_cosine": calculate_cam_cosine_similarity(clean_cam, adv_cam),
    }


def safe_mean(values: List[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    return float(np.mean(valid)) if valid else 0.0


def safe_std(values: List[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    return float(np.std(valid)) if len(valid) > 1 else 0.0


def evaluate_cases(args: argparse.Namespace, sources: List[str], attacks: List[str], device: torch.device) -> None:
    expected = expected_filenames(args.input_dir)
    clean_root = Path(args.input_dir) / "images"
    per_image_csv, summary_csv, tex_rows, full_tex, analysis_file = output_paths(args)
    for path in [per_image_csv, summary_csv, tex_rows, full_tex, analysis_file]:
        path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Loading fixed target model for metrics: {SWIN_TARGET_MODEL}")
    model, cam = build_swin_metric_model(device)
    if hasattr(cam, "batch_size"):
        cam.batch_size = args.cam_batchsize

    per_image_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    cam_output_root = Path(args.cam_output_dir) if args.save_cams else None

    for source in sources:
        for attack in attacks:
            case_dir = case_output_dir(args, source, attack)
            if not has_expected_images(case_dir, expected):
                missing = [name for name in expected if not (case_dir / name).is_file()]
                raise FileNotFoundError(
                    f"Missing adversarial images for source={source}, attack={attack}. "
                    f"Directory: {case_dir}. First missing file: {missing[0] if missing else 'unknown'}"
                )

            print("\n" + "-" * 80)
            print(f"Computing Swin-T feature/CAM metrics for source={source}, attack={attack}")
            print(f"Adversarial images: {case_dir}")

            metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
            for filename in tqdm(expected, desc=f"metrics/{source}/{attack}"):
                clean_path = clean_root / filename
                adv_path = case_dir / filename
                metrics = metric_row_for_pair(
                    model=model,
                    cam=cam,
                    clean_path=clean_path,
                    adv_path=adv_path,
                    device=device,
                    args=args,
                    cam_output_root=cam_output_root,
                    source=source,
                    attack=attack,
                    filename=filename,
                )
                for metric_name, value in metrics.items():
                    metric_values[metric_name].append(value)
                per_image_rows.append(
                    {
                        "source": source,
                        "source_label": display_source(source),
                        "attack": attack,
                        "attack_label": display_attack(attack),
                        "filename": filename,
                        **metrics,
                    }
                )

            summary = {
                "source": source,
                "source_label": display_source(source),
                "attack": attack,
                "attack_label": display_attack(attack),
                "target_model": SWIN_TARGET_MODEL,
                "samples": len(expected),
                "generation_seconds": load_generation_seconds(case_dir),
            }
            for metric_name in ALL_METRICS:
                summary[f"{metric_name}_mean"] = safe_mean(metric_values[metric_name])
                summary[f"{metric_name}_std"] = safe_std(metric_values[metric_name])
            summary_rows.append(summary)

            print(
                f"{display_attack(attack)}: "
                f"Feature L2={summary['feature_l2_mean']:.3f}, "
                f"KL={summary['feature_kl_mean']:.3f}, "
                f"Cos={summary['feature_cosine_mean']:.3f}, "
                f"CAM L2={summary['cam_l2_mean']:.3f}, "
                f"JS={summary['cam_js_mean']:.3f}, "
                f"Corr={summary['cam_corr_mean']:.3f}"
            )

    write_csv(per_image_csv, per_image_rows)
    write_csv(summary_csv, summary_rows)
    write_tex_outputs(tex_rows, full_tex, summary_rows, sources, attacks)
    write_analysis(analysis_file, args, sources, attacks, summary_rows)
    print("\nSaved:")
    print(f"  Per-image CSV: {per_image_csv}")
    print(f"  Summary CSV:   {summary_csv}")
    print(f"  LaTeX rows:    {tex_rows}")
    print(f"  LaTeX table:   {full_tex}")
    print(f"  Analysis:      {analysis_file}")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_best_values(rows: List[Dict[str, object]]) -> Dict[str, float]:
    best = {}
    for metric_name, direction in PRIMARY_METRICS:
        values = [float(row[f"{metric_name}_mean"]) for row in rows]
        best[metric_name] = max(values) if direction == "max" else min(values)
    return best


def format_metric(value: float, metric_name: str, best_values: Dict[str, float]) -> str:
    text = f"{value:.3f}"
    if metric_name in best_values and abs(value - best_values[metric_name]) <= 1e-9:
        return f"\\textbf{{{text}}}"
    return text


def summary_lookup(summary_rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], Dict[str, object]]:
    return {(str(row["source"]), str(row["attack"])): row for row in summary_rows}


def write_tex_outputs(
    tex_rows_path: Path,
    full_tex_path: Path,
    summary_rows: List[Dict[str, object]],
    sources: List[str],
    attacks: List[str],
) -> None:
    lookup = summary_lookup(summary_rows)
    best_values = metric_best_values(summary_rows)
    include_source_column = len(sources) > 1

    lines = []
    for source in sources:
        for attack in attacks:
            row = lookup[(source, attack)]
            cells = []
            if include_source_column:
                cells.append(display_source(source))
            cells.append(display_attack(attack))
            for metric_name, _ in PRIMARY_METRICS:
                cells.append(format_metric(float(row[f"{metric_name}_mean"]), metric_name, best_values))
            lines.append(" & ".join(cells) + r" \\")

    with open(tex_rows_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if include_source_column:
        tabular_spec = "l l *{3}{c} *{3}{c}"
        header = (
            "\\multirow{2}{*}{Source} & \\multirow{2}{*}{Attacks} "
            "& \\multicolumn{3}{c}{Feature map} & \\multicolumn{3}{c}{Grad-Cam}\\\\\n"
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
            "& & $L_2$(\\uparrow) & KL(\\uparrow) & Cosine(\\downarrow) "
            "& $L_2$(\\uparrow) & JS(\\uparrow) & Corr(\\downarrow) \\\\"
        )
    else:
        tabular_spec = "l *{3}{c} *{3}{c}"
        header = (
            "\\multirow{2}{*}{Attacks} "
            "& \\multicolumn{3}{c}{Feature map} & \\multicolumn{3}{c}{Grad-Cam}\\\\\n"
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
            "& $L_2$(\\uparrow) & KL(\\uparrow) & Cosine(\\downarrow) "
            "& $L_2$(\\uparrow) & JS(\\uparrow) & Corr(\\downarrow) \\\\"
        )

    full_table = rf"""\begin{{table*}}[!htb]
\centering
\caption{{Feature-space and attention metrics on Swin-T.}}
\label{{tab:other_metrics}}
\resizebox{{0.8\textwidth}}{{!}}{{
\begin{{tabular}}{{{tabular_spec}}}
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
    args: argparse.Namespace,
    sources: List[str],
    attacks: List[str],
    summary_rows: List[Dict[str, object]],
) -> None:
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("Swin-T feature-space and Grad-CAM metric analysis\n")
        f.write("=" * 56 + "\n")
        f.write(f"Target model: {SWIN_TARGET_MODEL}\n")
        f.write(f"Sources: {', '.join(sources)}\n")
        f.write(f"Attacks: {', '.join(attacks)}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"epsilon: {args.eps}\n")
        f.write(f"epoch: {args.epoch}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write("\nMetric interpretation:\n")
        f.write("Feature L2, Feature KL, CAM L2, and CAM JS: higher indicates larger clean-to-adversarial shift.\n")
        f.write("Feature cosine and CAM correlation: lower indicates weaker alignment with the clean representation or attention map.\n")
        f.write("\nSummary:\n")
        for row in summary_rows:
            f.write(
                f"{row['source_label']} | {row['attack_label']}: "
                f"Feature L2={row['feature_l2_mean']:.3f}, "
                f"KL={row['feature_kl_mean']:.3f}, "
                f"Cosine={row['feature_cosine_mean']:.3f}, "
                f"CAM L2={row['cam_l2_mean']:.3f}, "
                f"JS={row['cam_js_mean']:.3f}, "
                f"Corr={row['cam_corr_mean']:.3f}\n"
            )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adversarial examples and compute Swin-T feature-space/Grad-CAM metrics "
            "between clean and final adversarial images."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument("--sources", default=DEFAULT_SOURCE)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./feature_cam_metrics")
    parser.add_argument(
        "--batchsize",
        default=0,
        type=int,
        help="Generation batch size. Use 0 for automatic source/attack-specific values.",
    )
    parser.add_argument("--cam_batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--GPU_ID",
        default="0",
        help=(
            "Logical CUDA device id used by torch.cuda.set_device. "
            "When CUDA_VISIBLE_DEVICES is set externally, use --GPU_ID 0."
        ),
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
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--mesh_width", default=3, type=int)
    parser.add_argument("--mesh_height", default=3, type=int)
    parser.add_argument("--noise_scale", default=0.45, type=float)
    parser.add_argument("--num_warping", default=25, type=int)

    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--aug_smooth", action="store_true")
    parser.add_argument("--eigen_smooth", action="store_true")
    parser.add_argument("--save_cams", action="store_true")
    parser.add_argument("--cam_output_dir", default="./feature_cam_metrics/cam_visualizations")

    parser.add_argument("--per_image_csv", default="")
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--tex_rows", default="")
    parser.add_argument("--full_tex", default="")
    parser.add_argument("--analysis_file", default="")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    sources = parse_list(args.sources)
    attacks = parse_attacks(args.attacks)
    set_seed(args.seed)
    device = resolve_device(args)
    print(f"Using device: {device}")
    print(f"Fixed metric target model: {SWIN_TARGET_MODEL}")

    if args.mode in {"generate", "both"}:
        generate_cases(args, sources, attacks)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, sources, attacks, device)


if __name__ == "__main__":
    main()
