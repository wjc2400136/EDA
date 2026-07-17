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
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont
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
DEFAULT_SAMPLE_ID = "01f824264783f58d"

PADDING_CASES = [
    {
        "case_id": "zero",
        "display_name": "Zero",
        "padding_mode": "zero",
        "is_default": False,
    },
    {
        "case_id": "gaussian",
        "display_name": "Gaussian noise",
        "padding_mode": "gaussian",
        "is_default": False,
    },
    {
        "case_id": "border",
        "display_name": "Border",
        "padding_mode": "border",
        "is_default": False,
    },
    {
        "case_id": "reflect",
        "display_name": "Reflection",
        "padding_mode": "reflect",
        "is_default": True,
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

OKABE_ITO_SKY_BLUE = "#56B4E9"
OKABE_ITO_ORANGE = "#E69F00"
OKABE_ITO_GREEN = "#009E73"
COLORBREWER_REFLECTION_RED = "#CB181D"
BAR_COLORS = [
    OKABE_ITO_SKY_BLUE,
    OKABE_ITO_ORANGE,
    OKABE_ITO_GREEN,
    COLORBREWER_REFLECTION_RED,
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


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in str(model_arg).split(",") if name.strip()}


def case_dir_name(case: Dict[str, object]) -> str:
    return str(case["case_id"])


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


def apply_padding_mode(
    x: torch.Tensor,
    pad_h: int,
    pad_w: int,
    padding_mode: str,
    gaussian_std: float,
) -> torch.Tensor:
    if padding_mode == "reflect":
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    if padding_mode == "zero":
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
    if padding_mode == "border":
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    if padding_mode == "gaussian":
        padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        noise = torch.randn_like(padded) * gaussian_std
        mask = torch.zeros_like(padded)
        if pad_h > 0:
            mask[:, :, :pad_h, :] = 1.0
            mask[:, :, -pad_h:, :] = 1.0
        if pad_w > 0:
            mask[:, :, :, :pad_w] = 1.0
            mask[:, :, :, -pad_w:] = 1.0
        return padded + noise * mask
    raise ValueError(f"Unsupported padding mode: {padding_mode}")


def patch_expansion_mode(attacker, padding_mode: str, gaussian_std: float) -> None:
    def elastic_warp(self, x):
        batch_size, _, height, width = x.size()
        source_points, mesh_width, mesh_height = self._select_layout()
        source_points = source_points.to(device=x.device, dtype=x.dtype)
        target_points = source_points + self._sample_offsets(
            source_points,
            mesh_width,
            mesh_height,
        )

        pad_h, pad_w = self._padding_amounts(height, width)
        padded = apply_padding_mode(
            x,
            pad_h,
            pad_w,
            padding_mode=padding_mode,
            gaussian_std=gaussian_std,
        )
        canvas_h = height + 2 * pad_h
        canvas_w = width + 2 * pad_w

        source_pixel = self._to_pixel_coords(source_points, height, width)
        target_pixel = self._to_pixel_coords(target_points, height, width)
        pad_vector = source_points.new_tensor([pad_w, pad_h])

        source_canvas = self._to_normalized_coords(
            source_pixel + pad_vector,
            canvas_h,
            canvas_w,
        )
        target_canvas = self._to_normalized_coords(
            target_pixel + pad_vector,
            canvas_h,
            canvas_w,
        )

        tps = self._get_tps_grid(canvas_h, canvas_w, x.dtype)
        sampling_grid = tps(source_canvas[None, ...], target_canvas[None, ...])
        sampling_grid = sampling_grid.repeat(batch_size, 1, 1, 1)

        warped_canvas = F.grid_sample(
            padded,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped_canvas[:, :, pad_h : pad_h + height, pad_w : pad_w + width]

    attacker.elastic_warp = types.MethodType(elastic_warp, attacker)
    attacker.expansion_padding_mode = padding_mode
    attacker.expansion_gaussian_std = gaussian_std


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
        use_dual_grid=True,
        move_edge=True,
        use_appearance=True,
    )
    patch_expansion_mode(
        attacker,
        padding_mode=str(case["padding_mode"]),
        gaussian_std=args.gaussian_padding_std,
    )
    return attacker


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    result_csv = Path(args.result_csv) if args.result_csv else output_dir / "expansion_results.csv"
    analysis_file = (
        Path(args.analysis_file)
        if args.analysis_file
        else output_dir / "expansion_results_analysis.txt"
    )
    figure_prefix = (
        Path(args.figure_prefix)
        if args.figure_prefix
        else output_dir / "figures" / "expansion_ablation_combined"
    )
    return result_csv, analysis_file, figure_prefix


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
        "gaussian_padding_std": args.gaussian_padding_std,
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


def evaluate_one_case(
    args: argparse.Namespace,
    case: Dict[str, object],
    device: torch.device,
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
        model = wrap_model(model.eval().to(device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        asr = eval_asr(model, loader, args.targeted, device)
        results[model_name] = asr
        print(f"{case['case_id']} -> {model_name}: {asr:.2f}%")

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

    row: Dict[str, object] = {
        "case_id": case["case_id"],
        "display_name": case["display_name"],
        "padding_mode": case["padding_mode"],
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
        "display_name",
        "padding_mode",
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


def as_float(row: Dict[str, object], key: str) -> float:
    value = row[key]
    if value == "" or value is None:
        return float("nan")
    return float(value)


def write_analysis(analysis_file: Path, rows: List[Dict[str, object]]) -> None:
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("EDA expansion-mode ablation analysis\n")
        f.write("=" * 44 + "\n")
        f.write("CNN averages exclude source-identical targets whenever applicable.\n\n")
        for row in rows:
            f.write(
                f"{row['display_name']}: "
                f"CNN excl={as_float(row, 'cnn_avg_excl_source'):.2f}%, "
                f"ViT={as_float(row, 'vit_avg_excl_source'):.2f}%, "
                f"Transfer={as_float(row, 'transfer_avg_excl_source'):.2f}%\n"
            )


def evaluate_cases(args: argparse.Namespace, cases: List[Dict[str, object]]) -> None:
    result_csv, analysis_file, _ = output_paths(args)
    rows: List[Dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for case in cases:
        print("\n" + "=" * 80)
        print(f"Evaluating case {case['case_id']}: {case['display_name']}")
        rows.append(evaluate_one_case(args, case, device))
        write_result_csv(result_csv, rows)
        write_analysis(analysis_file, rows)

    write_result_csv(result_csv, rows)
    write_analysis(analysis_file, rows)


def load_result_rows(result_csv: Path) -> List[Dict[str, object]]:
    if not result_csv.is_file():
        raise FileNotFoundError(f"Missing result CSV: {result_csv}")
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def configure_plot_font(args: argparse.Namespace) -> str:
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
    return font_family


def load_pil_font(args: argparse.Namespace, size: int) -> ImageFont.ImageFont:
    candidates = []
    if args.font_path:
        candidates.append(args.font_path)
    candidates.extend(
        [
            "times.ttf",
            "timesbd.ttf",
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = torch.clamp(tensor.detach().cpu().squeeze(0), 0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def add_original_border(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    tensor = torch.clamp(tensor.clone(), 0.0, 1.0)
    _, _, height, width = tensor.shape
    if pad_h <= 0 or pad_w <= 0:
        return tensor

    top = pad_h
    bottom = height - pad_h - 1
    left = pad_w
    right = width - pad_w - 1
    tensor[:, :, top, left : right + 1] = 1.0
    tensor[:, :, bottom, left : right + 1] = 1.0
    tensor[:, :, top : bottom + 1, left] = 1.0
    tensor[:, :, top : bottom + 1, right] = 1.0
    return tensor


def load_sample_image(input_dir: Path, sample_id: str) -> Tuple[Path, Image.Image]:
    image_dir = input_dir / "images"
    candidates = sorted(image_dir.glob(f"{sample_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"Could not find sample image {sample_id} under {image_dir}")
    path = candidates[0]
    return path, Image.open(path).convert("RGB")


def create_padding_visual(args: argparse.Namespace, figure_prefix: Path) -> Path:
    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    original_path, original = load_sample_image(input_dir, args.sample_id)
    resized = original.resize((args.visual_size, args.visual_size), Image.BICUBIC)
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    height, width = args.visual_size, args.visual_size
    visual_noise_scale = args.visual_noise_scale
    max_offset_norm = 0.5 * visual_noise_scale
    pad_w = int(torch.ceil(torch.tensor(max_offset_norm * (width - 1) / 2)).item())
    pad_h = int(torch.ceil(torch.tensor(max_offset_norm * (height - 1) / 2)).item())
    pad_w = max(1, pad_w)
    pad_h = max(1, pad_h)
    print(
        "Padding visualization: "
        f"visual_noise_scale={visual_noise_scale:g}, pad_h={pad_h}, pad_w={pad_w}. "
        f"Attack generation still uses noise_scale={args.noise_scale:g}."
    )

    visual_dir = figure_prefix.parent / "padding_visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    padded_images = []
    for case in PADDING_CASES:
        set_seed(args.seed)
        padded = apply_padding_mode(
            tensor,
            pad_h,
            pad_w,
            padding_mode=str(case["padding_mode"]),
            gaussian_std=args.visual_gaussian_std,
        )
        bordered = add_original_border(padded, pad_h, pad_w)
        image = tensor_to_pil(bordered)
        image.save(visual_dir / f"{args.sample_id}_{case['case_id']}.png")
        padded_images.append(image)

    title_font = load_pil_font(args, 34)
    label_font = load_pil_font(args, 42)

    margin = 10
    image_spacing = 44
    text_height = 40
    orig_w, orig_h = resized.size
    padded_sizes = [image.size for image in padded_images]
    col2_width = max(padded_sizes[0][0], padded_sizes[1][0])
    col3_width = max(padded_sizes[2][0], padded_sizes[3][0])
    col2_height = padded_sizes[0][1] + image_spacing + padded_sizes[1][1]
    col3_height = padded_sizes[2][1] + image_spacing + padded_sizes[3][1]
    max_height = max(orig_h, col2_height, col3_height)
    canvas_width = orig_w + margin + col2_width + margin + col3_width + margin
    canvas_height = max_height + margin * 2 + text_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    col1_x = margin
    col2_x = col1_x + orig_w + margin
    col3_x = col2_x + col2_width + margin
    col2_y1 = margin + (max_height - col2_height) // 2
    col2_y2 = col2_y1 + padded_sizes[0][1] + image_spacing
    col3_y1 = margin + (max_height - col3_height) // 2
    col3_y2 = col3_y1 + padded_sizes[2][1] + image_spacing
    if args.original_align == "top":
        orig_y = margin
    elif args.original_align == "bottom":
        orig_y = margin + max_height - orig_h
    else:
        orig_y = margin + (max_height - orig_h) // 2

    canvas.paste(resized, (col1_x, orig_y))
    draw.text(
        (col1_x + orig_w // 2, orig_y + orig_h + 10),
        "Original Image",
        fill="black",
        font=title_font,
        anchor="mt",
    )

    positions = [
        (col2_x, col2_y1, col2_width, "(a)"),
        (col2_x, col2_y2, col2_width, "(b)"),
        (col3_x, col3_y1, col3_width, "(c)"),
        (col3_x, col3_y2, col3_width, "(d)"),
    ]
    for image, (x, y, column_width, label) in zip(padded_images, positions):
        canvas.paste(image, (x, y))
        draw.text(
            (x + column_width // 2, y + image.size[1] + 4),
            label,
            fill="black",
            font=label_font,
            anchor="mt",
        )

    canvas = crop_white_border(canvas, padding=8)
    comparison_path = visual_dir / f"{args.sample_id}_padding_comparison.png"
    canvas.save(comparison_path, dpi=(300, 300))
    canvas.save(comparison_path.with_suffix(".pdf"), dpi=(300, 300), format="PDF")
    print(f"Saved padding comparison panel: {comparison_path}")
    print(f"Sample source image: {original_path}")
    return comparison_path


def crop_white_border(image: Image.Image, padding: int = 8) -> Image.Image:
    background = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.size[0], right + padding)
    bottom = min(image.size[1], bottom + padding)
    return image.crop((left, top, right, bottom))


def sorted_rows_by_cases(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case = {row["case_id"]: row for row in rows}
    missing = [case["case_id"] for case in PADDING_CASES if case["case_id"] not in by_case]
    if missing:
        raise ValueError(f"Missing result rows for cases: {', '.join(missing)}")
    return [by_case[case["case_id"]] for case in PADDING_CASES]


def align_image_axis_to_reference(
    image_axis,
    reference_axis,
    image: Image.Image,
    horizontal_shift: float = 0.0,
) -> None:
    """Resize the image axis and center it in the reference tight bbox height."""
    image_width, image_height = image.size
    image_ratio = image_width / image_height
    figure = image_axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    fig_width, fig_height = figure.get_size_inches()
    image_position = image_axis.get_position()
    reference_position = reference_axis.get_position()
    reference_tight = reference_axis.get_tightbbox(renderer).transformed(
        figure.transFigure.inverted()
    )

    reference_bottom = max(0.0, reference_tight.y0)
    reference_top = min(1.0, reference_position.y1)
    available_height = reference_top - reference_bottom
    target_height = available_height
    target_width = target_height * image_ratio * (fig_height / fig_width)
    if target_width > image_position.width:
        target_width = image_position.width
        target_height = target_width * (fig_width / fig_height) / image_ratio

    x0 = image_position.x1 - target_width + horizontal_shift
    y0 = reference_bottom + (available_height - target_height) / 2.0
    image_axis.set_position([x0, y0, target_width, target_height])


def plot_results(args: argparse.Namespace) -> None:
    result_csv, _, figure_prefix = output_paths(args)
    rows = sorted_rows_by_cases(load_result_rows(result_csv))
    configure_plot_font(args)
    comparison_path = create_padding_visual(args, figure_prefix)
    comparison_image = Image.open(comparison_path).convert("RGB")

    labels = ["CNNs", "ViTs"]
    x = np.arange(len(labels))
    width = 0.2

    fig = plt.figure(figsize=(18, 6.0))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    ax_left.imshow(comparison_image, aspect="equal")
    ax_left.axis("off")

    for idx, (case, row) in enumerate(zip(PADDING_CASES, rows)):
        values = [
            as_float(row, "cnn_avg_excl_source"),
            as_float(row, "vit_avg_excl_source"),
        ]
        positions = x + (idx - 1.5) * width
        bars = ax_bar.bar(
            positions,
            values,
            width,
            label=str(case["display_name"]),
            color=BAR_COLORS[idx],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            is_reflection = bool(case["is_default"])
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.12,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color=COLORBREWER_REFLECTION_RED if is_reflection else "black",
                fontweight="bold" if is_reflection else "normal",
            )

    ax_bar.set_ylabel("Average Attack Success Rate (%)", fontsize=18)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=18)
    ax_bar.set_ylim(args.plot_ymin, args.plot_ymax)
    ax_bar.tick_params(axis="y", labelsize=15)
    ax_bar.legend(
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=14,
    )
    align_image_axis_to_reference(
        ax_left,
        ax_bar,
        comparison_image,
        horizontal_shift=args.left_panel_shift,
    )

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg", "eps"]:
        kwargs = {"bbox_inches": "tight", "transparent": False, "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(str(figure_prefix.with_suffix(f".{ext}")), **kwargs)
    plt.close(fig)
    print(f"Saved combined figure prefix: {figure_prefix}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, evaluate, and plot the EDA expansion-mode ablation "
            "for Zero, Gaussian noise, Border, and Reflection padding."
        )
    )
    parser.add_argument("--mode", choices=["generate", "eval", "plot", "both"], default="both")
    parser.add_argument("--attack", default="eda", choices=["eda"])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./expansion_ablation/resnet18")
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
    parser.add_argument(
        "--gaussian_padding_std",
        default=0.05,
        type=float,
        help="Gaussian std used in the attack-time padding region.",
    )
    parser.add_argument(
        "--visual_gaussian_std",
        default=0.25,
        type=float,
        help="Gaussian std used only for the one-image padding visualization.",
    )

    parser.add_argument("--result_csv", default="")
    parser.add_argument("--analysis_file", default="")
    parser.add_argument("--figure_prefix", default="")
    parser.add_argument("--sample_id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--visual_size", default=224, type=int)
    parser.add_argument(
        "--original_align",
        choices=["top", "center", "bottom"],
        default="center",
        help=(
            "Vertical placement of the Original Image in the left visual panel. "
            "Bottom alignment removes the empty area below the original image."
        ),
    )
    parser.add_argument(
        "--visual_noise_scale",
        default=0.75,
        type=float,
        help=(
            "Noise scale used only for the left padding visualization. "
            "The attack still uses --noise_scale."
        ),
    )
    parser.add_argument("--plot_ymin", default=75.0, type=float)
    parser.add_argument("--plot_ymax", default=100.0, type=float)
    parser.add_argument(
        "--left_panel_shift",
        default=0.01,
        type=float,
        help=(
            "Horizontal shift of the left visual panel in figure coordinates. "
            "Increase it to move the left panel closer to the bar chart."
        ),
    )
    parser.add_argument("--font_family", default="Times New Roman")
    parser.add_argument(
        "--font_path",
        default="",
        help="Optional path to a .ttf/.otf Times New Roman font file.",
    )
    return parser


def main() -> None:
    global GLOBAL_SEED
    parser = get_parser()
    args = parser.parse_args()
    GLOBAL_SEED = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    set_seed(args.seed)

    cases = PADDING_CASES
    if args.mode in {"generate", "both"}:
        generate_cases(args, cases)
    if args.mode in {"eval", "both"}:
        evaluate_cases(args, cases)
    if args.mode in {"plot", "both"}:
        plot_results(args)


if __name__ == "__main__":
    main()
