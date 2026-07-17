import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ImageNet-V2 matched-frequency data in the TransferAttack "
            "AdvDataset format and verify generated labels."
        )
    )
    parser.add_argument("--mode", choices=["generate", "verify", "both"], default="both")
    parser.add_argument(
        "--image_root",
        default="./imagenetv2-matched-frequency/imagenetv2-matched-frequency-format-val",
        help="Directory containing class folders 0, 1, ..., 999.",
    )
    parser.add_argument("--output_dir", default="./data_imagenetv2_matched")
    parser.add_argument(
        "--link_mode",
        choices=["hardlink", "symlink", "copy", "none"],
        default="hardlink",
        help="How to populate output_dir/images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target_seed", type=int, default=12345)
    parser.add_argument("--clean_check", action="store_true")
    parser.add_argument("--clean_model", default="resnet18")
    parser.add_argument(
        "--clean_samples",
        type=int,
        default=1000,
        help="Number of randomly selected samples for clean-accuracy verification. Use 0 for all images.",
    )
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_files(directory: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def deterministic_target(label: int, rng: random.Random) -> int:
    target = rng.randrange(1000)
    if target == label:
        target = (target + 1) % 1000
    return target


def unique_flat_name(source: Path, used_names: set, prefix: str) -> str:
    candidate = source.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    candidate = f"class{prefix}_{source.name}"
    counter = 1
    while candidate in used_names:
        candidate = f"class{prefix}_{counter}_{source.name}"
        counter += 1
    used_names.add(candidate)
    return candidate


def build_rows(image_root: Path, target_seed: int) -> List[Dict[str, object]]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"ImageNet-V2 image root does not exist: {image_root}")

    rows: List[Dict[str, object]] = []
    used_names = set()
    rng = random.Random(target_seed)

    class_dirs = []
    ignored_dirs = []
    for path in sorted([item for item in image_root.iterdir() if item.is_dir()], key=lambda p: p.name):
        if path.name.isdigit() and 0 <= int(path.name) <= 999:
            class_dirs.append(path)
        else:
            ignored_dirs.append(path.name)

    if len(class_dirs) != 1000:
        raise ValueError(
            f"Expected exactly 1000 class folders named 0..999 in {image_root}, "
            f"got {len(class_dirs)}. Ignored dirs: {ignored_dirs[:10]}"
        )

    for class_dir in class_dirs:
        label = int(class_dir.name)
        files = image_files(class_dir)
        for source_path in files:
            filename = unique_flat_name(source_path, used_names, class_dir.name)
            rows.append(
                {
                    "filename": filename,
                    "label": label,
                    "targeted_label": deterministic_target(label, rng),
                    "source_path": str(source_path),
                    "class_dir": class_dir.name,
                }
            )
    return rows


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        try:
            os.symlink(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    raise ValueError(f"Unsupported link mode: {mode}")


def write_labels_csv(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "targeted_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "label": row["label"],
                    "targeted_label": row["targeted_label"],
                }
            )


def write_manifest(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    with (output_dir / "labels_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label", "targeted_label", "class_dir", "source_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    counts = Counter(int(row["label"]) for row in rows)
    summary = {
        "dataset": "ImageNet-V2 matched-frequency",
        "num_images": len(rows),
        "num_classes": len(counts),
        "min_label": min(counts) if counts else None,
        "max_label": max(counts) if counts else None,
        "min_images_per_class": min(counts.values()) if counts else None,
        "max_images_per_class": max(counts.values()) if counts else None,
    }
    with (output_dir / "label_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def generate_dataset(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    image_output_dir = output_dir / "images"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(image_root, args.target_seed)
    if len(rows) != 10000:
        raise ValueError(f"Expected 10000 ImageNet-V2 matched-frequency images, got {len(rows)}")

    for index, row in enumerate(rows, start=1):
        source_path = Path(str(row["source_path"]))
        destination = image_output_dir / str(row["filename"])
        link_or_copy(source_path, destination, args.link_mode)
        if index % 1000 == 0:
            print(f"Prepared {index}/{len(rows)} images...")

    write_labels_csv(output_dir, rows)
    write_manifest(output_dir, rows)
    print(f"Generated dataset: {output_dir}")
    print(f"Images: {image_output_dir}")
    print(f"Labels: {output_dir / 'labels.csv'}")


def read_labels_csv(output_dir: Path) -> List[Dict[str, object]]:
    labels_path = output_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    rows: List[Dict[str, object]] = []
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "filename": row["filename"],
                    "label": int(row["label"]),
                    "targeted_label": int(row["targeted_label"]),
                }
            )
    return rows


def verify_structure(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    if len(rows) != 10000:
        raise ValueError(f"Expected 10000 rows in labels.csv, got {len(rows)}")
    counts = Counter(int(row["label"]) for row in rows)
    if set(counts.keys()) != set(range(1000)):
        missing = sorted(set(range(1000)) - set(counts.keys()))
        extra = sorted(set(counts.keys()) - set(range(1000)))
        raise ValueError(f"Class coverage is invalid. Missing={missing[:10]}, extra={extra[:10]}")
    if min(counts.values()) != 10 or max(counts.values()) != 10:
        raise ValueError(
            f"Expected exactly 10 ImageNet-V2 images per class, got min={min(counts.values())}, "
            f"max={max(counts.values())}"
        )
    missing_images = [
        str(output_dir / "images" / str(row["filename"]))
        for row in rows
        if not (output_dir / "images" / str(row["filename"])).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing prepared images, first examples: {missing_images[:5]}")
    invalid_targets = [
        row for row in rows if int(row["targeted_label"]) == int(row["label"])
    ]
    if invalid_targets:
        raise ValueError(f"targeted_label equals label for {len(invalid_targets)} rows.")

    print("\n[Label verification]")
    print(f"  Dataset: ImageNet-V2 matched-frequency")
    print(f"  Dataset directory: {output_dir}")
    print(f"  labels.csv: {output_dir / 'labels.csv'}")
    print(f"  images directory: {output_dir / 'images'}")
    print(f"  Number of rows: {len(rows)}")
    print("  Number of classes: 1000")
    print("  Images per class: 10")
    print("  Label index range: 0-999")
    print("  Target labels: every targeted_label is different from label")
    print("  Status: PASSED")
    print("  First five rows:")
    for row in list(rows)[:5]:
        print(
            f"    {row['filename']}, label={row['label']}, "
            f"targeted_label={row['targeted_label']}"
        )


def clean_accuracy_check(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms.functional import to_tensor

    class PreparedImageNetV2Dataset(Dataset):
        def __init__(self, output_dir: Path, selected_rows: Sequence[Dict[str, object]]):
            self.image_dir = output_dir / "images"
            self.rows = list(selected_rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            path = self.image_dir / str(row["filename"])
            image = Image.open(path).resize((224, 224)).convert("RGB")
            image = to_tensor(image)
            label = int(row["label"])
            return image, label

    class TransferAttackPreprocess(nn.Module):
        def __init__(self, resize: int, mean: Sequence[float], std: Sequence[float]):
            super().__init__()
            self.resize = resize
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x):
            if x.shape[-1] != self.resize or x.shape[-2] != self.resize:
                x = F.interpolate(
                    x,
                    size=(self.resize, self.resize),
                    mode="bilinear",
                    align_corners=False,
                )
            return (x - self.mean) / self.std

    def load_model(model_name: str):
        if model_name in models.__dict__:
            model = models.__dict__[model_name](weights="DEFAULT")
        elif model_name in timm.list_models():
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f"Unsupported clean-check model: {model_name}")

        if hasattr(model, "default_cfg"):
            mean = model.default_cfg["mean"]
            std = model.default_cfg["std"]
            resize = 224
        elif "Inception" in model.__class__.__name__:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            resize = 224
        return nn.Sequential(TransferAttackPreprocess(resize, mean, std), model)

    selected_rows = list(rows)
    if args.clean_samples > 0 and args.clean_samples < len(selected_rows):
        rng = random.Random(args.seed)
        selected_rows = rng.sample(selected_rows, args.clean_samples)

    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print("\n[Clean accuracy verification]")
    print(f"  Dataset: ImageNet-V2 matched-frequency")
    print(f"  Dataset directory: {Path(args.output_dir)}")
    print(f"  Model: {args.clean_model}")
    print(f"  Device: {device}")
    print(f"  Samples: {len(selected_rows)}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Num workers: {args.num_workers}")
    print("  Purpose: verify that generated labels are aligned with PyTorch/timm indices.")

    model = load_model(args.clean_model).eval().to(device)
    dataset = PreparedImageNetV2Dataset(Path(args.output_dir), selected_rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    accuracy = 100.0 * correct / max(total, 1)
    status = "PASSED" if accuracy >= 20.0 else "FAILED"
    report = {
        "dataset": "ImageNet-V2 matched-frequency",
        "dataset_dir": str(Path(args.output_dir)),
        "model": args.clean_model,
        "device": str(device),
        "samples": total,
        "correct": correct,
        "accuracy_percent": accuracy,
        "status": status,
        "note": (
            "ImageNet-V2 clean accuracy is expected to be lower than ImageNet-1K "
            "validation accuracy. A near-zero value usually indicates a label-index mismatch."
        ),
    }
    report_path = (
        Path(args.output_dir)
        / f"clean_accuracy_{args.clean_model}_{total if args.clean_samples != 0 else 'all'}.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Clean accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Status: {status}")
    print(f"  Report saved to: {report_path}")
    if status == "FAILED":
        print(
            "  Warning: clean accuracy is too low. Check whether labels are 0-999 "
            "and aligned with the ImageNet-V2 folder names."
        )
    else:
        print("  Interpretation: label mapping is consistent with PyTorch/timm models.")


def verify_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = read_labels_csv(output_dir)
    verify_structure(output_dir, rows)
    if args.clean_check:
        clean_accuracy_check(args, rows)


def main() -> None:
    args = parse_args()
    if args.mode in {"generate", "both"}:
        generate_dataset(args)
    if args.mode in {"verify", "both"}:
        verify_dataset(args)


if __name__ == "__main__":
    main()
