import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an ImageNet-1K validation dataset directory compatible with "
            "the TransferAttack AdvDataset format and verify the generated labels."
        )
    )
    parser.add_argument("--mode", choices=["generate", "verify", "both"], default="both")
    parser.add_argument("--image_root", default="./imagenet-val")
    parser.add_argument("--class_index", default="./imagenet_class_index.json")
    parser.add_argument("--devkit_dir", default="./ILSVRC2012_devkit_t12")
    parser.add_argument("--output_dir", default="./data_imagenet_val")
    parser.add_argument(
        "--link_mode",
        choices=["hardlink", "symlink", "copy", "none"],
        default="hardlink",
        help=(
            "How to populate output_dir/images. 'none' only writes labels.csv and "
            "keeps filenames relative to image_root; this is not recommended for "
            "the existing attack scripts."
        ),
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


def read_class_index(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        class_index = json.load(f)
    wnid_to_index: Dict[str, int] = {}
    index_to_wnid: Dict[int, str] = {}
    for index_text, value in class_index.items():
        index = int(index_text)
        wnid = value[0]
        if not 0 <= index <= 999:
            raise ValueError(f"Invalid class index {index} in {path}")
        wnid_to_index[wnid] = index
        index_to_wnid[index] = wnid
    if len(wnid_to_index) != 1000 or len(index_to_wnid) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes in {path}, got {len(wnid_to_index)}")
    return wnid_to_index, index_to_wnid


def image_files(directory: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def is_synset_layout(image_root: Path, wnid_to_index: Dict[str, int]) -> bool:
    subdirs = [path for path in image_root.iterdir() if path.is_dir()]
    if not subdirs:
        return False
    known = sum(1 for path in subdirs if path.name in wnid_to_index)
    return known >= 900


def validation_number(path: Path) -> int:
    # ILSVRC2012_val_00000001.JPEG -> 1
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse validation index from filename: {path.name}")
    return int(digits)


def read_devkit_ground_truth(devkit_dir: Path) -> List[int]:
    gt_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ImageNet validation ground truth file: {gt_path}")
    labels = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                labels.append(int(text) - 1)
    if len(labels) != 50000:
        raise ValueError(f"Expected 50000 validation labels in {gt_path}, got {len(labels)}")
    if min(labels) < 0 or max(labels) > 999:
        raise ValueError("Devkit labels are outside the expected 0-999 range after subtracting 1.")
    return labels


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
    candidate = f"{prefix}_{source.name}"
    counter = 1
    while candidate in used_names:
        candidate = f"{prefix}_{counter}_{source.name}"
        counter += 1
    used_names.add(candidate)
    return candidate


def build_rows_from_synset_layout(
    image_root: Path,
    wnid_to_index: Dict[str, int],
    target_seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    used_names = set()
    rng = random.Random(target_seed)
    for class_dir in sorted([path for path in image_root.iterdir() if path.is_dir()], key=lambda p: p.name):
        if class_dir.name not in wnid_to_index:
            continue
        label = wnid_to_index[class_dir.name]
        for source_path in image_files(class_dir):
            filename = unique_flat_name(source_path, used_names, class_dir.name)
            rows.append(
                {
                    "filename": filename,
                    "label": label,
                    "targeted_label": deterministic_target(label, rng),
                    "source_path": str(source_path),
                    "wnid": class_dir.name,
                }
            )
    return rows


def build_rows_from_flat_layout(
    image_root: Path,
    devkit_dir: Path,
    index_to_wnid: Dict[int, str],
    target_seed: int,
) -> List[Dict[str, object]]:
    files = sorted(image_files(image_root), key=validation_number)
    labels = read_devkit_ground_truth(devkit_dir)
    if len(files) != len(labels):
        raise ValueError(
            f"Flat validation layout has {len(files)} images, but devkit has {len(labels)} labels."
        )
    rng = random.Random(target_seed)
    rows: List[Dict[str, object]] = []
    used_names = set()
    for source_path, label in zip(files, labels):
        filename = unique_flat_name(source_path, used_names, index_to_wnid[label])
        rows.append(
            {
                "filename": filename,
                "label": label,
                "targeted_label": deterministic_target(label, rng),
                "source_path": str(source_path),
                "wnid": index_to_wnid[label],
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


def write_manifest(output_dir: Path, rows: Sequence[Dict[str, object]], layout: str) -> None:
    with (output_dir / "labels_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label", "targeted_label", "wnid", "source_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    counts = Counter(int(row["label"]) for row in rows)
    summary = {
        "layout": layout,
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
    class_index_path = Path(args.class_index)
    devkit_dir = Path(args.devkit_dir)
    output_dir = Path(args.output_dir)
    image_output_dir = output_dir / "images"

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not class_index_path.is_file():
        raise FileNotFoundError(f"ImageNet class index file does not exist: {class_index_path}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    wnid_to_index, index_to_wnid = read_class_index(class_index_path)
    if is_synset_layout(image_root, wnid_to_index):
        layout = "synset-folder"
        rows = build_rows_from_synset_layout(image_root, wnid_to_index, args.target_seed)
    else:
        layout = "flat-devkit"
        rows = build_rows_from_flat_layout(image_root, devkit_dir, index_to_wnid, args.target_seed)

    if len(rows) != 50000:
        raise ValueError(f"Expected 50000 ImageNet validation images, got {len(rows)}")

    for index, row in enumerate(rows, start=1):
        source_path = Path(str(row["source_path"]))
        destination = image_output_dir / str(row["filename"])
        link_or_copy(source_path, destination, args.link_mode)
        if index % 5000 == 0:
            print(f"Prepared {index}/{len(rows)} images...")

    write_labels_csv(output_dir, rows)
    write_manifest(output_dir, rows, layout)
    print(f"Generated dataset: {output_dir}")
    print(f"Images: {image_output_dir}")
    print(f"Labels: {output_dir / 'labels.csv'}")
    print(f"Layout detected: {layout}")


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
    if len(rows) != 50000:
        raise ValueError(f"Expected 50000 rows in labels.csv, got {len(rows)}")
    counts = Counter(int(row["label"]) for row in rows)
    if set(counts.keys()) != set(range(1000)):
        missing = sorted(set(range(1000)) - set(counts.keys()))
        extra = sorted(set(counts.keys()) - set(range(1000)))
        raise ValueError(f"Class coverage is invalid. Missing={missing[:10]}, extra={extra[:10]}")
    if min(counts.values()) != 50 or max(counts.values()) != 50:
        raise ValueError(
            f"Expected exactly 50 validation images per class, got min={min(counts.values())}, "
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
    print(f"  Dataset directory: {output_dir}")
    print(f"  labels.csv: {output_dir / 'labels.csv'}")
    print(f"  images directory: {output_dir / 'images'}")
    print(f"  Number of rows: {len(rows)}")
    print("  Number of classes: 1000")
    print("  Images per class: 50")
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

    class PreparedImageNetDataset(Dataset):
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
    print(f"  Dataset directory: {Path(args.output_dir)}")
    print(f"  Model: {args.clean_model}")
    print(f"  Device: {device}")
    print(f"  Samples: {len(selected_rows)}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Num workers: {args.num_workers}")
    print("  Purpose: verify that generated labels are aligned with PyTorch/timm indices.")

    model = load_model(args.clean_model).eval().to(device)
    dataset = PreparedImageNetDataset(Path(args.output_dir), selected_rows)
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
    status = "PASSED" if accuracy >= 30.0 else "FAILED"
    report = {
        "dataset_dir": str(Path(args.output_dir)),
        "model": args.clean_model,
        "device": str(device),
        "samples": total,
        "correct": correct,
        "accuracy_percent": accuracy,
        "status": status,
        "note": (
            "A normal torchvision ResNet-18 check on ImageNet validation should "
            "be clearly above chance level. A near-zero value usually indicates "
            "a label-index mismatch."
        ),
    }
    report_path = (
        Path(args.output_dir)
        / f"clean_accuracy_{args.clean_model}_{total if args.clean_samples != 0 else 'all'}.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"  Clean accuracy: {accuracy:.2f}% ({correct}/{total})"
    )
    print(
        f"  Status: {status}"
    )
    print(f"  Report saved to: {report_path}")
    if status == "FAILED":
        print(
            "  Warning: clean accuracy is too low. Check whether labels are 0-999 "
            "and aligned with the ImageNet class-index mapping."
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
