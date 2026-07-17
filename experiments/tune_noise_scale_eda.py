import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def parse_noise_scales(raw_values: str) -> List[float]:
    values = []
    for item in raw_values.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("At least one noise_scale value is required.")
    return values


def noise_dir_name(noise_scale: float) -> str:
    return f"noise{noise_scale:.2f}"


def extract_noise_scale(path: str) -> float:
    name = os.path.basename(path.rstrip(os.sep))
    return float(name.replace("noise", ""))


def source_model_names(model_arg: str) -> set:
    return {name.strip() for name in model_arg.split(",") if name.strip()}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the noise_scale parameter for EDA."
    )
    parser.add_argument("--mode", choices=["generate", "eval", "both"], default="both")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Compatibility option: evaluate existing noise* directories only.",
    )
    parser.add_argument("--attack", default="eda", choices=transferattack.attack_zoo.keys())
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument(
        "--output_dir",
        default="./tiaoyouresult/resnet18/eda_noise_scale",
        help="Directory that stores adversarial examples for each noise_scale.",
    )
    parser.add_argument("--batchsize", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--GPU_ID", default="0")
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)

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
    parser.add_argument("--num_warping", default=25, type=int)
    parser.add_argument(
        "--noise_scales",
        default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85",
        help="Comma-separated noise_scale values to evaluate.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output_dir before generating new adversarial examples.",
    )
    parser.add_argument(
        "--result_file",
        default="",
        help="Text result file. Defaults to output_dir/results_eval_eda_noise_scale.txt.",
    )
    parser.add_argument(
        "--analysis_file",
        default="",
        help="Analysis file. Defaults to output_dir/results_eval_eda_noise_scale_analysis.txt.",
    )
    return parser


def build_attacker(args: argparse.Namespace, noise_scale: float):
    model_name = args.model.split(",") if "," in args.model else args.model
    attack_class = transferattack.load_attack_class(args.attack)

    common_kwargs = dict(
        model_name=model_name,
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=args.random_start,
        norm=args.norm,
        loss=args.loss,
    )

    if args.attack.lower() == "eda":
        common_kwargs.update(
            mesh_width=args.mesh_width,
            mesh_height=args.mesh_height,
            noise_scale=noise_scale,
            num_warping=args.num_warping,
            seed=args.seed,
        )
    else:
        raise ValueError(
            "This tuning script is intended for the EDA attack. "
            "Please set --attack eda."
        )

    return attack_class(**common_kwargs)


def generate_for_noise_scales(args: argparse.Namespace, noise_scales: List[float]) -> None:
    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for noise_scale in noise_scales:
        set_seed(args.seed)
        run_dir = os.path.join(args.output_dir, noise_dir_name(noise_scale))
        os.makedirs(run_dir, exist_ok=True)

        print(f"\nGenerating adversarial images with noise_scale={noise_scale:.2f}")
        attacker = build_attacker(args, noise_scale)
        dataset = AdvDataset(
            input_dir=args.input_dir,
            output_dir=run_dir,
            targeted=args.targeted,
            eval=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for images, labels, filenames in tqdm(loader):
            perturbations = attacker(images, labels)
            save_images(run_dir, images + perturbations.cpu(), filenames)

        del attacker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_one_directory(
    args: argparse.Namespace, adv_dir: str, model_names: List[str]
) -> Tuple[Dict[str, float], float, float, float, float, float, float]:
    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=adv_dir,
        targeted=args.targeted,
        eval=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results: Dict[str, float] = {}
    for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
        print(f"Evaluating {os.path.basename(adv_dir)} on {model_name}")
        model = wrap_model(model.eval().cuda())
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(loader):
                if args.targeted:
                    labels = labels[1]
                images = images.cuda(non_blocking=True)
                outputs = model(images)
                predictions = outputs.argmax(dim=1).detach().cpu().numpy()
                labels_np = labels.numpy()

                if args.targeted:
                    correct += (predictions == labels_np).sum()
                else:
                    correct += (predictions != labels_np).sum()
                total += labels_np.shape[0]

        asr = 100.0 * correct / total
        results[model_name] = asr
        print(f"{model_name}: {asr:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_models = source_model_names(args.model)
    cnn_values_all = [
        results[name]
        for name in PAPER_CNN_MODELS
        if name in results
    ]
    vit_values_all = [
        results[name]
        for name in PAPER_VIT_MODELS
        if name in results
    ]
    all_values = [
        results[name]
        for name in model_names
        if name in results
    ]
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


def get_noise_dirs(output_dir: str) -> List[str]:
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    noise_dirs = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("noise") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not noise_dirs:
        raise FileNotFoundError(f"No noise* directories found in {output_dir}")
    return sorted(noise_dirs, key=extract_noise_scale)


def evaluate_noise_scales(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = args.result_file or os.path.join(
        args.output_dir, "results_eval_eda_noise_scale.txt"
    )
    analysis_file = args.analysis_file or os.path.join(
        args.output_dir, "results_eval_eda_noise_scale_analysis.txt"
    )

    model_names = PAPER_EVAL_MODELS
    noise_dirs = get_noise_dirs(args.output_dir)
    rows = []

    with open(result_file, "w", encoding="utf-8") as f:
        header = [
            "noise_scale",
            *model_names,
            "CNN Avg incl source",
            "ViT Avg incl source",
            "Overall Avg incl source",
            "CNN Avg excl source",
            "ViT Avg excl source",
            "Transfer Avg excl source",
        ]
        f.write(" | ".join(header) + "\n")
        f.write("-" * 120 + "\n")

        for adv_dir in noise_dirs:
            noise_scale = extract_noise_scale(adv_dir)
            set_seed(args.seed)
            (
                results,
                cnn_avg_all,
                vit_avg_all,
                overall_avg_all,
                cnn_avg,
                vit_avg,
                transfer_avg,
            ) = evaluate_one_directory(args, adv_dir, model_names)
            row = {
                "noise_scale": noise_scale,
                "results": results,
                "cnn_avg_all": cnn_avg_all,
                "vit_avg_all": vit_avg_all,
                "overall_avg_all": overall_avg_all,
                "cnn_avg": cnn_avg,
                "vit_avg": vit_avg,
                "transfer_avg": transfer_avg,
            }
            rows.append(row)

            values = [f"{noise_scale:.2f}"]
            values.extend(f"{results[name]:.2f}" for name in model_names)
            values.extend(
                [
                    f"{cnn_avg_all:.2f}",
                    f"{vit_avg_all:.2f}",
                    f"{overall_avg_all:.2f}",
                    f"{cnn_avg:.2f}",
                    f"{vit_avg:.2f}",
                    f"{transfer_avg:.2f}",
                ]
            )
            f.write(" | ".join(values) + "\n")
            f.flush()

    best_cnn_all = max(rows, key=lambda row: row["cnn_avg_all"])
    best_vit_all = max(rows, key=lambda row: row["vit_avg_all"])
    best_overall_all = max(rows, key=lambda row: row["overall_avg_all"])
    best_cnn = max(rows, key=lambda row: row["cnn_avg"])
    best_vit = max(rows, key=lambda row: row["vit_avg"])
    best_transfer = max(rows, key=lambda row: row["transfer_avg"])

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("EDA noise_scale tuning analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Attack: {args.attack}\n")
        f.write(f"Source model: {args.model}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"epsilon: {args.eps}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"epoch: {args.epoch}\n")
        f.write(f"mesh_width: {args.mesh_width}\n")
        f.write(f"mesh_height: {args.mesh_height}\n")
        f.write(f"num_warping: {args.num_warping}\n")
        f.write(f"seed: {args.seed}\n\n")
        f.write("CNN group: " + ", ".join(PAPER_CNN_MODELS) + "\n")
        f.write("ViT group: " + ", ".join(PAPER_VIT_MODELS) + "\n\n")
        f.write("Both source-included and source-excluded averages are reported.\n")
        f.write(
            "Source-excluded averages omit the source model if it appears in the "
            "evaluation list.\n\n"
        )

        f.write(
            "Best CNN Avg incl source: "
            f"noise_scale={best_cnn_all['noise_scale']:.2f}, "
            f"ASR={best_cnn_all['cnn_avg_all']:.2f}%\n"
        )
        f.write(
            "Best ViT Avg incl source: "
            f"noise_scale={best_vit_all['noise_scale']:.2f}, "
            f"ASR={best_vit_all['vit_avg_all']:.2f}%\n"
        )
        f.write(
            "Best Overall Avg incl source: "
            f"noise_scale={best_overall_all['noise_scale']:.2f}, "
            f"ASR={best_overall_all['overall_avg_all']:.2f}%\n\n"
        )

        f.write(
            "Best CNN Avg excl source: "
            f"noise_scale={best_cnn['noise_scale']:.2f}, "
            f"ASR={best_cnn['cnn_avg']:.2f}%\n"
        )
        f.write(
            "Best ViT Avg excl source: "
            f"noise_scale={best_vit['noise_scale']:.2f}, "
            f"ASR={best_vit['vit_avg']:.2f}%\n"
        )
        f.write(
            "Best Transfer Avg excl source: "
            f"noise_scale={best_transfer['noise_scale']:.2f}, "
            f"ASR={best_transfer['transfer_avg']:.2f}%\n\n"
        )

        f.write("All results sorted by Transfer Avg excl source:\n")
        for row in sorted(rows, key=lambda item: item["transfer_avg"], reverse=True):
            f.write(
                f"noise_scale={row['noise_scale']:.2f}: "
                f"CNN incl={row['cnn_avg_all']:.2f}%, "
                f"ViT incl={row['vit_avg_all']:.2f}%, "
                f"Overall incl={row['overall_avg_all']:.2f}%, "
                f"CNN excl={row['cnn_avg']:.2f}%, "
                f"ViT excl={row['vit_avg']:.2f}%, "
                f"Transfer excl={row['transfer_avg']:.2f}%\n"
            )

    print(f"\nResults saved to: {result_file}")
    print(f"Analysis saved to: {analysis_file}")
    print(
        "Best Transfer Avg excl source: "
        f"noise_scale={best_transfer['noise_scale']:.2f}, "
        f"ASR={best_transfer['transfer_avg']:.2f}%"
    )


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    if args.eval:
        args.mode = "eval"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    set_seed(args.seed)
    noise_scales = parse_noise_scales(args.noise_scales)

    if args.mode in ("generate", "both"):
        generate_for_noise_scales(args, noise_scales)

    if args.mode in ("eval", "both"):
        evaluate_noise_scales(args)


if __name__ == "__main__":
    main()
