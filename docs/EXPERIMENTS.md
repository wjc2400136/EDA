# Experiment Guide

[English](EXPERIMENTS.md) | [简体中文](EXPERIMENTS.zh-CN.md)

Run commands from the package root after `pip install -e .`.

## Common Configuration

| Parameter | Value |
|---|---:|
| Norm | Linf |
| Main budget | 16/255 |
| Iterations | 10 |
| Step size | epsilon / 10 |
| Momentum | 1.0 |
| EDA mesh | 3 x 3 |
| EDA deformation scale | 0.45 |
| Transformed views | 25 |
| Main seed | 42 |

CNN averages exclude a source-identical checkpoint. ViT averages include all
ten targets when the source is a CNN.

## Main CNN-source Tables

Run each method separately:

```bash
for ATTACK in l2t bsr decowa ops sid eda; do
  python experiments/run_multi_source.py \
    --mode both --attack "$ATTACK" \
    --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
    --input_dir ./data --output_root "./outputs/main/$ATTACK" \
    --num_warping 25 --noise_scale 0.45 --GPU_ID 0
done
```

The runner maps 25 samples to each method's own argument. EDA's deformation
scale is not passed to DeCoWA; DeCoWA retains its original setting.

## Multi-budget and Targeted Transfer

For a clean full reproduction, run all six attacks into a new output directory:

```bash
python experiments/run_budget_targeted_eda.py \
  --mode untargeted_both \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --epsilons 4/255,8/255,12/255,16/255 \
  --input_dir ./data --output_dir ./outputs/budget_targeted_full \
  --num_warping 25 --GPU_ID 0

python experiments/run_budget_targeted_eda.py \
  --mode targeted_both \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --epsilons 4/255,8/255,12/255,16/255 \
  --input_dir ./data --output_dir ./outputs/budget_targeted_full \
  --num_warping 25 --GPU_ID 0
```

After both generation passes finish, validate the complete inventory and run
evaluation only:

```bash
python experiments/run_budget_targeted_eda.py \
  --mode audit --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full

python experiments/run_budget_targeted_eda.py \
  --mode full_eval --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full --GPU_ID 0
```

`full_eval` requires all six attacks and all 192 generation cases. It does not
generate images. It writes `budget_targeted_results.csv`, the table-body rows,
the complete LaTeX table, an analysis file, and a generation-audit CSV. Use
`--resume_full_eval` only to resume an interrupted full evaluation. Without
this option, existing adversarial images are reused but cached ASR values are
discarded so that all six attacks are evaluated under one consistent protocol.

At `16/255`, untargeted values must match the main experiment when every
generation setting is identical. A mismatch indicates mixed provenance or a
configuration difference, not a different averaging convention.

## Random-seed Stability

```bash
python experiments/run_seed_stability_eda.py \
  --mode both --seeds 0,1,2,3,4 \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --input_dir ./data --output_dir ./outputs/seed_stability \
  --num_warping 25 --GPU_ID 0
```

Every seed regenerates the adversarial examples. Results are reported as mean
plus sample standard deviation.

## Ablations

```bash
python experiments/run_parts_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_parts --GPU_ID 0
python experiments/run_layout_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_layout --GPU_ID 0
python experiments/run_expansion_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_expansion --GPU_ID 0
python experiments/run_appearance_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_appearance --GPU_ID 0
python experiments/tune_noise_scale_eda.py --mode both --input_dir ./data --output_dir ./outputs/noise_scale --GPU_ID 0
```

Use `python SCRIPT --help` for script-specific plot and output options.

## Robustness and Perceptual Quality

```bash
python experiments/run_modern_robustbench_eval.py --mode both --input_dir ./data --output_dir ./outputs/robustbench --model_dir ./checkpoints/robustbench --allow_download --GPU_ID 0
python experiments/run_perceptual_quality_eda.py --mode both --input_dir ./data --output_dir ./outputs/perceptual --GPU_ID 0 --allow_missing_optional_metrics
```

Perceptual metrics compare clean images with final adversarial images, not with
intermediate TPS-warped views.

## Combination with Gradient-Based Attacks

```bash
python experiments/run_gradient_eda_combination.py \
  --mode both --sources resnet18 \
  --methods vmifgsm,emifgsm,pgn,mef,gaa \
  --variants base,eda --input_dir ./data \
  --output_dir ./outputs/gradient_combination --GPU_ID 0
```

The runner compares each gradient-based attack with and without the EDA input
transformation under the same source model, dataset, perturbation budget, and
evaluation targets.

## Larger and Shifted Datasets

```bash
python experiments/run_imagenet_val_generalization_eda.py --mode both --input_dir ./data_imagenet_val --output_dir ./outputs/imagenet_val --GPU_ID 0
python experiments/run_imagenetv2_generalization_eda.py --mode both --input_dir ./data_imagenetv2_matched --output_dir ./outputs/imagenetv2 --GPU_ID 0
```

Use separate output directories per dataset and never merge partial CSV files
from different seeds or script revisions.
