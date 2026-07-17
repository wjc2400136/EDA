# Enhanced Deformation Attack: Reproducibility Package

[English](README.md) | [简体中文](README.zh-CN.md)

This standalone package accompanies **Boosting Cross-Model Adversarial
Transferability by Enhanced Deformation Attack**. It contains EDA, the five
compared transfer attacks, and scripts for the main, ablation, robustness,
generalization, perceptual-quality, and VLM experiments. It is an isolated copy;
using it does not modify files outside this directory.

For a minimal end-to-end workflow, start with [`QUICKSTART.md`](QUICKSTART.md).

## Contents

```text
transferattack/   attack implementations and model utilities
experiments/      generation and evaluation entry points
data/             labels and dataset-format documentation
checkpoints/      checkpoint instructions; weights are not bundled
docs/             detailed experiment protocols
outputs/          generated artifacts, ignored by Git
```

## Environment

The paper environment used Python 3.9, PyTorch 1.12.1, torchvision 0.13.1,
CUDA 11.6, and timm 0.6.12.

```bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
```

Install optional evaluation packages only when needed:

```bash
pip install -r requirements-evaluation.txt
```

`pyiqa` is optional and may replace the pinned PyTorch build. Use a separate
environment for it if necessary. Verify the installation before a long run:

```bash
python -c "import torch, timm, transferattack; print(torch.__version__, timm.__version__, torch.cuda.is_available())"
```

## Data

The NIPS 2017 ImageNet-Compatible Dataset is expected as:

```text
data/
|-- images/*.png
`-- labels.csv
```

`labels.csv` contains `filename,label,targeted_label`, using zero-based ImageNet
indices. The label file is bundled, but the images must be obtained under their
original distribution terms. See [`data/README.md`](data/README.md) for complete
preparation instructions, including ImageNet validation and ImageNet-V2.

## Main Results

Run commands from this package root. The main setting is `epsilon=16/255`,
`T=10`, `alpha=1.6/255`, momentum `mu=1`, and seed 42.

```bash
python experiments/run_multi_source.py \
  --mode both --attack eda \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --input_dir ./data --output_root ./outputs/main/eda \
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

On AutoDL, expose physical GPU 1 and address it as logical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/run_multi_source.py --mode both --attack eda --input_dir ./data --output_root ./outputs/main/eda --GPU_ID 0
```

Change `--attack` to `l2t`, `bsr`, `decowa`, `ops`, or `sid` for baselines.
Detailed commands and table mappings are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). A complete entry-point list is in
[`docs/SCRIPT_INDEX.md`](docs/SCRIPT_INDEX.md).

## Sampling Fairness

The scripts preserve each method's original hyperparameters while aligning the
transformation-sampling scale used for gradient estimation:

| Attack | Setting |
|---|---|
| L2T | Original configuration |
| OPS | `OPS(10,5,5)` |
| BSR | `num_scale=25` |
| DeCoWA | `num_warping=25`; original deformation scale retained |
| SID | `num_scale=25` |
| EDA | `num_warping=25`, `noise_scale=0.45` |

For EDA, `num_warping` is the number of independently transformed views whose
gradients are averaged at each attack iteration. It is not the iteration count.
OPS uses 25 sampled pairs plus its base gradient, so this aligns sampling
complexity rather than claiming identical runtime or internal optimization.

## Extended Evaluations

```bash
# Multiple budgets and targeted transfer (fresh full reproduction)
python experiments/run_budget_targeted_eda.py --mode untargeted_both --attacks l2t,bsr,decowa,ops,sid,eda --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0
python experiments/run_budget_targeted_eda.py --mode targeted_both --attacks l2t,bsr,decowa,ops,sid,eda --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0

# After all adversarial images exist: audit them, then evaluate the complete table
python experiments/run_budget_targeted_eda.py --mode audit --input_dir ./data --output_dir ./outputs/budget_targeted
python experiments/run_budget_targeted_eda.py --mode full_eval --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0

# Five seeds: 0,1,2,3,4
python experiments/run_seed_stability_eda.py --mode both --input_dir ./data --output_dir ./outputs/seed_stability --GPU_ID 0

# Perceptual quality
python experiments/run_perceptual_quality_eda.py --mode both --input_dir ./data --output_dir ./outputs/perceptual_quality --GPU_ID 0 --allow_missing_optional_metrics

# Modern RobustBench targets
python experiments/run_modern_robustbench_eval.py --mode both --input_dir ./data --output_dir ./outputs/robustbench --model_dir ./checkpoints/robustbench --allow_download --GPU_ID 0
```

The targeted experiment uses the `targeted_label` supplied for each image by
the ImageNet-Compatible Dataset; it does not sample new target labels.
`full_eval` never regenerates adversarial images. It first validates all 192
generation cases, evaluates the six attacks on all 20 target models, and writes
the complete CSV and LaTeX table. Add `--reuse_existing` only when resuming an
interrupted evaluation from its checkpoint CSV.

## VLM Evaluation

The optional VLM workflow incurs provider costs. API credentials are read from
environment variables and are never stored in source files. See
[`docs/VLM_EVALUATION.md`](docs/VLM_EVALUATION.md) for preparation, prompts,
model routes, missing responses, and aggregation.

## Reproducibility Notes

- Torchvision/timm weights are downloaded to `TORCH_HOME` on first use.
- Set `TIMM_HUB_SERVER` to override the default timm mirror.
- RobustBench models are downloaded from their official registrations.
- Do not combine outputs generated with different seeds or configurations.
- Use `--reuse_existing` only when saved configuration metadata matches.
- GPU/CUDA versions and provider-hosted VLM revisions can cause small variation.

Before running experiments or redistributing the package, run:

```bash
python tools/check_release.py
```

The same dependency-free check runs automatically through GitHub Actions.

## License

The bundled TransferAttack-derived code is distributed under [`LICENSE`](LICENSE).
Dataset images and third-party weights retain their own licenses and are not
redistributed. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
attribution details and [`CITATION.cff`](CITATION.cff) for software citation.
