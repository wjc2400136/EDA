# Quick Start

[English](QUICKSTART.md) | [Simplified Chinese](QUICKSTART.zh-CN.md)

This package contains the code and documentation needed to reproduce the EDA
experiments. Dataset images, model weights, generated outputs, provider responses,
and API credentials are not bundled. Their preparation is documented in
[`data/README.md`](data/README.md) and
[`checkpoints/README.md`](checkpoints/README.md).

## 1. Install

```bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
```

## 2. Prepare the main dataset

Place the 1,000 NIPS 2017 ImageNet-Compatible images in `data/images/`. The
bundled `data/labels.csv` provides `filename`, `label`, and `targeted_label`.

## 3. Reproduce the main EDA evaluation

```bash
python experiments/run_multi_source.py \
  --mode both --attack eda \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --input_dir ./data --output_root ./outputs/main/eda \
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

The paper setting is `epsilon=16/255`, `T=10`, `alpha=1.6/255`, `mu=1`, and
seed 42. `num_warping=25` is the number of transformed views averaged per attack
iteration, not the number of attack iterations.

## 4. Run extended evaluations

Commands for multiple budgets, targeted transfer, five-seed stability,
RobustBench evaluation, perceptual quality, generalization, ablations, and VLM
aggregation are listed in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and
[`docs/SCRIPT_INDEX.md`](docs/SCRIPT_INDEX.md). The VLM workflow uses paid
providers and is optional; no additional provider calls are required to inspect
the included evaluation and aggregation code.

## 5. Output and verification

Generated files are written under `outputs/` and are excluded from version
control. Run `python tools/check_release.py` before comparing or redistributing
the package. See [`README.md`](README.md) for the complete protocol.
