# Quick Start

[English](QUICKSTART.md) | [Simplified Chinese](QUICKSTART.zh-CN.md)

## 1. Install

~~~bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
~~~

## 2. Prepare data

Place the 1,000 ImageNet-Compatible PNG images in data/images/. The matching
data/labels.csv is included.

## 3. Generate and evaluate

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

The command reproduces the principal CNN-source transfer protocol. The
--num_warping value is mapped to num_scale for BSR/SID and to num_warping for
DeCoWA/EDA; L2T and OPS retain their original configurations. The deformation
scale 0.45 is used only by EDA.

See [README.md](README.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for
the complete protocol and output definitions.
