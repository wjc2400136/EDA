# Enhanced Deformation Attack: Principal Transfer Experiments

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

This anonymous package contains the EDA implementation and the code required to
reproduce the principal CNN-source transfer experiments. It generates
adversarial examples with four CNN source models and evaluates them against the
ten CNN and ten ViT targets used in the main comparison. Supplementary analyses
outside these principal tables are not part of this package.

## Contents

~~~text
transferattack/                 attack implementations and model utilities
experiments/run_multi_source.py main generation and evaluation entry point
data/                           labels and dataset-format instructions
checkpoints/                    pretrained-weight instructions
docs/                           protocol and script documentation
outputs/                        generated artifacts, ignored by Git
~~~

## Environment

The reference environment uses Python 3.9, PyTorch 1.12.1,
torchvision 0.13.1, CUDA 11.6, and timm 0.6.12.

~~~bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
~~~

## Data

Place the 1,000 PNG images from the NIPS 2017 ImageNet-Compatible Dataset in
data/images/. The bundled data/labels.csv contains the filename and zero-based
ImageNet label for each image. Dataset images are not redistributed. See
[data/README.md](data/README.md).

## Reproduce the principal tables

Run the following command from the repository root:

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

The main protocol uses epsilon=16/255, alpha=1.6/255, 10 iterations, momentum
1.0, and seed 42. CNN averages exclude a source-identical checkpoint; ViT
averages include all ten ViT targets.

## Sampling configuration

The generic command-line option --num_warping sets the matched transformed-view
count N for methods that average transformed gradients. The runner maps N=25
to each method's native parameter:

| Method | Configuration used by the runner |
|---|---|
| L2T | Original configuration |
| OPS | Original OPS(10,5,5) configuration |
| BSR | num_scale=25 |
| DeCoWA | num_warping=25; original deformation scale retained |
| SID | num_scale=25 |
| EDA | num_warping=25 and noise_scale=0.45 |

For EDA, num_warping is the number of independently transformed views averaged
per attack iteration. It is not the attack iteration count. The option
noise_scale=0.45 is applied only to EDA; it is not forwarded to DeCoWA.

## Outputs

For each attack and source, the runner stores generated adversarial images,
generation metadata, and timing information below outputs/main/. It also writes
per-target ASR values and source-excluded CNN/ViT group averages. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for generation-only,
evaluation-only, and resume commands.

Pretrained model weights are obtained from the standard torchvision/timm
loaders and are not bundled. See [checkpoints/README.md](checkpoints/README.md).

## License

The bundled code is distributed under [LICENSE](LICENSE). Dataset images and
third-party weights retain their original licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
