# Principal Transfer Experiment

[English](EXPERIMENTS.md) | [Simplified Chinese](EXPERIMENTS.zh-CN.md)

Run all commands from the repository root after installing the package.

## Protocol

| Item | Setting |
|---|---|
| Dataset | NIPS 2017 ImageNet-Compatible Dataset, 1,000 images |
| Source models | RN-18, Inc-v3, Inc-v4, IncRes-v2 |
| Target groups | 10 CNNs and 10 ViTs |
| Norm and budget | Linf, 16/255 |
| Iterations and step size | 10 and 1.6/255 |
| Momentum and seed | 1.0 and 42 |
| Matched transformed views | 25 |
| EDA mesh and scale | 3 x 3 and 0.45 |

CNN group averages omit the source-identical checkpoint. ViT group averages
include all ten ViT targets because the sources in this experiment are CNNs.

## Complete run

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

## Separate generation and evaluation

Generate only:

~~~bash
python experiments/run_multi_source.py --mode generate --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

Evaluate existing adversarial images only:

~~~bash
python experiments/run_multi_source.py --mode eval --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

To continue an interrupted generation, add --reuse_existing. The runner checks
generation_meta.json and rejects reuse when the saved source, attack, seed,
budget, step size, iteration count, or transformed-sample configuration differs
from the requested protocol. Use --clean only when intentionally replacing a
case directory.

## Method-specific mapping

The command-line name --num_warping is a common interface, not one shared
internal parameter. The runner maps 25 to BSR num_scale, DeCoWA num_warping,
SID num_scale, and EDA num_warping. L2T and OPS retain their original method
settings. EDA noise_scale=0.45 is not passed to DeCoWA.

## Outputs

Each attack/source directory contains the generated PNG images and
generation_meta.json. The output root also contains:

- results_eval_ATTACK_multi_source.txt: per-target ASR, timing, and group means.
- results_eval_ATTACK_multi_source_analysis.txt: concise source-wise summary.
- generation_time_ATTACK_multi_source.txt: generation-only timing summary.

The evaluation covers VGG-19, RN-18, RN-50, RN-101, RNX-50, DN-121, MB-v2,
Inc-v3, Inc-v4, IR-v2, ViT-B, DeiT-B, LeViT, PiT-B, CaiT-S, ConViT-B, TNT-S,
Visformer-S, Swin-T, and CoaT-T.
