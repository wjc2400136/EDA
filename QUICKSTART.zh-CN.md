# 快速开始

[English](QUICKSTART.md) | [简体中文](QUICKSTART.zh-CN.md)

## 1. 安装

~~~bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
~~~

## 2. 准备数据

将 ImageNet-Compatible Dataset 的 1,000 张 PNG 图像放入 data/images/。
匹配的 data/labels.csv 已随代码包提供。

## 3. 生成并评估

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

该命令复现主要 CNN 源模型迁移协议。--num_warping 会映射为 BSR/SID 的
num_scale 和 DeCoWA/EDA 的 num_warping；L2T 与 OPS 保留原始配置。
形变尺度 0.45 仅用于 EDA。

完整协议和输出定义见 [README.zh-CN.md](README.zh-CN.md) 与
[docs/EXPERIMENTS.zh-CN.md](docs/EXPERIMENTS.zh-CN.md)。
