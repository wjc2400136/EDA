# 快速入门

[English](QUICKSTART.md) | [简体中文](QUICKSTART.zh-CN.md)

本代码包包含复现 EDA 实验所需的代码和文档。数据集图像、模型权重、实验输出、
服务商原始响应和 API 密钥不随包提供；其准备方法见
[`data/README.zh-CN.md`](data/README.zh-CN.md) 和
[`checkpoints/README.zh-CN.md`](checkpoints/README.zh-CN.md)。

## 1. 安装

```bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
```

## 2. 准备主数据集

将 NIPS 2017 ImageNet-Compatible Dataset 的 1,000 张图像放入
`data/images/`。随包提供的 `data/labels.csv` 包含 `filename`、`label` 和
`targeted_label`。

## 3. 复现 EDA 主实验

```bash
python experiments/run_multi_source.py \
  --mode both --attack eda \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --input_dir ./data --output_root ./outputs/main/eda \
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

论文主设置为 `epsilon=16/255`、`T=10`、`alpha=1.6/255`、`mu=1` 和随机种子
42。`num_warping=25` 表示每次攻击迭代中参与梯度平均的变换视图数，不是攻击迭代数。

## 4. 运行扩展评估

多预算、目标迁移、五随机种子、RobustBench、感知质量、跨数据集泛化、消融和 VLM
聚合命令见 [`docs/EXPERIMENTS.zh-CN.md`](docs/EXPERIMENTS.zh-CN.md) 与
[`docs/SCRIPT_INDEX.zh-CN.md`](docs/SCRIPT_INDEX.zh-CN.md)。VLM 工作流会产生
外部服务费用，因此属于可选流程；检查随包代码和聚合协议不需要再次调用服务商。

## 5. 输出与核验

生成结果写入 `outputs/`，且默认不进入版本控制。比较或重新分发代码包前运行
`python tools/check_release.py`。完整协议见 [`README.zh-CN.md`](README.zh-CN.md)。
