# Enhanced Deformation Attack：主要迁移实验

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库仅包含主要对比实际使用的六种攻击：L2T、BSR、DeCoWA、OPS、
SID 和 EDA；公共优化与模型加载代码仅作为内部依赖保留。该流程使用四个
CNN 源模型生成对抗样本，并在主实验采用的十个 CNN 和十个
ViT 目标模型上评估。主要表格之外的补充分析不属于本代码包范围。

## 目录

~~~text
transferattack/                 六种攻击实现及公共工具
experiments/run_multi_source.py 主要生成与评估入口
data/                           标签与数据格式说明
checkpoints/                    预训练权重说明
docs/                           实验协议与脚本说明
outputs/                        生成结果，默认不纳入 Git
~~~

## 环境

参考环境为 Python 3.9、PyTorch 1.12.1、torchvision 0.13.1、
CUDA 11.6 和 timm 0.6.12。

~~~bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
~~~

## 数据

将 NIPS 2017 ImageNet-Compatible Dataset 的 1,000 张 PNG 图像放入
data/images/。随包提供的 data/labels.csv 包含文件名和从 0 开始的
ImageNet 标签。由于许可限制，本仓库不分发数据集图像。详见
[data/README.zh-CN.md](data/README.zh-CN.md)。

## 复现主要表格

在仓库根目录运行：

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

主实验采用 epsilon=16/255、alpha=1.6/255、10 次迭代、动量 1.0 和随机种子
42。CNN 平均值排除与源模型完全相同的目标检查点；ViT 平均值包含全部十个
ViT 目标模型。

## 采样配置

通用命令行参数 --num_warping 用于设置参与变换梯度平均的匹配视图数 N。
脚本将 N=25 映射到各方法自身的参数：

| 方法 | 脚本实际采用的配置 |
|---|---|
| L2T | 原始配置 |
| OPS | 原始 OPS(10,5,5) 配置 |
| BSR | num_scale=25 |
| DeCoWA | num_warping=25，并保留原始形变尺度 |
| SID | num_scale=25 |
| EDA | num_warping=25，noise_scale=0.45 |

对 EDA 而言，num_warping 表示每次攻击迭代中独立生成并参与梯度平均的变换
视图数，并不是攻击迭代次数。noise_scale=0.45 仅传递给 EDA，不会传递给
DeCoWA。

## 输出

脚本按攻击方法和源模型在 outputs/main/ 下保存对抗样本、生成元数据和耗时，
并输出逐目标 ASR 以及排除同源模型后的 CNN/ViT 组平均值。仅生成、仅评估和
断点续跑方式见 [docs/EXPERIMENTS.zh-CN.md](docs/EXPERIMENTS.zh-CN.md)。

预训练模型权重由标准 torchvision/timm 加载器获取，不随代码包分发。详见
[checkpoints/README.zh-CN.md](checkpoints/README.zh-CN.md)。

## 许可证

随包代码依据 [LICENSE](LICENSE) 分发。数据集图像和第三方权重沿用其原始
许可证。详见 [THIRD_PARTY_NOTICES.zh-CN.md](THIRD_PARTY_NOTICES.zh-CN.md)。
