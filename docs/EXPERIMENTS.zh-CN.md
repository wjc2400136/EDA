# 主要迁移实验

[English](EXPERIMENTS.md) | [简体中文](EXPERIMENTS.zh-CN.md)

安装代码包后，请在仓库根目录运行所有命令。

## 协议

| 项目 | 设置 |
|---|---|
| 数据集 | NIPS 2017 ImageNet-Compatible Dataset，1,000 张图像 |
| 源模型 | RN-18、Inc-v3、Inc-v4、IncRes-v2 |
| 目标组 | 10 个 CNN 和 10 个 ViT |
| 范数与预算 | Linf，16/255 |
| 迭代与步长 | 10，1.6/255 |
| 动量与随机种子 | 1.0，42 |
| 匹配变换视图数 | 25 |
| EDA 网格与尺度 | 3 x 3，0.45 |

CNN 组平均值排除与源模型完全相同的目标检查点。由于本实验的源模型均为
CNN，ViT 组平均值包含全部十个 ViT 目标。

## 完整运行

~~~bash
python experiments/run_multi_source.py --mode both --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

## 分开生成与评估

仅生成：

~~~bash
python experiments/run_multi_source.py --mode generate --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

仅评估已有对抗样本：

~~~bash
python experiments/run_multi_source.py --mode eval --attacks l2t,bsr,decowa,ops,sid,eda --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 --input_dir ./data --output_root ./outputs/main --num_warping 25 --noise_scale 0.45 --GPU_ID 0
~~~

中断后继续生成时添加 --reuse_existing。脚本会检查 generation_meta.json；
若源模型、攻击、随机种子、预算、步长、迭代次数或变换样本配置不一致，则拒绝
复用。仅在确定需要替换整个配置目录时使用 --clean。

## 方法参数映射

命令行名称 --num_warping 是统一入口，不表示所有方法共享同一个内部参数。
脚本将 25 映射为 BSR 的 num_scale、DeCoWA 的 num_warping、SID 的
num_scale 和 EDA 的 num_warping。L2T 与 OPS 保留原始方法配置。
EDA 的 noise_scale=0.45 不会传递给 DeCoWA。

## 输出

每个攻击/源模型目录包含生成的 PNG 图像和 generation_meta.json。输出根目录
还包含逐目标 ASR、生成耗时、组平均值、源模型摘要和仅生成模式的耗时汇总。

评估目标包括 VGG-19、RN-18、RN-50、RN-101、RNX-50、DN-121、MB-v2、
Inc-v3、Inc-v4、IR-v2、ViT-B、DeiT-B、LeViT、PiT-B、CaiT-S、
ConViT-B、TNT-S、Visformer-S、Swin-T 和 CoaT-T。
