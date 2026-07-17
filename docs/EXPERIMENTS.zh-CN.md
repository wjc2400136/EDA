# 实验复现流程

[English](EXPERIMENTS.md) | [简体中文](EXPERIMENTS.zh-CN.md)

本文档给出从环境检查、主实验到大修新增实验的建议执行顺序。所有命令默认在
`EDA_reproducibility_package` 根目录运行，并已执行：

```bash
conda activate eda-repro
pip install -e .
```

## 1. 统一实验设置

| 参数 | 论文设置 |
|---|---:|
| 攻击范数 | Linf |
| 主实验预算 | 16/255 |
| 迭代次数 `T` | 10 |
| 步长 `alpha` | epsilon / T |
| 动量系数 `mu` | 1.0 |
| EDA full grid | 3 x 3 |
| EDA center grid | 2 x 2 |
| EDA noise scale | 0.45 |
| 变换样本数 `N` | 25 |
| 主实验种子 | 42 |

CNN 分组平均值排除与源模型完全相同的 checkpoint；相同模型的白盒结果仍可在表中
报告并标星。CNN 源模型攻击 ViT 时不存在 source-identical ViT target，因此 ViT
平均值使用表中全部目标。

## 2. 建议执行顺序

1. 检查 1K 数据和标签。
2. 用 RN-18 + EDA 做一小批 smoke test。
3. 完成六种方法、四个源模型的主实验。
4. 完成多预算和 targeted attack。
5. 完成五随机种子实验。
6. 运行消融和参数敏感性。
7. 运行现代鲁棒模型、感知质量和 feature/Grad-CAM 指标。
8. 最后运行 ImageNet-V2 或 ImageNet-Val 大规模泛化。
9. VLM 实验独立执行，不与 GPU 攻击实验共用输出目录。

## 3. 运行模式说明

大部分脚本支持：

- `--mode generate`：只生成对抗样本；
- `--mode eval`：只评估已有对抗样本；
- `--mode both`：先生成后评估。

多预算脚本使用更明确的模式名：

- `untargeted_generate`、`untargeted_eval`、`untargeted_both`；
- `targeted_generate`、`targeted_eval`、`targeted_both`；
- `corrected_both`：只为修复旧实验而重跑指定方法，不是全量模式。

## 4. 主实验：CNN 源模型

建议每种攻击单独运行并保存到独立目录：

```bash
for ATTACK in l2t bsr decowa ops sid eda; do
  python experiments/run_multi_source.py \
    --mode both \
    --attack "$ATTACK" \
    --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
    --input_dir ./data \
    --output_root "./outputs/main/$ATTACK" \
    --num_warping 25 \
    --noise_scale 0.45 \
    --seed 42 \
    --GPU_ID 0
done
```

虽然命令统一包含 `--noise_scale 0.45`，runner 只将它用于 EDA，不会覆盖
DeCoWA 自身的 deformation scale。`--num_warping 25` 会按方法映射为：

- BSR：`num_scale=25`；
- DeCoWA：`num_warping=25`；
- SID：`num_scale=25`；
- EDA：`num_warping=25`；
- OPS 和 L2T 保留自身配置。

建议先执行 `python experiments/run_multi_source.py --help`，确认当前代码版本的
参数名后再提交长任务。

## 5. 多预算和 targeted attack

### 完整、全新的六方法实验

使用一个全新的输出目录，先完成非目标，再完成目标攻击：

```bash
python experiments/run_budget_targeted_eda.py \
  --mode untargeted_both \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --epsilons 4/255,8/255,12/255,16/255 \
  --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full \
  --epoch 10 --num_warping 25 --noise_scale 0.45 \
  --seed 42 --GPU_ID 0

python experiments/run_budget_targeted_eda.py \
  --mode targeted_both \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --epsilons 4/255,8/255,12/255,16/255 \
  --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full \
  --epoch 10 --num_warping 25 --noise_scale 0.45 \
  --seed 42 --GPU_ID 0
```

两类对抗样本全部生成后，先核验完整性，再只运行评估：

```bash
python experiments/run_budget_targeted_eda.py \
  --mode audit --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full

python experiments/run_budget_targeted_eda.py \
  --mode full_eval --input_dir ./data \
  --output_dir ./outputs/budget_targeted_full --GPU_ID 0
```

`full_eval` 强制使用论文中的六种方法并检查全部 192 个生成案例，不会重新生成
图像。它输出完整 CSV、TeX 表格行、完整 LaTeX 总表、分析文本和生成审计 CSV。
只有从中断的完整评估续跑时才添加 `--reuse_existing`。

脚本按 `alpha=epsilon/epoch` 自动缩放步长。targeted attack 使用
`data/labels.csv` 中的数据集原始 `targeted_label`。

### 恢复旧实验

`corrected_both` 默认只处理 BSR、DeCoWA 和 SID，用于修复此前采样数量不一致的
旧输出。它会尝试保留 L2T、OPS 和 EDA 的已有结果，因此只有在这些已有结果的
metadata 与当前设置完全相同时才可使用：

```bash
python experiments/run_budget_targeted_eda.py \
  --mode corrected_both \
  --input_dir ./data \
  --output_dir ./outputs/budget_targeted_existing \
  --GPU_ID 0
```

公开复现包更推荐全量重跑模式，避免依赖无法追踪的旧目录。

### 一致性核对

在同一数据、种子、超参数和代码版本下，多预算表的 `16/255` untargeted 结果应
与主实验一致。若 Inc-v4/BSR 或其他条目不一致，依次检查：

1. 是否读取了旧对抗样本目录；
2. BSR 是否使用 `num_scale=25`；
3. DeCoWA 是否保持自身 `noise_scale=2`；
4. `alpha` 是否为当前 epsilon 除以 10；
5. 主表和预算表是否使用相同 seed；
6. CNN 平均值是否排除 source-identical target；
7. 是否把单次结果与五随机种子均值混在一起。

## 6. 五随机种子稳定性

```bash
python experiments/run_seed_stability_eda.py \
  --mode both --seeds 0,1,2,3,4 \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --attacks l2t,bsr,decowa,ops,sid,eda \
  --input_dir ./data --output_dir ./outputs/seed_stability \
  --eps 16/255 --epoch 10 --num_warping 25 \
  --noise_scale 0.45 --GPU_ID 0
```

每个 seed 都从头生成 adversarial examples。表中报告 mean ± sample standard
deviation，排序只按 mean。不要将主实验 seed 42 的单次结果加入五种子均值。

## 7. 方法组件和参数消融

```bash
python experiments/run_parts_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_parts --GPU_ID 0
python experiments/run_layout_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_layout --GPU_ID 0
python experiments/run_expansion_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_expansion --GPU_ID 0
python experiments/run_appearance_ablation_eda.py --mode both --input_dir ./data --output_dir ./outputs/ablation_appearance --GPU_ID 0
python experiments/tune_noise_scale_eda.py --mode both --input_dir ./data --output_dir ./outputs/noise_scale --GPU_ID 0
```

`edge movement` 是 dual control point layout 内部的设计因素，不应在论文贡献中
单列为第四项创新。论文主线保持三项组件：dual layout、adaptive canvas
expansion、appearance augmentation。

## 8. 现代鲁棒模型

```bash
python experiments/run_modern_robustbench_eval.py --list_models

python experiments/run_modern_robustbench_eval.py \
  --mode both --input_dir ./data --output_dir ./outputs/robustbench \
  --model_dir ./checkpoints/robustbench --allow_download --GPU_ID 0
```

后续复现可去掉 `--allow_download`。结果应描述为“against the selected modern
robust models”，不要宣称突破 certified robustness。

## 9. 感知质量和特征指标

```bash
python experiments/run_perceptual_quality_eda.py \
  --mode both --input_dir ./data --output_dir ./outputs/perceptual \
  --GPU_ID 0 --allow_missing_optional_metrics

python experiments/run_feature_cam_metrics_eda.py \
  --mode both --input_dir ./data --output_dir ./outputs/features --GPU_ID 0
```

SSIM、PSNR、LPIPS 等比较 clean 和最终 adversarial image。TV 应在扰动
`delta=x_adv-x` 上计算。feature/Grad-CAM 指标用于说明内部表示变化，不等同于
分类层机制的因果证明。

## 10. 大规模与分布偏移数据

```bash
python experiments/run_imagenet_val_generalization_eda.py \
  --mode both --input_dir ./data_imagenet_val \
  --output_dir ./outputs/imagenet_val --GPU_ID 0

python experiments/run_imagenetv2_generalization_eda.py \
  --mode both --input_dir ./data_imagenetv2_matched \
  --output_dir ./outputs/imagenetv2 --GPU_ID 0
```

50K 实验耗时较长。若论文最终未填入 IN-Val-50K 结果，应删除空表行，而不是保留
`--`。ImageNet-V2 10K 已同时提供更大样本和自然分布偏移证据。

## 11. 断点恢复与失败处理

- `--reuse_existing` 只补齐缺失图片，不能自动保证已有图片配置正确；
- 每次长任务应保存命令、日志和代码版本；
- OOM 时只降低 batch size，不改变预算、迭代数或采样数；
- 模型下载中断时检查缓存中是否存在不完整文件；
- 评估前核对生成目录图片数与 `labels.csv` 行数；
- 不同 source、attack、seed、epsilon 和 threat 使用独立子目录；
- 生成完成后再汇总 CSV，不从半完成目录计算最终表格。

## 12. 推荐保存的运行记录

每组实验至少保存以下信息：

```text
command
Python/PyTorch/CUDA/timm versions
GPU model
source_model
attack_name
seed
epsilon and alpha
targeted_or_untargeted
transformation sample count
generation time
per-target ASR
source-excluded group averages
```

可使用 `python --version`、`pip freeze` 和 `nvidia-smi` 记录环境。
