# 实验脚本索引

[English](SCRIPT_INDEX.md) | [简体中文](SCRIPT_INDEX.zh-CN.md)

| 脚本 | 用途 | 主要输出 |
|---|---|---|
| `run_multi_source.py` | 四个 CNN 源模型的主实验 | 对抗图像、逐目标 ASR、CNN/ViT 平均值 |
| `run_budget_targeted_eda.py` | 四个预算及 targeted/untargeted | CSV、TeX 表格行、分析文本 |
| `run_seed_stability_eda.py` | 种子 0--4 的稳定性 | 每种子结果、mean ± sample std |
| `run_modern_robustbench_eval.py` | 现代对抗训练模型 | 各鲁棒模型 ASR 和汇总表 |
| `run_perceptual_quality_eda.py` | 最终对抗样本感知质量 | SSIM、PSNR、LPIPS、TV、NMSE、NLPD、GMSD |
| `run_feature_cam_metrics_eda.py` | feature 和 Grad-CAM 变化 | L2、KL、cosine、JS、correlation 等 |
| `run_parts_ablation_eda.py` | 三个主要组件消融 | 组件组合 ASR |
| `run_layout_ablation_eda.py` | single/dual 与 edge movement | CNN/ViT ASR |
| `run_expansion_ablation_eda.py` | canvas expansion 模式 | ASR、边界可视化和图表 |
| `run_appearance_ablation_eda.py` | noise/brightness 及参数敏感性 | CSV 和敏感性图 |
| `tune_noise_scale_eda.py` | TPS noise scale 敏感性 | 不同 `s` 下 ASR |
| `run_imagenet_val_generalization_eda.py` | ImageNet-Val 50K | 大规模泛化 ASR |
| `run_imagenetv2_generalization_eda.py` | ImageNet-V2 10K | 分布偏移 ASR/CC-ASR |
| `run_imagenet_compatible_generalization_eda.py` | 统一泛化 runner 包装 | 1K 对照结果 |
| `prepare_imagenet_val_labels.py` | ImageNet-Val 格式转换 | `images/` 和 `labels.csv` |
| `prepare_imagenetv2_labels.py` | ImageNet-V2 格式转换 | `images/` 和 `labels.csv` |
| `vlm/prepare_vlm_subset.py` | 固定 100 图像和 prompt | 7 个方法目录、prompt 文件 |
| `vlm/run_vlm_batch.py` | VLM 请求和原始记录 | 每 prompt/method/trial JSON |
| `vlm/summarize_vlm_results.py` | VLM 指标汇总 | per-model 和 macro CSV |

## 旧防御脚本

`run_robust_table.py` 保留用于原稿中的旧防御与预处理模型，但依赖若干第三方实现和
checkpoint。大修新增的现代鲁棒模型实验应优先使用
`run_modern_robustbench_eval.py`。

## 脚本选择建议

- 只复现论文核心结论：主实验 + parts/layout ablation；
- 回应 Reviewer #2-5：budget/targeted 脚本；
- 回应 Reviewer #2-6：seed stability 脚本；
- 回应 Reviewer #2-7：modern RobustBench 脚本；
- 回应 Reviewer #2-9：perceptual quality 脚本；
- 回应 larger/distribution shift：ImageNet-V2 或 ImageNet-Val；
- 回应 VLM 定量协议：VLM 三个脚本共同使用。

所有脚本均可通过 `python experiments/<script>.py --help` 查看当前参数。运行长任务
前应把实际命令保存到实验输出目录之外的日志中。

