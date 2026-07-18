# Script Index

[English](SCRIPT_INDEX.md) | [简体中文](SCRIPT_INDEX.zh-CN.md)

| Script | Purpose |
|---|---|
| `run_multi_source.py` | Main CNN-source generation and CNN/ViT evaluation |
| `run_budget_targeted_eda.py` | Four budgets and untargeted/targeted objectives |
| `run_seed_stability_eda.py` | Five-seed mean and sample standard deviation |
| `run_modern_robustbench_eval.py` | Current adversarially trained RobustBench models |
| `run_perceptual_quality_eda.py` | SSIM, PSNR, LPIPS, TV, NMSE, NLPD, and GMSD |
| `run_gradient_eda_combination.py` | EDA combined with VMI-FGSM, EMI-FGSM, PGN, MEF, and GAA |
| `run_parts_ablation_eda.py` | Component ablation |
| `run_layout_ablation_eda.py` | Control-point layout and edge-movement ablation |
| `run_expansion_ablation_eda.py` | Canvas expansion modes and visualizations |
| `run_appearance_ablation_eda.py` | Noise/brightness component and parameter sensitivity |
| `tune_noise_scale_eda.py` | TPS deformation-scale sensitivity |
| `run_imagenet_val_generalization_eda.py` | Full ImageNet validation evaluation |
| `run_imagenetv2_generalization_eda.py` | ImageNet-V2 matched-frequency evaluation |
| `prepare_imagenet_val_labels.py` | Prepare ImageNet validation input schema |
| `prepare_imagenetv2_labels.py` | Prepare ImageNet-V2 input schema |
| `vlm/prepare_vlm_subset.py` | Build deterministic 100-image VLM tree and prompt manifest |
| `vlm/run_vlm_batch.py` | Provider API requests and raw JSON records |
| `vlm/summarize_vlm_results.py` | Per-model and macro VLM summary CSV |

`run_robust_table.py` is retained for the paper's legacy defense table but
requires external defense implementations/checkpoints. Prefer the RobustBench
runner for a portable modern-defense reproduction.
