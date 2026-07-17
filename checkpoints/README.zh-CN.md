# 模型权重与缓存

[English](README.md) | [简体中文](README.zh-CN.md)

本代码包不重新分发第三方预训练权重。标准模型、RobustBench 模型和旧防御模型的
来源及缓存方式不同，应分开管理。

## 1. 标准 torchvision 和 timm 模型

主实验中的 CNN、ViT 和混合架构模型会在首次运行时自动下载官方预训练权重。
建议将缓存固定在项目目录或大容量磁盘：

```bash
export TORCH_HOME=$PWD/checkpoints/torch
export HF_HOME=$PWD/checkpoints/huggingface
mkdir -p "$TORCH_HOME" "$HF_HOME"
```

代码默认允许通过 `TIMM_HUB_SERVER` 覆盖 timm 下载源：

```bash
export TIMM_HUB_SERVER=https://your-approved-model-mirror.example/
```

如果不设置，则使用代码中的默认镜像。首次运行需要网络，后续运行会复用缓存。
在无网络机器上，应先在兼容环境中下载，再完整复制缓存目录。

## 2. RobustBench 模型

现代鲁棒模型通过 RobustBench 官方注册信息加载：

```bash
python experiments/run_modern_robustbench_eval.py \
  --mode both \
  --input_dir ./data \
  --output_dir ./outputs/robustbench \
  --model_dir ./checkpoints/robustbench \
  --allow_download \
  --GPU_ID 0
```

首次下载后可去掉 `--allow_download`，以避免运行过程中意外联网或下载了不同版本
的权重。为保证可复现性，应记录：

- RobustBench 包版本；
- model name；
- dataset 和 threat model；
- checkpoint 文件名和哈希；
- 模型要求的输入预处理。

可以先列出当前安装版本可识别的模型：

```bash
python experiments/run_modern_robustbench_eval.py --list_models
```

## 3. 旧防御模型

`experiments/run_robust_table.py` 用于原稿中的旧防御/预处理表，部分模型需要从
原作者项目下载实现或 checkpoint。由于授权和依赖问题，这些文件不包含在本包中。

如需复现，应为每个模型记录：

```text
model_name
original_repository_or_paper
checkpoint_filename
sha256
preprocessing
framework_version
```

不要用名称相近但训练协议不同的 checkpoint 替代。现代 defense 评估优先使用
`run_modern_robustbench_eval.py`，因为该流程更标准化。

## 4. 缓存与输出不要提交

`checkpoints/` 已在 `.gitignore` 中排除，仅保留本说明文件。提交代码前确认没有：

- `.pth`、`.pt`、`.ckpt`、`.safetensors`；
- provider API token；
- 带有个人绝对路径的配置；
- 未获授权的模型文件。

若通过 Zenodo 单独发布可再分发的权重，应提供校验和、许可证和与代码版本对应的
固定归档链接。
