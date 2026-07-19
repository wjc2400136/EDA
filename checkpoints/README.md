# Model Weights

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

No third-party model weights are redistributed. The principal experiment uses
the standard pretrained CNN and ViT checkpoints loaded by the included
torchvision/timm model wrappers. Configure cache locations when needed:

~~~bash
export TORCH_HOME=$PWD/checkpoints/torch
export HF_HOME=$PWD/checkpoints/huggingface
~~~

Internet access is required when a checkpoint is not already cached. Preserve
the package versions in environment.yml and requirements.txt because model
registrations and preprocessing defaults can vary across releases.
