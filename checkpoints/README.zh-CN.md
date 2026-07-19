# 模型权重

[English](README.md) | [简体中文](README.zh-CN.md)

本代码包不分发第三方模型权重。主要实验通过随包提供的 torchvision/timm
模型包装器加载标准预训练 CNN 和 ViT 检查点。需要时可设置缓存目录：

~~~bash
export TORCH_HOME=$PWD/checkpoints/torch
export HF_HOME=$PWD/checkpoints/huggingface
~~~

若权重尚未缓存，首次加载需要网络连接。请保持 environment.yml 和
requirements.txt 中的软件版本，因为不同版本的模型注册和预处理默认值可能
不同。
