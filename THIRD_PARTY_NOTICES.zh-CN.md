# 第三方说明

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

本仓库包含源自第三方项目或与第三方项目协同工作的代码。本说明用于补充而非
替代 [LICENSE](LICENSE) 中的条款。

## TransferAttack

攻击框架和部分攻击实现源自 Trustworthy-AI-Group 的 TransferAttack 项目，
仓库保留其上游版权声明和 MIT 许可证。EDA 实现、主要实验脚本和复现文档由
匿名作者补充。

## 运行依赖

PyTorch、torchvision、timm、NumPy、pandas、Pillow、SciPy、Matplotlib、
scikit-optimize 以及 requirements.txt 中的其他软件包仍受各自许可证约束。

## 数据集和模型权重

ImageNet、NIPS 2017 ImageNet-Compatible Dataset 和预训练模型权重不由本
仓库授权，也不随代码分发。使用者须从获授权来源获取并遵守相应条款。

## 对比方法

仓库包含用于研究比较的已发表迁移攻击实现。相关论文已在稿件和源代码中引用；
重新分发时还应检查各原始项目及其许可证。
