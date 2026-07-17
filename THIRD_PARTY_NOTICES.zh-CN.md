# 第三方软件与资源声明

[English](THIRD_PARTY_NOTICES.md) | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

本仓库包含基于第三方项目修改的代码，或与第三方项目共同运行的接口。本说明补充
但不替代 [`LICENSE`](LICENSE) 中的许可条款。

## TransferAttack

攻击框架和部分攻击实现源自 Trustworthy-AI-Group 的 TransferAttack 项目。仓库
保留了上游版权声明和 MIT License。

EDA 实现、实验 runner、复现文档及相关评估工具包含 本项目贡献者 在 2026 年完成的新增和修改。

## 运行依赖

PyTorch、torchvision、timm、RobustBench、LPIPS、pytorch-grad-cam、OpenAI
客户端及 requirements 中列出的其他包均适用各自许可证。安装本项目不会改变这些
第三方许可条件。

## 数据集和模型权重

ImageNet、NIPS 2017 ImageNet-Compatible Dataset、ImageNet-V2、预训练模型
权重以及 provider 托管的 VLM 不由本仓库授权，也不在仓库中重新分发。使用者必须
通过被授权渠道获取并遵守相应条款。

## 对比方法实现

仓库为了科研比较包含若干已发表攻击方法的实现。论文和源代码中给出了对应引用。
如需单独重新分发某一方法实现，还应核查该方法原始项目的许可证。
