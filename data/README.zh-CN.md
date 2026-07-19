# 数据准备

[English](README.md) | [简体中文](README.zh-CN.md)

请依据原始分发条款获取 NIPS 2017 ImageNet-Compatible Dataset 的 1,000 张
PNG 图像，并放入 data/images/。

目录结构：

~~~text
data/
|-- images/
|   |-- IMAGE_1.png
|   |-- ...
|-- labels.csv
~~~

随包提供的 labels.csv 包含 filename、label 和 targeted_label 列，标签均为
从 0 开始的 ImageNet 类别索引。主要非目标迁移实验使用 filename 和 label。
请勿重命名、缩放或重新压缩图像；模型包装器会执行所需预处理。
