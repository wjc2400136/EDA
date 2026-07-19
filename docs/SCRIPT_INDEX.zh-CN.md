# 脚本索引

[English](SCRIPT_INDEX.md) | [简体中文](SCRIPT_INDEX.zh-CN.md)

匿名代码包只提供一个实验入口：

| 脚本 | 用途 |
|---|---|
| experiments/run_multi_source.py | 为主要 CNN 源模型迁移表生成并评估六种对比攻击。 |

使用 --mode generate、--mode eval 或 --mode both 选择流程；使用 --attack
运行单个方法，或使用 --attacks 运行逗号分隔的方法列表。全部参数可通过
python experiments/run_multi_source.py --help 查看。
