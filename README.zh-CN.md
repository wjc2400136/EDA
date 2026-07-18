# Enhanced Deformation Attack：复现代码包

[English](README.md) | [简体中文](README.zh-CN.md)

本目录是论文 **Boosting Cross-Model Adversarial Transferability by Enhanced
Deformation Attack** 的独立复现代码包，包含 EDA、五种对比攻击，以及主实验、
消融实验、现代鲁棒模型、跨数据集泛化、感知质量、梯度攻击组合和 VLM 定量评估
所需脚本。

该目录为独立复现代码包，运行或修改其中的文件不会影响包外的工程目录。

最小端到端流程见 [`QUICKSTART.zh-CN.md`](QUICKSTART.zh-CN.md)。

## 1. 目录结构

```text
EDA_reproducibility_package/
|-- transferattack/              # 攻击算法、模型加载及数据工具
|-- experiments/                 # 对抗样本生成和评估入口
|   `-- vlm/                     # VLM 子集准备、请求和结果汇总
|-- data/                        # 标签文件和数据准备说明
|-- checkpoints/                 # 模型权重缓存目录，不附带第三方权重
|-- docs/                        # 详细实验流程和脚本索引
|-- outputs/                     # 实验输出，默认不提交到 Git
|-- environment.yml             # Conda 环境入口
|-- requirements.txt            # 核心实验依赖
`-- requirements-evaluation.txt # 可选评估依赖
```

主要文档：

- [完整实验流程](docs/EXPERIMENTS.zh-CN.md)
- [数据准备](data/README.zh-CN.md)
- [模型权重与缓存](checkpoints/README.zh-CN.md)
- [VLM 定量评估协议](docs/VLM_EVALUATION.zh-CN.md)
- [实验脚本索引](docs/SCRIPT_INDEX.zh-CN.md)

## 2. 环境配置

论文实验环境为 Python 3.9、PyTorch 1.12.1、torchvision 0.13.1、CUDA
11.6 和 timm 0.6.12。建议在 Linux 或 AutoDL 中创建独立环境：

```bash
cd /path/to/EDA_reproducibility_package
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
```

`pip install -e .` 会以可编辑方式安装当前目录，因此实验脚本可以直接导入
`transferattack`。核心环境完成后先检查：

```bash
python -c "import torch, timm, transferattack; print(torch.__version__, timm.__version__, torch.cuda.is_available())"
nvidia-smi
```

仅在运行相应实验时安装可选依赖：

```bash
pip install -r requirements-evaluation.txt
```

其中：

- `robustbench` 用于现代鲁棒模型；
- `lpips`、`scikit-image` 用于感知质量；
- `openai` 用于兼容 OpenAI Chat Completions 格式的 VLM 接口；
- `pyiqa` 是可选项，可能升级或替换固定的 PyTorch，建议单独建环境安装。

## 3. AutoDL 显卡映射

脚本中的 `--GPU_ID` 指进程可见的逻辑显卡编号。若希望使用机器上的物理显卡
1，可将它映射为进程内的逻辑显卡 0：

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/run_multi_source.py \
  --mode both --attack eda --input_dir ./data \
  --output_root ./outputs/main/eda --GPU_ID 0
```

此时应使用 `--GPU_ID 0`，而不是 1。若设置 `CUDA_VISIBLE_DEVICES=1,3`，则
物理显卡 1 和 3 分别对应逻辑显卡 0 和 1。

显存不足时优先降低 `--batchsize` 或 `--eval_batchsize`，不要修改
`num_warping=25`、迭代次数或攻击预算，否则实验设置会发生变化。

## 4. 数据准备

NIPS 2017 ImageNet-Compatible Dataset 应整理为：

```text
data/
|-- images/
|   |-- 0c7ac4a8c9dfa802.png
|   `-- ... 共 1000 张
`-- labels.csv
```

`labels.csv` 已包含在代码包中，字段为：

```csv
filename,label,targeted_label
0c7ac4a8c9dfa802.png,305,778
```

`label` 和 `targeted_label` 都是从 0 开始的 ImageNet 类别索引。非目标攻击使用
`label`；目标攻击直接使用数据集提供的 `targeted_label`，不会重新随机生成目标
类别。数据下载、完整性检查、ImageNet-Val 和 ImageNet-V2 的准备方法见
[数据准备文档](data/README.zh-CN.md)。

## 5. 复现主实验

所有命令均应在代码包根目录执行。论文主设置为：

| 设置 | 数值 |
|---|---:|
| 范数 | Linf |
| 最大扰动 | 16/255 |
| 迭代次数 | 10 |
| 步长 | 1.6/255 |
| 动量衰减 | 1.0 |
| EDA 网格 | 3 x 3 |
| EDA noise scale | 0.45 |
| 每次迭代变换样本数 | 25 |
| 主实验随机种子 | 42 |

生成并评估四个 CNN 源模型上的 EDA：

```bash
python experiments/run_multi_source.py \
  --mode both --attack eda \
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \
  --input_dir ./data --output_root ./outputs/main/eda \
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

将 `--attack` 改为 `l2t`、`bsr`、`decowa`、`ops` 或 `sid` 即可运行对应
基线。建议每种方法使用独立输出目录，避免结果混用。

## 6. 公平性设置

各方法保留其原论文超参数，同时统一输入变换的采样规模：

| 方法 | 实际设置 |
|---|---|
| L2T | 保留原始配置 |
| OPS | `OPS(10,5,5)` |
| BSR | `num_scale=25` |
| DeCoWA | `num_warping=25`，保留其自身 deformation scale |
| SID | `num_scale=25` |
| EDA | `num_warping=25`，`noise_scale=0.45` |

EDA 的 `num_warping` 表示**每次攻击迭代中独立生成并用于梯度平均的变换视图
数量**，不是攻击迭代次数。攻击仍执行 `T=10` 次。OPS 在每轮使用 25 个采样
组合及一个基础梯度，因此这里只对齐采样复杂度，不声称所有方法具有完全相同的
运行时间或内部反向传播结构。

## 7. 扩展评估

### 多预算和目标攻击

完整重跑必须分别执行非目标和目标模式：

```bash
python experiments/run_budget_targeted_eda.py \
  --mode untargeted_both --attacks l2t,bsr,decowa,ops,sid,eda \
  --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0

python experiments/run_budget_targeted_eda.py \
  --mode targeted_both --attacks l2t,bsr,decowa,ops,sid,eda \
  --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0

# 所有对抗样本生成完毕后：先检查，再只评估并输出完整总表
python experiments/run_budget_targeted_eda.py \
  --mode audit --input_dir ./data --output_dir ./outputs/budget_targeted
python experiments/run_budget_targeted_eda.py \
  --mode full_eval --input_dir ./data --output_dir ./outputs/budget_targeted --GPU_ID 0
```

默认预算为 `4/255,8/255,12/255,16/255`。完整复现应使用上述
`untargeted_both` 和 `targeted_both` 模式。
`full_eval` 不会重新生成对抗样本；它会先核验 192 个生成案例，再用 20 个目标
模型评估六种方法，并输出完整 CSV 和 LaTeX 总表。只有从中断的评估 CSV 续跑时
才添加 `--resume_full_eval`。

### 五个随机种子

```bash
python experiments/run_seed_stability_eda.py \
  --mode both --seeds 0,1,2,3,4 \
  --input_dir ./data --output_dir ./outputs/seed_stability --GPU_ID 0
```

每个种子都会重新生成对抗样本，最终报告均值和样本标准差。

### 现代鲁棒模型与感知质量

```bash
python experiments/run_modern_robustbench_eval.py \
  --mode both --input_dir ./data --output_dir ./outputs/robustbench \
  --model_dir ./checkpoints/robustbench --allow_download --GPU_ID 0

python experiments/run_perceptual_quality_eda.py \
  --mode both --input_dir ./data --output_dir ./outputs/perceptual_quality \
  --GPU_ID 0 --allow_missing_optional_metrics

python experiments/run_gradient_eda_combination.py \
  --mode both --input_dir ./data --output_dir ./outputs/gradient_combination \
  --GPU_ID 0
```

感知质量指标比较 clean image 与**最终对抗样本**，不比较 TPS 中间变换图像。

## 8. 输出、断点恢复和结果核对

多数脚本会保存：

- 对抗样本 PNG；
- 当前运行配置和耗时元数据；
- 每个目标模型的 ASR；
- CNN/ViT 分组平均值；
- CSV、TeX 表格行和分析文本。

使用 `--reuse_existing` 前必须确认已有目录的攻击方法、源模型、目标/非目标
模式、预算、种子、迭代数和采样数完全相同。不要将旧 CSV 与新生成的图像目录
组合使用。

在相同设置和随机种子下，多预算表中 `16/255` 的非目标结果应与主表一致。
若不一致，应检查输出目录是否混用了旧结果、公平性参数是否统一，以及平均值是否
排除了 source-identical checkpoint。

## 9. VLM 定量评估

VLM 实验会产生外部服务费用。检查已有结果及聚合流程不需要再次调用服务商。复现包包含：

- 固定 100 张图像的准备脚本；
- 完整固定 prompt；
- 无状态 API 批处理器；
- 原始 JSON 保存和完整性检查；
- 每个模型及跨模型宏平均汇总脚本。

API 密钥只能通过环境变量传入，不能写入 Python、Markdown 或提交到 Git。现有
五次实验使用不同 temperature，因此应表述为“五个 temperature 配置下的评估”，
不能表述为完全相同条件下的五个随机种子重复。详细协议见
[VLM 文档](docs/VLM_EVALUATION.zh-CN.md)。

## 10. 复现包检查

1. 确认 `data/images/`、模型权重和大型输出未提交。
2. 运行绝对路径和 API 密钥扫描。
3. 确认 README 中命令可从仓库根目录执行。
4. 确认主表、多预算表和多种子表使用同一平均协议。
5. 记录 Python、PyTorch、CUDA、timm、RobustBench 版本和 GPU 型号。
6. 在代码仓库中提供 README、环境文件、数据准备、命令示例和随机种子。
7. 对第三方数据和模型仅提供下载说明，不重新分发无授权文件。

运行实验或重新分发代码包前执行：

```bash
python tools/check_release.py
```

该检查不依赖 PyTorch，会检查必需文档、本地链接、疑似密钥、个人路径、大型文件和
误提交的模型权重。GitHub Actions 会在每次 push 和 pull request 时运行同一检查。

## 许可证与引用

TransferAttack 衍生代码按 [LICENSE](LICENSE) 提供。数据集图像和第三方模型权重
沿用各自许可证，不包含在本复现包中。第三方署名见
[THIRD_PARTY_NOTICES.zh-CN.md](THIRD_PARTY_NOTICES.zh-CN.md)，软件引用元数据见
[CITATION.cff](CITATION.cff)。
