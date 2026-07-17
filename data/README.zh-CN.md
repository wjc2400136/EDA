# 数据准备

[English](README.md) | [简体中文](README.zh-CN.md)

所有攻击脚本统一使用“图像目录 + `labels.csv`”格式。不要为不同方法复制或修改
标签文件，否则 targeted ASR 和跨方法比较将失去一致性。

## 1. NIPS 2017 ImageNet-Compatible Dataset

将 1,000 张 PNG 图像放入：

```text
data/
|-- images/
|   |-- 0c7ac4a8c9dfa802.png
|   |-- f43fbfe8a9ea876c.png
|   `-- ...
`-- labels.csv
```

代码包已附带 `labels.csv`，格式为：

```csv
filename,label,targeted_label
0c7ac4a8c9dfa802.png,305,778
```

- `filename` 必须与 `images/` 中的文件名完全一致；
- `label` 是真实类别；
- `targeted_label` 是数据集提供的目标类别；
- 两种标签均为 0-based ImageNet 类别编号，范围为 0--999；
- targeted attack 直接使用 `targeted_label`，不会通过随机种子重新采样目标类别。

运行前检查图像数量、缺失文件、重复文件和标签范围：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

df = pd.read_csv('data/labels.csv')
image_dir = Path('data/images')
files = [p.name for p in image_dir.glob('*.png')]
file_set = set(files)
expected = set(df['filename'])

print({
    'csv_rows': len(df),
    'png_files': len(files),
    'missing': len(expected - file_set),
    'unexpected': len(file_set - expected),
    'duplicate_csv_filenames': int(df['filename'].duplicated().sum()),
})

assert len(df) == 1000
assert len(files) == 1000
assert not (expected - file_set)
assert not df['filename'].duplicated().any()
assert df['label'].between(0, 999).all()
assert df['targeted_label'].between(0, 999).all()
assert (df['label'] != df['targeted_label']).all()
PY
```

## 2. ImageNet Validation 50K

从官方渠道获得 ILSVRC2012 validation images 和 development kit。假设：

```text
/datasets/ILSVRC2012_img_val/       # 50,000 张验证图像
/datasets/ILSVRC2012_devkit_t12/    # 官方 ground truth
```

生成攻击代码需要的统一目录：

```bash
python experiments/prepare_imagenet_val_labels.py \
  --image_root /datasets/ILSVRC2012_img_val \
  --devkit_dir /datasets/ILSVRC2012_devkit_t12 \
  --class_index ./data/imagenet_class_index.json \
  --output_dir ./data_imagenet_val \
  --mode both
```

输出应为：

```text
data_imagenet_val/
|-- images/        # 统一后的图像
`-- labels.csv
```

`--mode generate` 只生成文件，`--mode verify` 只检查现有目录，`--mode both`
先生成再检查。已有正确数据时优先使用 verify，避免重复复制 50K 图像。

## 3. ImageNet-V2 Matched Frequency 10K

解压 matched-frequency 数据集后运行：

```bash
python experiments/prepare_imagenetv2_labels.py \
  --image_root /datasets/imagenetv2-matched-frequency-format-val \
  --output_dir ./data_imagenetv2_matched \
  --mode both
```

检查输出是否包含 10,000 张图像及对应标签。ImageNet-V2 用于自然分布偏移评估，
正文应明确它不是原 1K 数据的简单扩容，而是独立重新收集的数据。

## 4. 辅助数据集中的目标标签

准备脚本为了保持统一 CSV schema，会根据固定随机种子为 ImageNet-Val 和
ImageNet-V2 生成与真实类别不同的 `targeted_label`。论文中的跨数据集泛化实验
如果仅报告 untargeted ASR，则不会使用这些目标标签。不要把这种生成方式描述成
NIPS 2017 数据集的目标标签来源；NIPS 2017 的目标标签是数据集原始提供的。

## 5. 图像预处理

- 不要手动调整大小、裁剪或重新压缩源图像；
- 不要提前执行 ImageNet mean/std 归一化；
- 模型包装器会根据 torchvision 或 timm 配置完成 resize 和 normalization；
- 生成后的对抗图像保持原坐标系，TPS warping 仅用于梯度估计中的中间视图；
- 感知质量实验比较 clean image 和最终 adversarial image。

## 6. 数据许可与分发限制

代码包只包含标签、准备脚本和下载说明，不分发 ImageNet 或 NIPS 图像。使用者应
自行获取数据，并确认和遵守相应授权条款。
