# Data Preparation

[English](README.md) | [简体中文](README.zh-CN.md)

## NIPS 2017 ImageNet-Compatible Dataset

Place the 1,000 PNG images in `data/images/`. The bundled CSV uses:

```csv
filename,label,targeted_label
0c7ac4a8c9dfa802.png,305,778
```

Both labels are zero-based ImageNet indices. Untargeted attacks use `label`;
targeted attacks use the dataset-provided `targeted_label`. Keep this file
identical across all attacks and budgets.

Verify the dataset before running:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
df = pd.read_csv('data/labels.csv')
files = {p.name for p in Path('data/images').glob('*.png')}
missing = sorted(set(df.filename) - files)
print({'rows': len(df), 'images': len(files), 'missing': len(missing)})
assert len(df) == 1000 and not missing
PY
```

## ImageNet Validation

Obtain ILSVRC2012 validation images and the official development kit under
their original terms, then run:

```bash
python experiments/prepare_imagenet_val_labels.py \
  --image_root /path/to/ILSVRC2012_img_val \
  --devkit_dir /path/to/ILSVRC2012_devkit_t12 \
  --class_index ./data/imagenet_class_index.json \
  --output_dir ./data_imagenet_val --mode both
```

## ImageNet-V2 Matched Frequency

Extract the matched-frequency release and run:

```bash
python experiments/prepare_imagenetv2_labels.py \
  --image_root /path/to/imagenetv2-matched-frequency-format-val \
  --output_dir ./data_imagenetv2_matched --mode both
```

Target labels generated for these auxiliary datasets are deterministic and only
provide a consistent file schema; untargeted generalization does not use them.

Do not resize or recompress source images manually. Model wrappers apply the
required resize and normalization.
