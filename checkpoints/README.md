# Model Weights

[English](README.md) | [简体中文](README.zh-CN.md)

No third-party weights are redistributed in this package.

Torchvision and timm download official pretrained weights automatically. Set
cache locations if desired:

```bash
export TORCH_HOME=$PWD/checkpoints/torch
export HF_HOME=$PWD/checkpoints/huggingface
```

For modern robust targets, `run_modern_robustbench_eval.py` loads registered
models through RobustBench. Use `--model_dir ./checkpoints/robustbench
--allow_download` on the first run.

`run_robust_table.py` covers legacy defenses and may require checkpoints from
their original authors. Record the source URL and checksum for every manually
downloaded checkpoint. The RobustBench runner is the recommended self-contained
defense evaluation for the revision.
