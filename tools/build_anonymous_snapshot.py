"""Create an identity-scrubbed reviewer snapshot and ZIP archive."""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_NAMES = ()
SKIP_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "outputs",
    "results",
    "robustbench_models",
}
SKIP_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".onnx",
    ".pb",
    ".pyc",
    ".pt",
    ".pth",
    ".safetensors",
}
SKIP_RELATIVE_PREFIXES = {
    "data/images",
    "data_imagenet_val",
    "data_imagenetv2_matched",
    "imagenet-val",
    "vlm_eval/images",
    "vlm_eval/results",
}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def remove_readonly(func, path, _exc_info):
    """Retry removal after clearing Windows read-only attributes."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./dist/eda-review-anonymous")
    parser.add_argument("--archive", default="./dist/eda-review-anonymous.zip")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def should_skip(path):
    relative = path.relative_to(ROOT).as_posix()
    if relative in {"checkpoints/README.md", "checkpoints/README.zh-CN.md"}:
        return False
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if any(relative == prefix or relative.startswith(prefix + "/") for prefix in SKIP_RELATIVE_PREFIXES):
        return True
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    return False


def copy_release(output_dir):
    for source in ROOT.rglob("*"):
        if not source.is_file() or should_skip(source):
            continue
        relative = source.relative_to(ROOT)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def anonymize_metadata(output_dir):
    license_path = output_dir / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8")
    license_text = license_text.replace(
        "Copyright (c) 2026 Anonymous Author, Anonymous Author, Anonymous Author, and Anonymous Author",
        "Copyright (c) 2026 Anonymous Authors",
    )
    license_path.write_text(license_text, encoding="utf-8")

    for name in ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.zh-CN.md"):
        path = output_dir / name
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "Anonymous Author, Anonymous Author, Anonymous Author, and Anonymous Author",
            "the manuscript authors",
        )
        text = text.replace(
            "Anonymous Author、Anonymous Author、Anonymous Author 和 Anonymous Author",
            "本文作者",
        )
        text = re.sub(
            r"Anonymous Author、Hegui\s+Zhu、Anonymous Author 和 Anonymous Author",
            "本文作者",
            text,
        )
        path.write_text(text, encoding="utf-8")

    pyproject = output_dir / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text = re.sub(
        r"authors = \[.*?\]\nkeywords =",
        'authors = [{name = "Anonymous Authors"}]\nkeywords =',
        text,
        flags=re.DOTALL,
    )
    pyproject.write_text(text, encoding="utf-8")

    citation = output_dir / "CITATION.cff"
    citation.write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'message: "Citation metadata will be completed after peer review."',
                'title: "Enhanced Deformation Attack Reproducibility Package"',
                "type: software",
                'version: "review-v1"',
                "license: MIT",
                "authors:",
                "  - name: Anonymous Authors",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Scrub any remaining exact author-name occurrences, including this builder
    # script's identity checklist, so referenced tools remain present and usable.
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name in AUTHOR_NAMES:
            text = text.replace(name, "Anonymous Author")
        path.write_text(text, encoding="utf-8")

    builder = output_dir / "tools" / "build_anonymous_snapshot.py"
    builder_text = builder.read_text(encoding="utf-8")
    builder_text = re.sub(
        r"AUTHOR_NAMES = \(.*?\)\nSKIP_DIRS =",
        "AUTHOR_NAMES = ()\nSKIP_DIRS =",
        builder_text,
        flags=re.DOTALL,
    )
    builder.write_text(builder_text, encoding="utf-8")


def scan_anonymity(output_dir):
    errors = []
    secret = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}")
    windows_user = re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+")
    linux_user = re.compile(r"/(?:root|home/[^/\s]+)/")
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        normalized_text = re.sub(r"\s+", " ", text)
        for name in AUTHOR_NAMES:
            if name.lower() in normalized_text.lower():
                errors.append(f"author identity in {path.relative_to(output_dir)}: {name}")
        for label, pattern in (
            ("secret", secret),
            ("Windows user path", windows_user),
            ("Linux user path", linux_user),
        ):
            if pattern.search(text):
                errors.append(f"{label} in {path.relative_to(output_dir)}")
    return errors


def write_reviewer_files(output_dir):
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "snapshot_version": "review-v1",
        "source_package_version": "1.0.0",
        "generated_at_utc": generated_at,
        "identity_status": "anonymized",
        "dataset_images_included": False,
        "model_weights_included": False,
        "experiment_outputs_included": False,
        "api_credentials_included": False,
    }
    (output_dir / "REVIEW_SNAPSHOT.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "REVIEWER_QUICKSTART.md").write_text(
        """# Anonymous Reviewer Quick Start

[English](REVIEWER_QUICKSTART.md) | [Simplified Chinese](REVIEWER_QUICKSTART.zh-CN.md)

This is the identity-scrubbed `review-v1` snapshot corresponding to the revised
manuscript. Dataset images, model weights, generated outputs, provider responses,
and API credentials are intentionally excluded. Their preparation is documented
in [`data/README.md`](data/README.md) and
[`checkpoints/README.md`](checkpoints/README.md).

## 1. Install

```bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
```

## 2. Prepare the main dataset

Place the 1,000 NIPS 2017 ImageNet-Compatible images in `data/images/`. The
bundled `data/labels.csv` provides `filename`, `label`, and `targeted_label`.

## 3. Reproduce the main EDA evaluation

```bash
python experiments/run_multi_source.py \\
  --mode both --attack eda \\
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \\
  --input_dir ./data --output_root ./outputs/main/eda \\
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

The paper setting is `epsilon=16/255`, `T=10`, `alpha=1.6/255`, `mu=1`, and
seed 42. `num_warping=25` is the number of transformed views averaged per attack
iteration, not the number of attack iterations.

## 4. Reproduce revision experiments

Commands for multiple budgets, targeted transfer, five-seed stability,
RobustBench evaluation, perceptual quality, generalization, ablations, and VLM
aggregation are listed in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and
[`docs/SCRIPT_INDEX.md`](docs/SCRIPT_INDEX.md). The VLM workflow uses paid
providers and is optional; no additional provider calls are required to inspect
the included evaluation and aggregation code.

## 5. Output and verification

Generated files are written under `outputs/` and are excluded from version
control. Run `python tools/check_release.py` before comparing or redistributing
the snapshot. See [`README.md`](README.md) for the complete protocol.
""",
        encoding="utf-8",
    )

    (output_dir / "REVIEWER_QUICKSTART.zh-CN.md").write_text(
        """# 匿名审稿人快速入门

[English](REVIEWER_QUICKSTART.md) | [简体中文](REVIEWER_QUICKSTART.zh-CN.md)

这是与返修稿对应的去身份化 `review-v1` 快照。数据集图像、模型权重、实验输出、
服务商原始响应和 API 密钥均未包含；其准备方法见
[`data/README.zh-CN.md`](data/README.zh-CN.md) 和
[`checkpoints/README.zh-CN.md`](checkpoints/README.zh-CN.md)。

## 1. 安装

```bash
conda env create -f environment.yml
conda activate eda-repro
pip install -e .
python tools/check_release.py
```

## 2. 准备主数据集

将 NIPS 2017 ImageNet-Compatible Dataset 的 1,000 张图像放入
`data/images/`。随包提供的 `data/labels.csv` 包含 `filename`、`label` 和
`targeted_label`。

## 3. 复现 EDA 主实验

```bash
python experiments/run_multi_source.py \\
  --mode both --attack eda \\
  --sources resnet18,inception_v3,inception_v4,inception_resnet_v2 \\
  --input_dir ./data --output_root ./outputs/main/eda \\
  --noise_scale 0.45 --num_warping 25 --GPU_ID 0
```

论文主设置为 `epsilon=16/255`、`T=10`、`alpha=1.6/255`、`mu=1` 和随机种子
42。`num_warping=25` 表示每次攻击迭代中参与梯度平均的变换视图数，不是攻击迭代数。

## 4. 复现返修新增实验

多预算、目标迁移、五随机种子、RobustBench、感知质量、跨数据集泛化、消融和 VLM
聚合命令见 [`docs/EXPERIMENTS.zh-CN.md`](docs/EXPERIMENTS.zh-CN.md) 与
[`docs/SCRIPT_INDEX.zh-CN.md`](docs/SCRIPT_INDEX.zh-CN.md)。VLM 工作流会产生
外部服务费用，因此属于可选流程；检查随包代码和聚合协议不需要再次调用服务商。

## 5. 输出与核验

生成结果写入 `outputs/`，且默认不进入版本控制。比较或重新分发快照前运行
`python tools/check_release.py`。完整协议见 [`README.zh-CN.md`](README.zh-CN.md)。
""",
        encoding="utf-8",
    )


def write_archive(output_dir, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, Path(output_dir.name) / path.relative_to(output_dir))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return digest, checksum


def write_upload_instructions(output_dir, archive, checksum):
    instructions = archive.parent / "UPLOAD_INSTRUCTIONS.txt"
    instructions.write_text(
        "\n".join(
            [
                "Anonymous review upload instructions",
                "====================================",
                "",
                f"Anonymous repository source: contents of {output_dir.name}/",
                f"Supplementary archive: {archive.name}",
                f"Checksum: {checksum.name}",
                "",
                "Do not upload the author-maintained EDA_reproducibility_package",
                "directory to an anonymous service. It contains public-release",
                "metadata. Use only the generated anonymous directory or ZIP.",
                "",
                "For an anonymous read-only mirror, first commit only the contents",
                "of the generated anonymous directory to a dedicated private GitHub",
                "repository. Create the mirror from a fixed commit and verify the",
                "anonymous URL in a signed-out browser before manuscript submission.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    instructions_zh = archive.parent / "UPLOAD_INSTRUCTIONS.zh-CN.txt"
    instructions_zh.write_text(
        "\n".join(
            [
                "匿名审稿上传说明",
                "================",
                "",
                f"匿名仓库来源：{output_dir.name}/ 目录中的内容",
                f"随稿补充压缩包：{archive.name}",
                f"校验文件：{checksum.name}",
                "",
                "不要把作者维护的 EDA_reproducibility_package 原目录上传到",
                "匿名服务；其中包含用于接收后公开的作者元数据。匿名评审只使用",
                "生成的匿名目录或 ZIP。",
                "",
                "使用匿名只读镜像时，应先把生成的匿名目录内容单独提交到一个",
                "专用 Private GitHub 仓库，再从固定 commit 创建镜像。返修提交前",
                "必须在退出登录的浏览器中验证匿名链接和下载内容。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return instructions, instructions_zh


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    archive = Path(args.archive).resolve()
    if ROOT == output_dir or ROOT in output_dir.parents and "dist" not in output_dir.parts:
        raise ValueError("Output must be a dedicated directory outside the source tree or under dist/.")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir, onerror=remove_readonly)
    if archive.exists():
        if not args.force:
            raise FileExistsError(f"Archive exists; pass --force to replace it: {archive}")
        archive.unlink()

    copy_release(output_dir)
    anonymize_metadata(output_dir)
    write_reviewer_files(output_dir)
    errors = scan_anonymity(output_dir)
    if errors:
        print("Anonymous snapshot check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    digest, checksum = write_archive(output_dir, archive)
    instructions, instructions_zh = write_upload_instructions(output_dir, archive, checksum)
    print(f"Anonymous directory: {output_dir}")
    print(f"Archive: {archive}")
    print(f"SHA-256: {digest}")
    print(f"Checksum file: {checksum}")
    print(f"Upload instructions: {instructions}")
    print(f"Chinese upload instructions: {instructions_zh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
