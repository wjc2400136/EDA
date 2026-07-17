"""Build the deterministic 100-image VLM evaluation tree and prompt manifest."""

import argparse
import csv
import shutil
from pathlib import Path


METHODS = ("clean", "l2t", "bsr", "decowa", "ops", "sid", "eda")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="./data/vlm_first100_manifest.csv")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--l2t_dir", required=True)
    parser.add_argument("--bsr_dir", required=True)
    parser.add_argument("--decowa_dir", required=True)
    parser.add_argument("--ops_dir", required=True)
    parser.add_argument("--sid_dir", required=True)
    parser.add_argument("--eda_dir", required=True)
    parser.add_argument("--output_dir", default="./vlm_eval/images")
    parser.add_argument("--prompt_file", default="./vlm_eval/prompts.txt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "true_label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}: {path}")
    return rows


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    if len(rows) != 100:
        raise ValueError(f"Expected 100 manifest rows, found {len(rows)}")

    sources = {method: Path(getattr(args, f"{method}_dir")) for method in METHODS}
    output_dir = Path(args.output_dir)
    missing = []

    for method, source_dir in sources.items():
        destination = output_dir / method
        destination.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows, 1):
            source = source_dir / row["filename"]
            target = destination / f"{index:03d}_{row['filename']}"
            if not source.is_file():
                missing.append(str(source))
                continue
            if target.exists() and not args.overwrite:
                continue
            shutil.copy2(source, target)

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} required images. First entries:\n{preview}"
        )

    prompt_file = Path(args.prompt_file)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with prompt_file.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            handle.write(f"Prompt #{index:03d}\n")
            handle.write(f"Filename: {row['filename']}\n")
            handle.write(f"Ground-truth label: {row['true_label']}\n")
            handle.write("VLM response records:\n")
            for method in METHODS:
                handle.write(f"{method}:\n\n")

    print(f"Prepared {len(rows)} images for {len(METHODS)} methods in {output_dir}")
    print(f"Prompt manifest: {prompt_file}")


if __name__ == "__main__":
    main()
