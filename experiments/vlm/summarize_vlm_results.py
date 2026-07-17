"""Summarize strict VLM trial records into per-model and macro-average CSV rows."""

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = ("clean", "l2t", "bsr", "decowa", "ops", "sid", "eda")
TRIALS = range(1, 6)
PROMPTS = range(1, 101)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-result",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Repeat once per VLM result directory.",
    )
    parser.add_argument("--output", default="./outputs/vlm/vlm_summary.csv")
    return parser.parse_args()


def parse_model_result(value):
    if "=" not in value:
        raise ValueError(f"Expected NAME=DIR, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"Expected NAME=DIR, got: {value}")
    return name.strip(), Path(raw_path).expanduser()


def extract_score(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    response = obj.get("response", obj) if isinstance(obj, dict) else None
    if not isinstance(response, dict):
        return None
    match = response.get("match")
    confidence = response.get("confidence")
    if match not in (0, 1) or not isinstance(confidence, (int, float)):
        return None
    if not 1 <= float(confidence) <= 10:
        return None
    return float(match) * float(confidence) / 10.0


def aggregate_score(result_dir, prompt, method, trial):
    path = result_dir / f"{prompt:03d}_{method}.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(records, list) or len(records) < trial:
        return None
    record = records[trial - 1]
    response = record.get("response", record) if isinstance(record, dict) else None
    if not isinstance(response, dict):
        return None
    match = response.get("match")
    confidence = response.get("confidence")
    if match not in (0, 1) or not isinstance(confidence, (int, float)):
        return None
    if not 1 <= float(confidence) <= 10:
        return None
    return float(match) * float(confidence) / 10.0


def trial_scores(result_dir, method, trial):
    scores = []
    for prompt in PROMPTS:
        path = result_dir / f"prompt_{prompt:03d}" / method / f"trial_{trial:02d}.json"
        score = extract_score(path)
        if score is None:
            score = aggregate_score(result_dir, prompt, method, trial)
        if score is not None:
            scores.append(score)
    return scores


def summarize_model(model, result_dir):
    rows = []
    trial_means_by_method = {}
    for method in METHODS:
        trial_means = []
        trial_counts = []
        for trial in TRIALS:
            scores = trial_scores(result_dir, method, trial)
            trial_counts.append(len(scores))
            if scores:
                trial_means.append(sum(scores) / len(scores))
        trial_means_by_method[method] = trial_means
        rows.append(
            {
                "model": model,
                "method": method,
                "mean_percent": 100.0 * statistics.fmean(trial_means) if trial_means else "",
                "sample_std_percent": 100.0 * statistics.stdev(trial_means) if len(trial_means) > 1 else "",
                "valid_images_min": min(trial_counts) if trial_counts else 0,
                "valid_images_max": max(trial_counts) if trial_counts else 0,
                "valid_trials": len(trial_means),
            }
        )
    return rows, trial_means_by_method


def main():
    args = parse_args()
    all_rows = []
    model_trials = {}
    for spec in args.model_result:
        model, result_dir = parse_model_result(spec)
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")
        rows, trial_means = summarize_model(model, result_dir)
        all_rows.extend(rows)
        model_trials[model] = trial_means

    for method in METHODS:
        macro_trial_means = []
        for trial_index in range(5):
            values = [
                methods[method][trial_index]
                for methods in model_trials.values()
                if len(methods[method]) == 5
            ]
            if values:
                macro_trial_means.append(statistics.fmean(values))
        all_rows.append(
            {
                "model": "MACRO_AVERAGE",
                "method": method,
                "mean_percent": 100.0 * statistics.fmean(macro_trial_means) if macro_trial_means else "",
                "sample_std_percent": 100.0 * statistics.stdev(macro_trial_means) if len(macro_trial_means) > 1 else "",
                "valid_images_min": "",
                "valid_images_max": "",
                "valid_trials": len(macro_trial_means),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "method",
        "mean_percent",
        "sample_std_percent",
        "valid_images_min",
        "valid_images_max",
        "valid_trials",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {output}")


if __name__ == "__main__":
    main()
