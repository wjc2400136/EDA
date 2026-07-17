# VLM Evaluation Protocol

[English](VLM_EVALUATION.md) | [简体中文](VLM_EVALUATION.zh-CN.md)

This optional evaluation is provider-dependent and potentially expensive. It is
an empirical label-conditioned recognition study, not a certified robustness
test.

## Inputs

The deterministic subset contains the first 100 ImageNet-Compatible filenames
after lexicographic sorting. `data/vlm_first100_manifest.csv` records filename,
ground-truth label text, and target label text. Every method uses the same image
identities and labels.

```text
vlm_eval/images/
|-- clean/001_<filename>.png
|-- l2t/001_<filename>.png
|-- bsr/001_<filename>.png
|-- decowa/001_<filename>.png
|-- ops/001_<filename>.png
|-- sid/001_<filename>.png
`-- eda/001_<filename>.png
```

Build the tree with `experiments/vlm/prepare_vlm_subset.py` after all six attack
directories are available.

## Prompt and Metric

Each stateless request contains one image and a fixed label-conditioned prompt.
The exact prompt is constructed by `build_prompt_text()` in `run_vlm_batch.py`
and requests only `{"match": 0 or 1, "confidence": 1 to 10}`.

For image `i`, VLM `v`, and evaluation `t`:

```text
s(i,v,t) = match(i,v,t) * confidence(i,v,t) / 10.
```

Average over successfully evaluated images. Never impute provider rejections;
report the valid denominator and mark missing entries.

## API Credentials

Never place keys in code, Markdown, committed shell scripts, or results. Export
a provider-specific variable:

```bash
export OPENAI_API_KEY='REDACTED'
# or
export DASHSCOPE_API_KEY='REDACTED'
```

Run an OpenAI-compatible endpoint with the exact provider model ID:

```bash
python experiments/vlm/run_vlm_batch.py \
  --base-url https://provider.example.com/v1 \
  --model PROVIDER_MODEL_ID \
  --image-dir ./vlm_eval/images \
  --prompt-file ./vlm_eval/prompts.txt \
  --results-dir ./outputs/vlm/PROVIDER_MODEL_ID \
  --workers 1
```

Record provider, access route, model ID, access date, temperature, top-p support,
and whether the route was an API or an isolated interactive task.

## Existing Five-evaluation Protocol

The existing results use temperatures `0.65, 0.80, 0.95, 1.10, 1.25` and
`top-p=0.9` when supported. These are five temperature-varying evaluations, not
five identical-condition random-seed repetitions. Therefore:

- report the standard deviation as variation across five evaluation settings;
- do not claim that it isolates provider stochasticity;
- state that unsupported parameters remain at provider defaults;
- emphasize within-model clean-to-adversarial reductions, macro averages, and
  per-model consistency.

If a provider rejects an image, retain the rejection record, do not substitute
another image, and report the model, prompt index, error, and valid sample count.

After all provider runs, create the consolidated summary:

```bash
python experiments/vlm/summarize_vlm_results.py \
  --model-result MODEL_A=./outputs/vlm/MODEL_A \
  --model-result MODEL_B=./outputs/vlm/MODEL_B \
  --output ./outputs/vlm/vlm_summary.csv
```
