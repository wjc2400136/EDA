r"""
Batch VLM evaluator: calls any OpenAI-compatible vision API for all images.

Usage:
    python experiments/vlm/run_vlm_batch.py \
        --base-url https://api.example.com/v1 \
        --model PROVIDER_MODEL_ID \
        --image-dir ./vlm_eval/images \
        --results-dir ./outputs/vlm/PROVIDER_MODEL_ID \
        --prompt-file ./vlm_eval/prompts.txt

Supports:
    - DashScope (qwen-vl models)
    - OpenAI-compatible APIs (any provider with vision)
    - Any model that accepts image + text and returns JSON

For DashScope (Qwen VL models):
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
    --model qwen-vl-max   (or the exact Qwen 3.6 Plus model name)

For Zhipu AI (GLM models):
    --base-url https://open.bigmodel.cn/api/paas/v4
    --model glm-4v-plus

Protocol:
    - 100 independent prompt sessions are executed in prompt-file order.
    - Each session evaluates all seven methods:
      clean / l2t / bsr / decowa / ops / sid / eda.
    - Each method gets five stateless calls. Every trial re-reads the image,
      sends only that image and the fixed prompt, and writes one JSON file.
    - Per-trial JSON files are the primary records. Legacy per-method aggregate
      files are also written so older result directories remain resumable.
    - The prompt text file is rebuilt only after strict verification finds
      3500 JSON responses (100 prompts * 7 methods * 5 trials).
"""

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Install with: pip install openai")
    sys.exit(1)

# --- Constants ---
METHODS = ["clean", "l2t", "bsr", "decowa", "ops", "sid", "eda"]
NUM_TRIALS = 5
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_prompts(txt_path):
    """Parse the prompt file to extract entries."""
    import re
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    pattern = re.compile(
        r"Prompt #(\d+)\s*\nFilename:\s*(\S+)\s*\nGround-truth label:\s*(.+?)\s*\n",
        re.DOTALL
    )
    for m in pattern.finditer(content):
        entries.append({
            "prompt_num": int(m.group(1)),
            "filename": m.group(2),
            "label": m.group(3).strip(),
        })
    return entries


def resolve_existing_or_script_relative(raw_path):
    """Resolve CLI paths while preserving old commands run from this folder."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return SCRIPT_DIR / path


def resolve_output_dir(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return SCRIPT_DIR / path


def build_prompt_text(label):
    """Build the system prompt for the VLM."""
    return (
        f"You are a strict image classification expert system.\n\n"
        f"Task:\n"
        f"Determine whether the main visual content of the given image belongs to the following label:\n\n"
        f"Label: {label}\n\n"
        f"Instructions:\n"
        f"1. Judge only from the visual content of the image.\n"
        f'2. Return "match": 1 if the main object or scene clearly matches the given label.\n'
        f'3. Return "match": 0 if the image does not match the given label.\n'
        f"4. Return a confidence score from 1 to 10.\n"
        f"5. Do not output any text other than the JSON object.\n\n"
        f"Required JSON format:\n"
        f'{{"match": 1 or 0, "confidence": integer between 1 and 10}}'
    )


def get_response_text(response):
    """Extract assistant text content and report empty responses clearly."""
    if not getattr(response, "choices", None):
        raise ValueError("Empty API response: no choices returned")

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(text)
        content = "".join(parts)

    if content is None or not str(content).strip():
        finish_reason = getattr(choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None) if message is not None else None
        raise ValueError(
            "Empty model response content "
            f"(finish_reason={finish_reason}, refusal={refusal})"
        )

    return str(content)


def strip_markdown_fence(text):
    """Remove common markdown code fences without changing the prompt protocol."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def parse_response_json(output_text):
    """
    Parse the first JSON object in a model response.

    The experimental prompt still asks for exactly one JSON object. This parser
    only tolerates provider/model formatting noise, such as code fences or text
    after the first JSON object, so that the recorded response remains the
    parsed match/confidence pair for the same fixed prompt.
    """
    cleaned = strip_markdown_fence(output_text)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(cleaned[start:])

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}: {output_text}")

    match_val = result.get("match")
    conf_val = result.get("confidence")
    if match_val in (0, 1) and isinstance(conf_val, (int, float)) and 1 <= int(conf_val) <= 10:
        return {"match": int(match_val), "confidence": int(conf_val)}
    raise ValueError(f"Unexpected response: {output_text}")


def call_vlm(client, model, image_base64, prompt_text, trial, temperature=0.7, top_p=0.9, max_tokens=256):
    """
    Call a vision model via OpenAI-compatible API.
    Returns {"match": int, "confidence": int} or None.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    }
                },
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ]
        }
    ]

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        request_kwargs["top_p"] = top_p

    response = client.chat.completions.create(**request_kwargs)

    output_text = get_response_text(response)
    return parse_response_json(output_text)


def trial_result_file(results_dir, prompt_num, method, trial):
    return Path(results_dir) / f"prompt_{prompt_num:03d}" / method / f"trial_{trial:02d}.json"


def aggregate_result_file(results_dir, prompt_num, method):
    return Path(results_dir) / f"{prompt_num:03d}_{method}.json"


def is_valid_response(obj):
    return (
        isinstance(obj, dict)
        and obj.get("match") in (0, 1)
        and isinstance(obj.get("confidence"), int)
        and 1 <= obj["confidence"] <= 10
    )


def read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_response(obj):
    """Accept both strict trial records and legacy bare response objects."""
    if is_valid_response(obj):
        return {"match": int(obj["match"]), "confidence": int(obj["confidence"])}
    if isinstance(obj, dict) and is_valid_response(obj.get("response")):
        response = obj["response"]
        return {"match": int(response["match"]), "confidence": int(response["confidence"])}
    return None


def read_trial_response(results_dir, prompt_num, method, trial):
    obj = read_json_file(trial_result_file(results_dir, prompt_num, method, trial))
    return extract_response(obj)


def read_aggregate_responses(results_dir, prompt_num, method):
    data = read_json_file(aggregate_result_file(results_dir, prompt_num, method))
    if not isinstance(data, list) or len(data) != NUM_TRIALS:
        return None
    responses = [extract_response(obj) for obj in data]
    if all(response is not None for response in responses):
        return responses
    return None


def read_aggregate_trial_response(results_dir, prompt_num, method, trial):
    data = read_json_file(aggregate_result_file(results_dir, prompt_num, method))
    if not isinstance(data, list) or len(data) < trial:
        return None
    return extract_response(data[trial - 1])


def read_existing_trial_or_aggregate(results_dir, prompt_num, method, trial):
    response = read_trial_response(results_dir, prompt_num, method, trial)
    if response is not None:
        return response
    return read_aggregate_trial_response(results_dir, prompt_num, method, trial)


def write_trial_record(args, entry, method, trial, image_path, temperature, response):
    prompt_num = entry["prompt_num"]
    out_path = trial_result_file(args.results_dir, prompt_num, method, trial)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_num": prompt_num,
        "filename": entry["filename"],
        "method": method,
        "trial": trial,
        "model": args.model,
        "base_url": args.base_url,
        "image_path": str(image_path),
        "label": entry["label"],
        "temperature": temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "prompt_text": build_prompt_text(entry["label"]),
        "response": response,
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def materialize_trial_records_from_aggregate(args, entry, method, responses, image_path):
    for trial, response in enumerate(responses, start=1):
        out_path = trial_result_file(args.results_dir, entry["prompt_num"], method, trial)
        if out_path.exists() and read_trial_response(args.results_dir, entry["prompt_num"], method, trial):
            continue
        temperature = args.temperature_start + trial * args.temperature_step
        write_trial_record(args, entry, method, trial, image_path, temperature, response)


def write_aggregate_responses(results_dir, prompt_num, method, responses):
    aggregate_result_file(results_dir, prompt_num, method).write_text(
        json.dumps(responses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_complete_responses(results_dir, prompt_num, method):
    trial_responses = [
        read_trial_response(results_dir, prompt_num, method, trial)
        for trial in range(1, NUM_TRIALS + 1)
    ]
    if all(response is not None for response in trial_responses):
        return trial_responses
    return read_aggregate_responses(results_dir, prompt_num, method)


def process_one(args, entry, method):
    """Process one image-method pair under a stateless five-trial protocol."""
    prompt_num = entry["prompt_num"]
    filename_no_prefix = entry["filename"]
    # Actual filenames are prefixed like "001_000b7d55b6184b08.png"
    filename = f"{prompt_num:03d}_{filename_no_prefix}"
    label = entry["label"]

    image_path = Path(args.image_dir) / method / filename
    result_file = aggregate_result_file(args.results_dir, prompt_num, method)

    # Check if image exists
    if not image_path.exists():
        print(f"  [#{prompt_num:03d}/{method}] SKIP: image not found at {image_path}")
        return False

    # Check if already complete. If only a legacy aggregate exists, materialize
    # the strict per-trial files so verification is based on 3500 JSON records.
    existing = get_complete_responses(args.results_dir, prompt_num, method)
    if existing is not None:
        materialize_trial_records_from_aggregate(args, entry, method, existing, image_path)
        write_aggregate_responses(args.results_dir, prompt_num, method, existing)
        print(f"  [#{prompt_num:03d}/{method}] SKIP: already complete")
        return True

    prompt_text = build_prompt_text(label)

    responses = []
    for trial in range(1, NUM_TRIALS + 1):
        temp = args.temperature_start + trial * args.temperature_step
        existing_trial = read_existing_trial_or_aggregate(args.results_dir, prompt_num, method, trial)
        if existing_trial is not None:
            responses.append(existing_trial)
            write_trial_record(args, entry, method, trial, image_path, temp, existing_trial)
            print(f"    Trial {trial}: SKIP existing {existing_trial}")
            continue

        # A chat-completions request is stateless: each trial carries only the
        # current image and prompt. Re-reading the image prevents hidden reuse.
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        for retry in range(MAX_RETRIES):
            try:
                client = OpenAI(
                    api_key=args.api_key,
                    base_url=args.base_url,
                )
                result = call_vlm(
                    client,
                    args.model,
                    image_base64,
                    prompt_text,
                    trial,
                    temperature=temp,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                responses.append(result)
                write_trial_record(args, entry, method, trial, image_path, temp, result)
                print(f"    Trial {trial}: {result}")
                break
            except Exception as e:
                print(f"    Trial {trial} attempt {retry+1}/{MAX_RETRIES} failed: {e}")
                if retry < MAX_RETRIES - 1:
                    time.sleep(2 ** retry)
        else:
            print(f"    Trial {trial}: ALL RETRIES FAILED")
            # Write what we have so far (partial)
            if responses:
                write_aggregate_responses(args.results_dir, prompt_num, method, responses)
            return False

        time.sleep(RATE_LIMIT_DELAY)

    # Write all 5 responses
    write_aggregate_responses(args.results_dir, prompt_num, method, responses)
    print(f"  [#{prompt_num:03d}/{method}] DONE -> {result_file.name}")

    return True


def process_prompt_session(args, entry):
    """Independent worker/session for one prompt and all seven methods."""
    prompt_num = entry["prompt_num"]
    print(f"\n{'='*60}")
    print(f"Prompt session #{prompt_num:03d}: {entry['filename']}")
    print(f"{'='*60}")

    completed = 0
    for method in METHODS:
        try:
            if process_one(args, entry, method):
                completed += 1
        except Exception as e:
            print(f"  [#{prompt_num:03d}/{method}] FATAL: {e}")
            traceback.print_exc()
    return prompt_num, completed


def rebuild_txt(prompt_file, results_dir, methods):
    """Rebuild the prompt text file with filled-in JSON responses."""
    import re

    # Backup
    backup = str(prompt_file) + ".before_batch"
    if not Path(backup).exists():
        Path(backup).write_text(prompt_file.read_text(encoding="utf-8"), encoding="utf-8")

    content = prompt_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    prompt_num = None
    in_records = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Prompt #"):
            m = re.search(r"Prompt #(\d+)", stripped)
            if m:
                prompt_num = int(m.group(1))
            in_records = False
            new_lines.append(line)
            continue

        if stripped == "VLM response records:":
            in_records = True
            new_lines.append(line)
            continue

        if not in_records:
            new_lines.append(line)
            continue

        # Check if this is a method header
        method_name = stripped.rstrip(":")
        if method_name in methods and stripped.endswith(":"):
            new_lines.append(line)
            # Load responses
            data = get_complete_responses(results_dir, prompt_num, method_name)
            if data is not None:
                for obj in data:
                    new_lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

            if method_name == "eda":
                new_lines.extend([""] * 6)
            else:
                new_lines.append("")
            continue

        continue  # skip existing response lines

    prompt_file.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Rebuilt: {prompt_file}")


def verify_results(entries, results_dir, methods):
    complete_pairs = 0
    complete_trial_records = 0
    missing = []
    for entry in entries:
        prompt_num = entry["prompt_num"]
        for method in methods:
            responses = get_complete_responses(results_dir, prompt_num, method)
            if responses is None:
                missing.append(f"{prompt_num:03d}/{method}")
                continue
            complete_pairs += 1
            complete_trial_records += len(responses)
    return complete_pairs, complete_trial_records, missing


def main():
    parser = argparse.ArgumentParser(description="Batch VLM evaluator")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or set OPENAI_API_KEY / DASHSCOPE_API_KEY)")
    parser.add_argument("--base-url", type=str, required=True,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. qwen-vl-max, glm-4v-plus)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Base dir containing clean/, l2t/, bsr/, decowa/, ops/, sid/, eda/")
    parser.add_argument("--prompt-file", type=str, required=True,
                        help="Path to vlm_prompts_first100_*.txt")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Dir to store individual JSON result files")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent prompt sessions to run concurrently (default: 1)")
    parser.add_argument("--temperature-start", type=float, default=0.5,
                        help="Trial temperature formula start; temp=start+trial*step (default: 0.5)")
    parser.add_argument("--temperature-step", type=float, default=0.15,
                        help="Trial temperature formula step (default: 0.15)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Optional nucleus sampling top_p value (default: 0.9)")
    parser.add_argument("--no-top-p", action="store_true",
                        help="Do not send top_p. Use for providers/models that reject top_p.")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum output tokens for the JSON response (default: 256)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify and rebuild from existing JSON files; do not call the API")
    args = parser.parse_args()

    if args.no_top_p:
        args.top_p = None

    global RATE_LIMIT_DELAY
    RATE_LIMIT_DELAY = args.delay

    # Resolve API key. Verification-only mode never calls the API.
    args.api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not args.verify_only and not args.api_key:
        print("ERROR: No API key. Use --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        sys.exit(1)

    args.image_dir = resolve_existing_or_script_relative(args.image_dir)
    args.prompt_file = resolve_existing_or_script_relative(args.prompt_file)
    args.results_dir = resolve_output_dir(args.results_dir)

    if not args.prompt_file.exists():
        print(f"ERROR: prompt file not found: {args.prompt_file}")
        sys.exit(1)
    if not args.image_dir.exists():
        print(f"ERROR: image dir not found: {args.image_dir}")
        sys.exit(1)

    # Create dirs
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Parse prompts
    entries = parse_prompts(args.prompt_file)
    total = len(entries) * len(METHODS)
    print(f"Prompts: {len(entries)}")
    print(f"Methods: {METHODS}")
    print(f"Total tasks: {total}")
    print(f"Expected JSON responses: {total * NUM_TRIALS}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Image dir: {args.image_dir}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Results dir: {args.results_dir}")
    print(f"Workers: {args.workers}")
    print(f"Temperature schedule: start={args.temperature_start}, step={args.temperature_step}")
    print(f"Top-p: {'disabled' if args.top_p is None else args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Process
    if not args.verify_only:
        if args.workers <= 1:
            for entry in entries:
                process_prompt_session(args, entry)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_prompt_session, args, entry) for entry in entries]
                for future in as_completed(futures):
                    prompt_num, completed = future.result()
                    print(f"Prompt session #{prompt_num:03d} completed methods: {completed}/{len(METHODS)}")

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    completed, json_responses, missing = verify_results(entries, args.results_dir, METHODS)

    print(f"Completed: {completed}/{total}")
    print(f"JSON responses: {json_responses}/{total * NUM_TRIALS}")

    if completed == total:
        print("\nAll complete! Rebuilding text file...")
        rebuild_txt(Path(args.prompt_file), args.results_dir, METHODS)
        _, final_json_responses, _ = verify_results(entries, args.results_dir, METHODS)
        print(f"Final strict verification: {final_json_responses}/{total * NUM_TRIALS} JSON responses")
    else:
        print(f"\n{total - completed} tasks remain. Re-run to fill them.")
        if missing:
            print("First missing/incomplete tasks:")
            for item in missing[:20]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
