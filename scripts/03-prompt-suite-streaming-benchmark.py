#!/usr/bin/env python3

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
PROMPTS_FILE = Path(os.environ.get("PROMPTS_FILE", "benchmarks/prompts.jsonl"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "benchmark-results"))

RUN_ID = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
RESULTS_FILE = Path(
    os.environ.get(
        "RESULTS_FILE",
        str(RESULTS_DIR / f"prompt-suite-streaming-{RUN_ID}.jsonl"),
    )
)

CONTINUE_ON_ERROR = os.environ.get("CONTINUE_ON_ERROR", "false").lower() == "true"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompts(path: Path) -> list[dict]:
    prompts = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if "prompt_id" not in item or "prompt" not in item:
                raise ValueError(
                    f"Line {line_number} must contain both prompt_id and prompt"
                )

            prompts.append(
                {
                    "prompt_id": str(item["prompt_id"]),
                    "prompt": str(item["prompt"]),
                }
            )

    return prompts


def run_streaming_request(prompt_id: str, prompt: str) -> dict:
    url = f"{BASE_URL}/v1/chat/completions"

    request_body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }

    data = json.dumps(request_body).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_content_time = None
    end = None

    content_chunks = 0
    streamed_text_parts = []
    finish_reason = None

    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            if not line.startswith("data: "):
                continue

            payload = line.removeprefix("data: ").strip()

            if payload == "[DONE]":
                break

            event = json.loads(payload)
            choice = event["choices"][0]

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta", {})
            content = delta.get("content")

            if content:
                if first_content_time is None:
                    first_content_time = time.perf_counter()

                content_chunks += 1
                streamed_text_parts.append(content)

    end = time.perf_counter()

    streamed_text = "".join(streamed_text_parts)
    total_elapsed_ms = (end - start) * 1000

    if first_content_time is None:
        time_to_first_content_chunk_ms = None
    else:
        time_to_first_content_chunk_ms = (first_content_time - start) * 1000

    return {
        "timestamp_utc": now_utc_iso(),
        "mode": "streaming",
        "base_url": BASE_URL,
        "model": MODEL,
        "prompt_id": prompt_id,
        "max_tokens": MAX_TOKENS,
        "time_to_first_content_chunk_ms": round(time_to_first_content_chunk_ms, 2)
        if time_to_first_content_chunk_ms is not None
        else None,
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "content_chunks": content_chunks,
        "output_chars": len(streamed_text),
        "finish_reason": finish_reason,
    }


def print_summary(results: list[dict]) -> None:
    if not results:
        print("No successful results to summarize.")
        return

    ttfc_values = [
        row["time_to_first_content_chunk_ms"]
        for row in results
        if row["time_to_first_content_chunk_ms"] is not None
    ]
    total_elapsed_values = [row["total_elapsed_ms"] for row in results]
    chunk_values = [row["content_chunks"] for row in results]
    output_char_values = [row["output_chars"] for row in results]

    summary = {
        "runs": len(results),
        "avg_time_to_first_content_chunk_ms": round(mean(ttfc_values), 2)
        if ttfc_values
        else None,
        "min_time_to_first_content_chunk_ms": min(ttfc_values)
        if ttfc_values
        else None,
        "max_time_to_first_content_chunk_ms": max(ttfc_values)
        if ttfc_values
        else None,
        "avg_total_elapsed_ms": round(mean(total_elapsed_values), 2),
        "min_total_elapsed_ms": min(total_elapsed_values),
        "max_total_elapsed_ms": max(total_elapsed_values),
        "avg_content_chunks": round(mean(chunk_values), 2),
        "avg_output_chars": round(mean(output_char_values), 2),
        "finish_reasons": sorted(
            {row["finish_reason"] for row in results if row["finish_reason"]}
        ),
    }

    print()
    print("Summary from this run:")
    print(json.dumps(summary, indent=2))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running prompt-suite streaming benchmark...")
    print(f"Base URL:          {BASE_URL}")
    print(f"Model:             {MODEL}")
    print(f"Max tokens:        {MAX_TOKENS}")
    print(f"Prompts file:      {PROMPTS_FILE}")
    print(f"Results file:      {RESULTS_FILE}")
    print(f"Continue on error: {CONTINUE_ON_ERROR}")
    print(f"Run ID:            {RUN_ID}")
    print()

    if not PROMPTS_FILE.exists():
        print(f"ERROR: prompts file not found: {PROMPTS_FILE}", file=sys.stderr)
        return 1

    try:
        prompts = load_prompts(PROMPTS_FILE)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not prompts:
        print(f"ERROR: no prompts found in {PROMPTS_FILE}", file=sys.stderr)
        return 1

    print(f"Loaded {len(prompts)} prompts.")
    print()

    successful_results = []
    failed = 0

    for index, item in enumerate(prompts, start=1):
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]

        print(f"[{index}/{len(prompts)}] Running prompt_id={prompt_id}...")

        try:
            result_row = run_streaming_request(prompt_id, prompt)

        except urllib.error.HTTPError as exc:
            failed += 1
            print(f"ERROR: HTTP {exc.code} for prompt_id={prompt_id}", file=sys.stderr)
            print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)

            if not CONTINUE_ON_ERROR:
                return 1
            continue

        except urllib.error.URLError as exc:
            failed += 1
            print(f"ERROR: request failed for prompt_id={prompt_id}: {exc}", file=sys.stderr)

            if not CONTINUE_ON_ERROR:
                return 1
            continue

        except json.JSONDecodeError as exc:
            failed += 1
            print(
                f"ERROR: failed to parse streaming payload for prompt_id={prompt_id}: {exc}",
                file=sys.stderr,
            )

            if not CONTINUE_ON_ERROR:
                return 1
            continue

        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result_row, separators=(",", ":")) + "\n")

        successful_results.append(result_row)

        print(
            "OK: "
            f"ttfc_ms={result_row['time_to_first_content_chunk_ms']} "
            f"total_ms={result_row['total_elapsed_ms']} "
            f"chunks={result_row['content_chunks']} "
            f"chars={result_row['output_chars']} "
            f"finish_reason={result_row['finish_reason']}"
        )

    print()
    print("Prompt-suite streaming benchmark complete.")
    print(f"Successful prompts: {len(successful_results)}")
    print(f"Failed prompts:     {failed}")
    print(f"Results file:       {RESULTS_FILE}")

    print_summary(successful_results)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
