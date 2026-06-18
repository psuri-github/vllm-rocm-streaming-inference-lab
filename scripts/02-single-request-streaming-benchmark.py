#!/usr/bin/env python3

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
PROMPT_ID = os.environ.get("PROMPT_ID", "manual")
PROMPT = os.environ.get(
    "PROMPT",
    "Explain GPU inference in one short paragraph.",
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "benchmark-results"))
RESULTS_FILE = Path(
    os.environ.get(
        "RESULTS_FILE",
        str(RESULTS_DIR / "single-request-streaming-results.jsonl"),
    )
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/v1/chat/completions"

    request_body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
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

    print("Running single-request streaming benchmark...")
    print(f"Base URL:     {BASE_URL}")
    print(f"Model:        {MODEL}")
    print(f"Prompt ID:    {PROMPT_ID}")
    print(f"Max tokens:   {MAX_TOKENS}")
    print(f"Results file: {RESULTS_FILE}")
    print()

    start = time.perf_counter()
    first_content_time = None
    end = None

    content_chunks = 0
    streamed_text_parts = []
    finish_reason = None

    try:
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

    except urllib.error.HTTPError as exc:
        end = time.perf_counter()
        print(f"ERROR: HTTP {exc.code}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    except urllib.error.URLError as exc:
        end = time.perf_counter()
        print(f"ERROR: request failed: {exc}", file=sys.stderr)
        return 1

    except json.JSONDecodeError as exc:
        end = time.perf_counter()
        print(f"ERROR: failed to parse streaming JSON payload: {exc}", file=sys.stderr)
        return 1

    streamed_text = "".join(streamed_text_parts)
    total_elapsed_ms = (end - start) * 1000

    if first_content_time is None:
        time_to_first_content_chunk_ms = None
    else:
        time_to_first_content_chunk_ms = (first_content_time - start) * 1000

    result_row = {
        "timestamp_utc": now_utc_iso(),
        "mode": "streaming",
        "base_url": BASE_URL,
        "model": MODEL,
        "prompt_id": PROMPT_ID,
        "max_tokens": MAX_TOKENS,
        "time_to_first_content_chunk_ms": round(time_to_first_content_chunk_ms, 2)
        if time_to_first_content_chunk_ms is not None
        else None,
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "content_chunks": content_chunks,
        "output_chars": len(streamed_text),
        "finish_reason": finish_reason,
    }

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result_row, separators=(",", ":")) + "\n")

    print("Streaming benchmark complete.")
    print(f"Time to first content chunk ms: {result_row['time_to_first_content_chunk_ms']}")
    print(f"Total elapsed ms:               {result_row['total_elapsed_ms']}")
    print(f"Content chunks:                 {content_chunks}")
    print(f"Output chars:                   {len(streamed_text)}")
    print(f"Finish reason:                  {finish_reason}")
    print(f"Result appended to:             {RESULTS_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
