#!/usr/bin/env python3

import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def get_url(path: str) -> tuple[int, str]:
    url = f"{BASE_URL}{path}"

    request = urllib.request.Request(
        url,
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body


def main() -> int:
    print("Checking vLLM server...")
    print(f"Base URL: {BASE_URL}")
    print()

    try:
        status, health_body = get_url("/health")
    except urllib.error.HTTPError as exc:
        print(f"ERROR: /health returned HTTP {exc.code}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach /health: {exc}", file=sys.stderr)
        return 1

    print(f"OK: /health returned HTTP {status}")

    try:
        status, models_body = get_url("/v1/models")
    except urllib.error.HTTPError as exc:
        print(f"ERROR: /v1/models returned HTTP {exc.code}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach /v1/models: {exc}", file=sys.stderr)
        return 1

    print(f"OK: /v1/models returned HTTP {status}")

    try:
        models = json.loads(models_body)
    except json.JSONDecodeError:
        print("WARNING: /v1/models response was not valid JSON")
        print(models_body)
        return 1

    print()
    print("Models:")
    for item in models.get("data", []):
        model_id = item.get("id", "<unknown>")
        print(f"  - {model_id}")

    print()
    print("vLLM server check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
