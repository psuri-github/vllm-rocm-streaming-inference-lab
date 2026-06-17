# vLLM ROCm Streaming Inference Lab

This repository explores streaming LLM inference behavior using vLLM on AMD ROCm GPU infrastructure.

The focus is on vLLM’s OpenAI-compatible `/v1/chat/completions` endpoint with `stream: true`, and on measuring user-perceived latency signals such as:

* time to first content chunk
* total response time
* streamed chunk count
* response reconstruction
* finish reason

This is a learning project and not a production benchmark.

## Why Streaming?

In a non-streaming request, the client receives the full response only after generation is complete.

In a streaming request, the client receives partial response chunks as the model generates output. This makes it possible to measure how quickly a user starts seeing content, not just how long the full response takes.

## Initial Scope

The first version of this lab includes:

* a streaming smoke-test script
* a single-request streaming benchmark script
* JSONL result output
* simple timing metrics

Future experiments may include:

* repeated streaming runs
* prompt-suite streaming benchmarks
* concurrent streaming benchmarks
* model comparison
* comparison with non-streaming results

## Assumptions

This lab assumes a vLLM server is already running and reachable through an OpenAI-compatible API endpoint.

Example default endpoint:

```text
http://localhost:8002/v1/chat/completions
```

The default model used by the scripts is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Both the endpoint and model can be overridden using environment variables.

## Scripts

### Streaming smoke test

```bash
BASE_URL=http://localhost:8002 \
MAX_TOKENS=1024 \
./scripts/01-test-chat-completion-streaming.py
```

This script prints streamed content chunks as they arrive and reports basic timing information.

### Single-request streaming benchmark

```bash
BASE_URL=http://localhost:8002 \
MAX_TOKENS=1024 \
./scripts/02-single-request-streaming-benchmark.py
```

This script records one JSONL result row with streaming-specific metrics.

## Metrics Captured

The benchmark script captures:

| Metric                           | Meaning                                                                |
| -------------------------------- | ---------------------------------------------------------------------- |
| `time_to_first_content_chunk_ms` | Time from request start until the first streamed content chunk arrives |
| `total_elapsed_ms`               | Time from request start until the stream completes                     |
| `content_chunks`                 | Number of streamed chunks containing content                           |
| `output_chars`                   | Number of reconstructed output characters                              |
| `finish_reason`                  | Final finish reason reported by the API                                |
| `max_tokens`                     | Configured output-token cap                                            |
| `model`                          | Model requested through the API                                        |

## Example Result Row

```json
{"timestamp_utc":"2026-06-17T00:00:00+00:00","mode":"streaming","base_url":"http://localhost:8002","model":"Qwen/Qwen2.5-0.5B-Instruct","prompt_id":"manual","max_tokens":1024,"time_to_first_content_chunk_ms":123.45,"total_elapsed_ms":456.78,"content_chunks":42,"output_chars":1200,"finish_reason":"stop"}
```

## Relationship to Non-Streaming Benchmarks

This repo is focused on streaming behavior.

The related non-streaming benchmark work lives in the separate `vllm-rocm-inference-lab` project, which focuses on request completion time, capped-output behavior, natural-completion behavior, and mixed-prompt concurrency.

This repo focuses on a different question:

```text
How quickly does the user start seeing useful output?
```
