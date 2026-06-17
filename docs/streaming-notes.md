# Streaming Notes

## Non-Streaming vs Streaming

Both non-streaming and streaming requests use the `/v1/chat/completions` endpoint.

The difference is how the response is returned.

In non-streaming mode, the client waits until the full response is complete and receives one JSON object.

In streaming mode, the client receives multiple chunks as the model generates output.

## Why Time to First Content Chunk Matters

For user-facing inference systems, total response time is not the only important metric.

A streamed response can feel faster because the user starts seeing output before the full response is complete.

This lab uses `time_to_first_content_chunk_ms` as a simple first metric for user-perceived responsiveness.

## Initial Metrics

The first streaming benchmark captures:

* time to first content chunk
* total elapsed time
* number of content chunks
* reconstructed output character count
* finish reason

## Notes

These scripts currently measure client-observed timing from a simple Python HTTP client.

They do not yet measure:

* server-side queue time
* prefill time
* decode time
* GPU utilization during the request
* p95 or p99 latency
* sustained concurrent streaming behavior

Those may be added in future milestones.

