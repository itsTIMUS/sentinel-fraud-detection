# Latency Report

## Load Test Results (200 requests)

| Metric | Value |
|---|---|
| p50 (median) | 12.4 ms |
| p95 | 14.7 ms |
| p99 | 33.0 ms |
| min | 11.0 ms |
| max | 72.1 ms |
| mean | 13.2 ms |
| error rate | 0/200 |

## Setup
- Hardware: Windows laptop, single uvicorn worker
- Python 3.11.8, LightGBM 4.3.0
- Includes HTTP round-trip overhead
- Model loaded once at startup, warmed with dummy predict
