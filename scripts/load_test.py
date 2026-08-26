"""Simple load test — measure p50/p95/p99 latency."""

import requests
import time
import numpy as np

API_URL = "http://127.0.0.1:8000/v1/score"

PAYLOAD = {
    "trans_date_trans_time": "2020-06-21 12:14:25",
    "cc_num": 2703186189652095,
    "merchant": "fraud_Kirlin and Sons",
    "category": "personal_care",
    "amt": 500.0,
    "first": "Jeff", "last": "Elliott", "gender": "M",
    "street": "351 Darlene Green", "city": "Columbia", "state": "SC",
    "zip": 29209, "lat": 33.9659, "long": -80.9355,
    "city_pop": 333497, "job": "Mechanical engineer",
    "dob": "1968-03-19",
    "trans_num": "load_test_txn",
    "unix_time": 1371816865,
    "merch_lat": 33.986391, "merch_long": -81.200714,
}

NUM_REQUESTS = 200

print(f"Running {NUM_REQUESTS} requests against {API_URL}...")
print("Make sure the API is running (uvicorn).\n")

# Warmup
for _ in range(5):
    requests.post(API_URL, json=PAYLOAD, timeout=10)

latencies = []
errors = 0

for i in range(NUM_REQUESTS):
    # Vary the transaction slightly
    payload = {**PAYLOAD, "trans_num": f"load_test_{i}", "amt": float(50 + (i * 7) % 5000)}
    
    start = time.perf_counter()
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            latencies.append(elapsed)
        else:
            errors += 1
    except Exception:
        errors += 1

    if (i + 1) % 50 == 0:
        print(f"  ...{i + 1}/{NUM_REQUESTS} done")

latencies = np.array(latencies)

print(f"\n{'='*50}")
print(f"  LOAD TEST RESULTS")
print(f"{'='*50}")
print(f"  Requests:  {NUM_REQUESTS}")
print(f"  Successes: {len(latencies)}")
print(f"  Errors:    {errors}")
print(f"")
print(f"  p50 (median): {np.percentile(latencies, 50):>8.1f} ms")
print(f"  p95:          {np.percentile(latencies, 95):>8.1f} ms")
print(f"  p99:          {np.percentile(latencies, 99):>8.1f} ms")
print(f"  min:          {latencies.min():>8.1f} ms")
print(f"  max:          {latencies.max():>8.1f} ms")
print(f"  mean:         {latencies.mean():>8.1f} ms")
print(f"")
print(f"  Hardware: Windows, Python 3.11, single uvicorn worker")
print(f"  Note: Includes HTTP round-trip overhead (requests library)")

# Save report
report = f"""# Latency Report

## Load Test Results ({NUM_REQUESTS} requests)

| Metric | Value |
|---|---|
| p50 (median) | {np.percentile(latencies, 50):.1f} ms |
| p95 | {np.percentile(latencies, 95):.1f} ms |
| p99 | {np.percentile(latencies, 99):.1f} ms |
| min | {latencies.min():.1f} ms |
| max | {latencies.max():.1f} ms |
| mean | {latencies.mean():.1f} ms |
| error rate | {errors}/{NUM_REQUESTS} |

## Setup
- Hardware: Windows laptop, single uvicorn worker
- Python 3.11.8, LightGBM 4.3.0
- Includes HTTP round-trip overhead
- Model loaded once at startup, warmed with dummy predict
"""

with open("reports/latency.md", "w") as f:
    f.write(report)

print("✅ Report saved to reports/latency.md")