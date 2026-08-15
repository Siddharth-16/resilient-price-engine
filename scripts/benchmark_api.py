from __future__ import annotations

import argparse
import statistics
import time

from fastapi.testclient import TestClient

from api.main import app

SAMPLE_REQUEST = {
    "manufacturer": "ford",
    "model": "f-150",
    "fuel": "gas",
    "title_status": "clean",
    "transmission": "automatic",
    "drive": "4wd",
    "type": "truck",
    "paint_color": "white",
    "state": "ca",
    "odometer": 90000,
    "car_age": 8,
}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
    return ordered[index]


def benchmark(requests: int, warmup: int) -> None:
    latencies_ms: list[float] = []

    with TestClient(app) as client:
        for _ in range(warmup):
            response = client.post("/predict", json=SAMPLE_REQUEST)
            response.raise_for_status()

        wall_start = time.perf_counter()
        for _ in range(requests):
            start = time.perf_counter()
            response = client.post("/predict", json=SAMPLE_REQUEST)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - start) * 1000)
        elapsed = time.perf_counter() - wall_start

    print(f"requests={requests}")
    print(f"mean_latency_ms={statistics.mean(latencies_ms):.2f}")
    print(f"p50_latency_ms={percentile(latencies_ms, 0.50):.2f}")
    print(f"p95_latency_ms={percentile(latencies_ms, 0.95):.2f}")
    print(f"throughput_requests_per_second={requests / elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.requests <= 0 or args.warmup < 0:
        raise SystemExit("--requests must be > 0 and --warmup must be >= 0")

    benchmark(args.requests, args.warmup)
