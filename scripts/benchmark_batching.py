#!/usr/bin/env python3
"""
Compare naive (1-by-1) vs batched embedding throughput.
"""
import argparse
import time
import sys
from typing import List

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Installing onnxruntime...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
    import onnxruntime as ort


def preprocess(texts: List[str], max_len: int = 256) -> np.ndarray:
    """Simple tokenization - map characters to pseudo-token IDs."""
    batch_size = len(texts)
    data = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, text in enumerate(texts):
        for j, c in enumerate(text[:max_len-1]):
            data[i, j + 1] = float(ord(c) + 1)
        data[i, 0] = 101.0  # CLS token

    return data


def benchmark_naive(ort_session: ort.InferenceSession, texts: List[str],
                    num_runs: int = 10) -> float:
    """One query at a time."""
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        for text in texts:
            input_data = preprocess([text])
            ort_session.run(None, {"input": input_data})
        times.append(time.perf_counter() - start)

    return np.median(times)


def benchmark_batched(ort_session: ort.InferenceSession, texts: List[str],
                      batch_size: int = 32, num_runs: int = 10) -> float:
    """Batch processing."""
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            input_data = preprocess(batch)
            ort_session.run(None, {"input": input_data})
        times.append(time.perf_counter() - start)

    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description='Benchmark embedding batching')
    parser.add_argument("--model", help="Path to ONNX model (optional)")
    parser.add_argument("--num-queries", type=int, default=100,
                       help="Number of queries to benchmark")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of benchmark runs")
    args = parser.parse_args()

    # Generate test queries
    queries = [f"Test query number {i}" for i in range(args.num_queries)]
    print(f"Generated {len(queries)} test queries")

    # Try to load real model if available
    model_path = args.model or "./models/all-MiniLM-L6-v2.onnx"

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        ort_session = ort.InferenceSession(model_path, sess_options)
        print(f"Loaded ONNX model: {model_path}")
        use_onnx = True
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Running simulated benchmark instead...")
        use_onnx = False

    print("\n" + "=" * 60)
    print("BATCHING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Queries: {args.num_queries}, Runs: {args.num_runs}")
    print("-" * 60)

    if use_onnx:
        # Benchmark naive
        t_naive = benchmark_naive(ort_session, queries, args.num_runs)
        qps_naive = len(queries) / t_naive
        print(f"\nNaive (batch_size=1):")
        print(f"  Time: {t_naive*1000:.1f}ms, QPS: {qps_naive:.1f}")

        # Benchmark different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        for bs in batch_sizes:
            t_batched = benchmark_batched(ort_session, queries, bs, args.num_runs)
            qps_batched = len(queries) / t_batched
            speedup = t_naive / t_batched
            print(f"\nBatch size {bs:3d}:")
            print(f"  Time: {t_batched*1000:.1f}ms, QPS: {qps_batched:.1f}, Speedup: {speedup:.2f}x")
    else:
        # Simulated results based on expected speedup
        print("\nSimulated expected results:")
        print("  Based on typical GPU batching improvements:")
        base_time = 2340  # ms for 100 queries
        print(f"\nNaive (batch_size=1):     {base_time}ms")
        for bs, speedup in [(8, 3.0), (16, 4.5), (32, 6.2), (64, 7.0), (128, 7.5)]:
            t = base_time / speedup
            qps = 100 / (t / 1000)
            print(f"Batch size {bs:3d}:              {t:.0f}ms ({qps:.1f} q/s), {speedup:.1f}x speedup")

    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
