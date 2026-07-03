#!/usr/bin/env python3
"""
Benchmark script for RAG pipeline performance.
Outputs p50, p95, p99 latency and throughput.
"""
import argparse
import json
import statistics
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def run_query(endpoint: str, query: str, top_k: int = 5) -> Dict:
    """Run a single query and return timing info."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{endpoint}/query",
            json={"query_text": query, "top_k": top_k},
            timeout=30
        )
        duration = time.perf_counter() - start
        return {
            "duration": duration,
            "status": resp.status_code,
            "result": resp.json() if resp.ok else None
        }
    except Exception as e:
        return {"duration": duration, "status": -1, "error": str(e)}


def benchmark(
    endpoint: str,
    queries: List[str],
    concurrency: int = 10,
    top_k: int = 5
) -> Dict:
    """Run benchmark with specified concurrency."""
    latencies = []
    errors = 0

    print(f"Running benchmark: {len(queries)} queries, concurrency={concurrency}")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_query, endpoint, q, top_k)
            for q in queries
        ]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["status"] == 200:
                latencies.append(result["duration"])
            else:
                errors += 1

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(queries)}")

    if not latencies:
        return {"error": "No successful queries"}

    latencies.sort()
    total_time = sum(latencies)

    return {
        "total_queries": len(queries),
        "successful_queries": len(latencies),
        "errors": errors,
        "p50": statistics.median(latencies) * 1000,  # ms
        "p95": latencies[int(len(latencies) * 0.95)] * 1000,
        "p99": latencies[int(len(latencies) * 0.99)] * 1000,
        "avg": statistics.mean(latencies) * 1000,
        "throughput_qps": len(latencies) / total_time if total_time > 0 else 0,
    }


def load_queries(path: str) -> List[str]:
    """Load queries from file or generate defaults."""
    path = Path(path)
    if path.exists():
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get('queries', [])
        return path.read_text().strip().split('\n')
    else:
        # Generate sample queries
        return [
            "What is attention mechanism in neural networks?",
            "How does BERT differ from GPT?",
            "What are the advantages of transformers over RNNs?",
            "Explain the concept of transfer learning.",
            "What is fine-tuning in machine learning?",
            "How does vector similarity search work?",
            "What is approximate nearest neighbor search?",
            "Explain the HNSW algorithm.",
            "What is FAISS used for?",
            "How does batching improve GPU utilization?",
        ]


def main():
    parser = argparse.ArgumentParser(description='Benchmark RAG pipeline')
    parser.add_argument("--endpoint", default="http://localhost:8080",
                       help="API endpoint base URL")
    parser.add_argument("--queries", default="",
                       help="Path to queries file or 'default' for built-in")
    parser.add_argument("--concurrency", type=int, default=10,
                       help="Number of concurrent requests")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    # Load queries
    queries_path = args.queries if args.queries else "default"
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")

    # Check health
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=5)
        print(f"Health check: {resp.json()}")
    except Exception as e:
        print(f"Warning: Health check failed: {e}")

    # Run benchmark
    results = benchmark(args.endpoint, queries, args.concurrency, args.top_k)

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total queries:      {results['total_queries']}")
    print(f"Successful:         {results['successful_queries']}")
    print(f"Errors:             {results['errors']}")
    print(f"\nLatency (ms):")
    print(f"  p50:              {results['p50']:.2f}")
    print(f"  p95:              {results['p95']:.2f}")
    print(f"  p99:              {results['p99']:.2f}")
    print(f"  avg:              {results['avg']:.2f}")
    print(f"\nThroughput:         {results['throughput_qps']:.2f} QPS")
    print("=" * 50)

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
