"""
ONNX Runtime inference benchmark.

Usage:
    python inference_benchmark.py [--onnx checkpoints/model_fp32.onnx] [--iters 100]
"""
import os
import sys
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime benchmark")
    parser.add_argument("--onnx", type=str, default="checkpoints/model_fp32.onnx")
    parser.add_argument("--iters", type=int, default=100, help="Warmup + benchmark iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check ONNX file
    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX model not found: {args.onnx}")
        print("Run quantize_export.py first to generate it.")
        sys.exit(1)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnx / onnxruntime not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    # Validate
    print(f"[1/3] Validating ONNX model: {args.onnx}")
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print("  Model is valid ✓")
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

    # Create session
    print(f"\n[2/3] Creating ONNX Runtime session...")
    sess = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )

    # Dummy inputs
    spec = np.random.randn(1, 3, 224, 224).astype(np.float32)
    imu = np.random.randn(1, 6, 128).astype(np.float32)
    inputs = {"spectrogram": spec, "imu_signal": imu}

    # Warmup
    print(f"\n[3/3] Benchmarking ({args.iters} iterations)...")
    for _ in range(10):
        _ = sess.run(None, inputs)

    # Benchmark
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = sess.run(None, inputs)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000  # ms
    print("\n" + "=" * 50)
    print("ONNX Inference Benchmark Results")
    print("=" * 50)
    print(f"  Mean latency:  {times_ms.mean():.2f} ms")
    print(f"  Std latency:   {times_ms.std():.2f} ms")
    print(f"  Min latency:   {times_ms.min():.2f} ms")
    print(f"  Max latency:   {times_ms.max():.2f} ms")
    print(f"  P95 latency:   {np.percentile(times_ms, 95):.2f} ms")
    print(f"  Throughput:    {1000 / times_ms.mean():.1f} inferences/sec")

    # Compare FP32 vs INT8 if both exist
    int8_path = args.onnx.replace("fp32", "int8")
    if os.path.exists(int8_path):
        print(f"\n  Comparing with INT8 model...")
        sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        for _ in range(10):
            _ = sess_int8.run(None, inputs)

        times_int8 = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = sess_int8.run(None, inputs)
            times_int8.append(time.perf_counter() - t0)

        t_int8 = np.array(times_int8).mean() * 1000
        t_fp32 = times_ms.mean()
        speedup = t_fp32 / t_int8
        print(f"  INT8 mean latency: {t_int8:.2f} ms")
        print(f"  Speedup:           {speedup:.2f}x")

    print("\nDone!")


if __name__ == "__main__":
    main()
