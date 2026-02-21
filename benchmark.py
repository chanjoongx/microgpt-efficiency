#!/usr/bin/env python3
"""
benchmark.py — "Everything else is just for efficiency."

Trains each backend for N steps on identical data, then reports:
  • per-step median latency
  • total wall-clock time
  • final training loss
  • speedup over the scalar baseline

Usage:
  python benchmark.py                          # all backends, 1000 steps
  python benchmark.py --backends scalar numpy  # selected backends only
  python benchmark.py --steps 200 --samples 8 # quick run with more samples
"""

import argparse
import random
import time

import numpy as np

from data import load_dataset
from backends.scalar        import MicroGPTScalar
from backends.numpy_backend import MicroGPTNumpy
# MicroGPTTorch is imported lazily inside build_model() so that
# numpy/scalar backends still work when torch is not installed.


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="microgpt efficiency benchmark")
    p.add_argument(
        "--backends", nargs="+",
        choices=["scalar", "numpy", "torch_cpu", "torch_gpu", "all"],
        default=["all"],
        help="backends to run (default: all)",
    )
    p.add_argument("--steps",      type=int,   default=1000, help="training steps")
    p.add_argument("--n_embd",     type=int,   default=16)
    p.add_argument("--n_layer",    type=int,   default=1)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument(
        "--samples", type=int, default=5,
        help="names to generate after training (0 = skip)",
    )
    p.add_argument(
        "--log_every", type=int, default=200,
        help="print progress every N steps",
    )
    return p.parse_args()


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(backend: str, cfg: dict, seed: int):
    if backend == "scalar":
        return MicroGPTScalar(**cfg, seed=seed)

    if backend == "numpy":
        return MicroGPTNumpy(**cfg, seed=seed)

    if backend in ("torch_cpu", "torch_gpu"):
        try:
            from backends.torch_backend import MicroGPTTorch
        except ImportError:
            raise RuntimeError("PyTorch is not installed — cannot run torch backends")

        if backend == "torch_cpu":
            return MicroGPTTorch(**cfg, device="cpu", seed=seed)

        if backend == "torch_gpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available — cannot run torch_gpu")
            return MicroGPTTorch(**cfg, device="cuda", seed=seed)

    raise ValueError(f"Unknown backend: {backend!r}")


# ── training loop ─────────────────────────────────────────────────────────────

def run_training(model, docs, encode, n_steps, lr, log_every):
    """
    Train for n_steps, log progress, return a results dict.
    The LR schedule matches Karpathy's original: linear decay to 0.
    """
    losses     = []
    step_times = []

    for step in range(n_steps):
        lr_t   = lr * (1 - step / n_steps)
        tokens = encode(docs[step % len(docs)])

        t0   = time.perf_counter()
        loss = model.train_step(tokens, lr_t)
        t1   = time.perf_counter()

        losses.append(loss)
        step_times.append(t1 - t0)

        if (step + 1) % log_every == 0 or step == 0:
            ms = (t1 - t0) * 1000
            print(f"  step {step+1:5d}/{n_steps}  |  loss {loss:.4f}  |  {ms:8.2f} ms")

    return {
        "final_loss":  losses[-1],
        "avg_loss":    float(np.mean(losses[-100:])),   # smoothed tail loss
        "total_s":     sum(step_times),
        "median_ms":   float(np.median(step_times)) * 1000,
        "mean_ms":     float(np.mean(step_times))   * 1000,
    }


# ── results table ─────────────────────────────────────────────────────────────

def print_summary(results):
    print()
    print("=" * 73)
    print("  BENCHMARK SUMMARY")
    print("=" * 73)

    # find scalar baseline for speedup calculation
    baseline_s = next((r["total_s"] for n, r in results if n == "scalar"), None)

    header = f"{'Backend':<14}  {'Final Loss':<11}  {'Avg(last100)':<13}  {'Total(s)':<10}  {'Median ms':<10}  {'Speedup'}"
    print(header)
    print("-" * 73)

    for name, r in results:
        speedup = (
            f"{baseline_s / r['total_s']:.1f}x"
            if baseline_s is not None
            else "—"
        )
        print(
            f"  {name:<12}  {r['final_loss']:<11.4f}  {r['avg_loss']:<13.4f}"
            f"  {r['total_s']:<10.2f}  {r['median_ms']:<10.2f}  {speedup}"
        )

    print("=" * 73)

    if baseline_s is not None and len(results) > 1:
        print()
        print("  Speedup relative to scalar baseline:")
        for name, r in results:
            if name != "scalar":
                sp = baseline_s / r["total_s"]
                bar = "█" * min(int(sp), 80)
                print(f"    {name:<12}  {sp:6.1f}x  {bar}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset …")
    docs, encode, decode, BOS, vocab_size = load_dataset(seed=args.seed)
    print(f"  {len(docs)} docs  |  vocab size {vocab_size}")

    cfg = dict(
        vocab_size  = vocab_size,
        n_embd      = args.n_embd,
        n_layer     = args.n_layer,
        n_head      = args.n_head,
        block_size  = args.block_size,
    )

    to_run = (
        ["scalar", "numpy", "torch_cpu", "torch_gpu"]
        if "all" in args.backends
        else args.backends
    )

    results = []

    for backend in to_run:
        bar = "─" * 60
        print(f"\n{bar}")
        print(f"  Backend: {backend}")
        print(f"{bar}")

        try:
            model  = build_model(backend, cfg, args.seed)
            result = run_training(
                model, docs, encode,
                n_steps=args.steps,
                lr=args.lr,
                log_every=args.log_every,
            )
            results.append((backend, result))

            # generate a few names to confirm the model learned something
            if args.samples > 0:
                print(f"\n  Generated names ({args.samples} samples, T=0.7):")
                for _ in range(args.samples):
                    ids  = model.generate(BOS, vocab_size, temperature=0.7)
                    name = decode(ids)
                    print(f"    {name}")

        except RuntimeError as exc:
            print(f"  SKIPPED — {exc}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
