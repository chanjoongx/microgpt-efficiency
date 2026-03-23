#!/usr/bin/env python3
"""
verify_gradients.py — finite-difference gradient checking for numpy_backend.

Verifies every hand-derived gradient in numpy_backend.py against numerical
finite differences (eps = 1e-5).  Expected errors are within float64
precision (~1e-8).

Usage:
  python verify_gradients.py
"""

import sys
import numpy as np

from data import load_dataset
from backends.numpy_backend import MicroGPTNumpy

# ensure unicode output on Windows
sys.stdout.reconfigure(encoding="utf-8")


def finite_diff_check(model, tokens, eps=1e-5):
    """
    Compare analytical gradients from _backward() against numerical
    finite differences for every element of every parameter.

    Returns dict  {param_name: max_absolute_error}.
    """
    # analytical gradients
    loss0, cache = model._forward(tokens)
    grads = model._backward(cache)

    results = {}

    for key in model.p:
        param = model.p[key]
        grad = grads[key]
        max_err = 0.0

        for idx in np.ndindex(param.shape):
            orig = param[idx]

            param[idx] = orig + eps
            loss_plus, _ = model._forward(tokens)

            param[idx] = orig - eps
            loss_minus, _ = model._forward(tokens)

            param[idx] = orig  # restore

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = grad[idx]
            max_err = max(max_err, abs(numerical - analytical))

        results[key] = max_err

    return results


def main():
    docs, encode, decode, BOS, vocab_size = load_dataset(seed=42)

    model = MicroGPTNumpy(
        vocab_size=vocab_size,
        n_embd=16,
        n_layer=1,
        n_head=4,
        block_size=16,
        seed=42,
    )

    tokens = encode(docs[0])
    print(f"Checking gradients on '{docs[0]}' ({len(tokens)} tokens) \u2026\n")

    results = finite_diff_check(model, tokens)

    threshold = 1e-5
    all_ok = True
    for key, err in results.items():
        ok = err < threshold
        mark = "\u2713" if ok else "\u2717"
        print(f"  {key:<8s} max_err = {err:.2e}  {mark}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All gradients verified.")
    else:
        print("SOME GRADIENTS FAILED \u2014 check the derivations.")


if __name__ == "__main__":
    main()
