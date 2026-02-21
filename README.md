# microgpt-efficiency

> **"Everything else is just for efficiency."** — Andrej Karpathy

![Image](https://github.com/user-attachments/assets/796b5fac-d3cf-4ae2-8ba0-1c7035ba178d)

Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) trains and runs a GPT in 243 lines of pure, dependency-free Python. His words: "This is the *full* algorithmic content of what is needed." It's an art project — deliberately written at the scalar level so every weight update, every gradient, every attention score is a visible Python object.

This project asks the natural follow-up question: **what does "just for efficiency" actually cost?**

The identical algorithm is implemented across four backends. Each one swaps out exactly one layer of abstraction. The result is a controlled experiment on where the speed comes from and what it takes to get there.

---

## Results

Tested on **NVIDIA GeForce RTX 5080** · PyTorch 2.9.0+cu128 · Ubuntu 24.04  
Dataset: names.txt (32,033 docs, vocab size 27) · seed 42

### Median ms / step

| Backend | 1,000 steps | 3,000 steps | 5,000 steps | 10,000 steps |
|---|---|---|---|---|
| `scalar` | 35.99 ms | 50.71 ms | 51.51 ms | 39.16 ms |
| `numpy` | 0.15 ms | 0.22 ms | 0.21 ms | 0.15 ms |
| `torch_cpu` | 0.83 ms | 0.83 ms | 0.83 ms | 0.84 ms |
| `torch_gpu` | 1.80 ms | 1.81 ms | 1.06 ms | 1.49 ms |

### Speedup over scalar (total wall-clock)

| Backend | 1,000 steps | 3,000 steps | 5,000 steps | 10,000 steps |
|---|---|---|---|---|
| `numpy` | 251.7× | 237.3× | 248.4× | 274.1× |
| `torch_cpu` | 2.0× | 3.0× | 2.7× | 2.2× |
| `torch_gpu` | 18.6× | 29.4× | 37.3× | 26.4× |

All four backends converge to the same loss range (~2.2 avg over last 100 steps at 10,000 steps) — the algorithm is identical across all implementations.

**Reading the numbers:**

`numpy` is consistently ~250× faster than scalar regardless of how many steps are run. Median per-step time is stable at 0.15–0.22 ms across all runs. At n_embd=16 the actual matrix operations take microseconds, and numpy adds almost nothing on top.

`torch_cpu` median is rock-stable at 0.83–0.84 ms, but total wall-clock speedup is low (2–3×) because PyTorch's JIT compiler fires several spikes of 40–200 ms mid-run. The median is the honest number here; total is inflated by initialisation cost.

`torch_gpu` is slower than `numpy` per step. The model is too small to keep the GPU busy — each step mostly pays CPU→GPU transfer and CUDA kernel launch overhead rather than doing useful parallel computation. The per-step time drifts 1.0–1.8 ms across runs as CUDA warms up. The crossover where GPU wins requires larger batch sizes or wider models.

---

## What changes between backends

```
scalar ── Python scalar Value objects, computation graph autograd, no libs
   │
   ↓  replace scalar graph with vectorised matrix ops + hand-derived gradients
   │
numpy ─── [T, C] matrix operations, manual backward pass, BLAS (OpenBLAS/MKL)
   │
   ↓  replace manual gradients with PyTorch autograd engine
   │
torch_cpu  PyTorch tensors on CPU, .backward(), torch.optim.Adam
   │
   ↓  replace CPU execution with CUDA kernels
   │
torch_gpu  CUDA kernels on RTX 5080, same graph, parallel matrix execution
```

### scalar → numpy

The `Value` autograd engine goes away. Gradients are derived analytically using matrix calculus and implemented as NumPy array operations. Every formula has a corresponding derivation comment.

Key gradient derivations:

**RMSNorm backward** — let `y = x / rms`, `rms = sqrt(mean(x²) + ε)`:

```
dL/dx_i = dL/dy_i / rms  -  x_i · Σ(dL/dy_j · x_j) / (C · rms³)
```

**Softmax backward** — let `p = softmax(z)`, `g = dL/dp`:

```
dL/dz_i = p_i · (g_i - Σ_j p_j · g_j)
```

**Softmax + cross-entropy combined** (used for lm_head):

```
dL/dlogit_i = p_i - 1{i = y}  scaled by 1/T
```

Each formula collapses what would be hundreds of scalar graph nodes into a single matrix operation. The math is the same; only the bookkeeping changes.

### numpy → torch_cpu

Manual gradient code disappears. PyTorch's autograd handles backpropagation. The forward pass is expressed in the same terms; `.backward()` takes care of the rest. The optimizer also switches from a hand-written Adam to `torch.optim.Adam`.

### torch_cpu → torch_gpu

One line changes in `build_model`: `device="cpu"` becomes `device="cuda"`. PyTorch dispatches the same operations to CUDA kernels on the RTX 5080. All tensor allocations move to GPU memory automatically.

---

## Quick start

```bash
# check GPU availability (optional — scalar and numpy run without torch)
python -c "import torch; print(torch.cuda.get_device_name(0))"

git clone https://github.com/chanjoongx/microgpt-efficiency
cd microgpt-efficiency

pip install -r requirements.txt

# full benchmark — all four backends, 1,000 steps
python benchmark.py

# specific backends only
python benchmark.py --backends scalar numpy

# quick validation run
python benchmark.py --backends numpy torch_cpu torch_gpu --steps 100 --log_every 50

# on a machine with multiple GPUs, pin to one
CUDA_VISIBLE_DEVICES=1 python benchmark.py --steps 1000
```

---

## How to read `numpy_backend.py`

The file is structured as: **forward pass → backward pass → Adam → public API**.

The backward pass is the educational core. Each block corresponds to one module in the forward pass, traversed in reverse:

```
lm_head backward
    ↓
for each layer (reversed):
    MLP backward
      → fc2
      → squared ReLU
      → fc1
      → RMSNorm
    Attention backward
      → output projection (wo)
      → weighted sum (attn @ V)
      → softmax
      → scaled dot-product scores
      → Q, K, V projections
      → RMSNorm
    ↓
embedding + positional embedding backward
```

Every gradient formula includes a derivation comment. Reading the forward pass first and tracing each variable into the backward pass is the fastest path to understanding.

---

## Gradient verification

All gradients in `numpy_backend.py` are verified against finite differences (`ε = 1e-5`). Errors are within float64 numerical precision (~1e-8):

```
wte     max_err = 2.98e-08  ✓
wpe     max_err = 2.79e-08  ✓
l0.wq   max_err = 0.00e+00  ✓
l0.wk   max_err = 0.00e+00  ✓
l0.wv   max_err = 0.00e+00  ✓
l0.wo   max_err = 1.63e-11  ✓
l0.fc1  max_err = 0.00e+00  ✓
l0.fc2  max_err = 9.33e-12  ✓
```

---

## Architecture

All backends implement the same GPT variant from Karpathy's gist:

| | |
|---|---|
| Tokenisation | character-level, vocab = 27 (a–z + BOS) |
| Embedding dim | 16 |
| Attention heads | 4 |
| Head dim | 4 |
| Context length | 16 |
| Layers | 1 |
| MLP activation | squared ReLU |
| Normalisation | RMSNorm (pre-norm) |
| Weight tying | `wte` shared with lm_head |
| Optimiser | Adam (β₁=0.9, β₂=0.95, ε=1e-8) |
| LR schedule | linear decay to 0 |

---

## Project layout

```
microgpt-efficiency/
├── backends/
│   ├── scalar.py              # Karpathy's original — zero changes to the algorithm
│   ├── numpy_backend.py       # NumPy, manual backward pass
│   └── torch_backend.py       # PyTorch, cpu or cuda
├── data.py                    # dataset download + tokenisation (shared)
├── benchmark.py               # runs all backends, prints summary table
├── requirements.txt
├── LICENSE
└── results/                   # benchmark output files
```

---

## Acknowledgements

This project builds entirely on Andrej Karpathy's **microgpt** — the original source and the idea that a GPT can be trained and run in 243 lines of pure Python.

- Original gist: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Blog post: https://karpathy.github.io/2026/02/12/microgpt/
- Published under the MIT License

`backends/scalar.py` is Karpathy's algorithm with a minimal class wrapper added for benchmarking. The algorithm itself is his, unchanged.
