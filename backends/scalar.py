"""
backends/scalar.py — Karpathy's original microgpt, verbatim.

The algorithm is not touched.  Every Value node, every scalar Python loop
is identical to the source gist.  We only wrap it in a class so all four
backends share the same train_step() / generate() interface.

Original:
  https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import math
import random


# ── autograd engine ───────────────────────────────────────────────────────────

class Value:
    """
    Stores a single scalar value and its gradient.
    Copied verbatim from Karpathy's gist (slot-optimised variant).
    """
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data         = data
        self.grad         = 0
        self._children    = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data),  (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data),  (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data),    (self,), (float(self.data > 0),))

    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):   return self * other ** -1
    def __rtruediv__(self, other):  return other * self ** -1

    def backward(self):
        topo, seen = [], set()

        def visit(v):
            if v not in seen:
                seen.add(v)
                for c in v._children:
                    visit(c)
                topo.append(v)

        visit(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ── model ops (free functions, matching the gist) ─────────────────────────────

def _linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def _softmax(logits):
    m = max(v.data for v in logits)
    e = [(v - m).exp() for v in logits]
    s = sum(e)
    return [ei / s for ei in e]


def _rmsnorm(x):
    ms    = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# ── model class ───────────────────────────────────────────────────────────────

class MicroGPTScalar:
    """
    Karpathy's microgpt with a minimal class wrapper for benchmarking.
    The GPT forward pass, backward pass, and Adam update are all unchanged.
    """

    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size, seed=42):
        random.seed(seed)

        self.vocab_size = vocab_size
        self.C          = n_embd
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.block_size = block_size
        self.head_dim   = n_embd // n_head

        def mat(nout, nin, std=0.02):
            return [
                [Value(random.gauss(0, std)) for _ in range(nin)]
                for _ in range(nout)
            ]

        self.sd = {
            "wte": mat(vocab_size, n_embd),
            "wpe": mat(block_size, n_embd),
        }
        for i in range(n_layer):
            self.sd[f"l{i}.wq"]  = mat(n_embd,      n_embd)
            self.sd[f"l{i}.wk"]  = mat(n_embd,      n_embd)
            self.sd[f"l{i}.wv"]  = mat(n_embd,      n_embd)
            self.sd[f"l{i}.wo"]  = mat(n_embd,      n_embd,     std=0)
            self.sd[f"l{i}.fc1"] = mat(4 * n_embd,  n_embd)
            self.sd[f"l{i}.fc2"] = mat(n_embd,      4 * n_embd, std=0)

        self.params = [p for m in self.sd.values() for row in m for p in row]
        self._m = [0.0] * len(self.params)
        self._v = [0.0] * len(self.params)
        self._t = 0

    # ── forward ───────────────────────────────────────────────────────────────

    def _gpt(self, token_id, pos_id, keys, vals):
        """Single-token forward. Verbatim from Karpathy's gpt() function."""
        sd = self.sd
        tok_emb = sd["wte"][token_id]
        pos_emb = sd["wpe"][pos_id % self.block_size]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = _rmsnorm(x)

        for li in range(self.n_layer):
            x_res = x
            x = _rmsnorm(x)
            q = _linear(x, sd[f"l{li}.wq"])
            k = _linear(x, sd[f"l{li}.wk"])
            v = _linear(x, sd[f"l{li}.wv"])
            keys[li].append(k)
            vals[li].append(v)

            x_attn = []
            for h in range(self.n_head):
                hs  = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in vals[li]]
                al = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim))
                    / self.head_dim ** 0.5
                    for t in range(len(k_h))
                ]
                aw = _softmax(al)
                ho = [
                    sum(aw[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(ho)

            x = _linear(x_attn, sd[f"l{li}.wo"])
            x = [a + b for a, b in zip(x, x_res)]

            x_res = x
            x = _rmsnorm(x)
            x = _linear(x, sd[f"l{li}.fc1"])
            x = [xi.relu() ** 2 for xi in x]   # squared ReLU  (gist version)
            x = _linear(x, sd[f"l{li}.fc2"])
            x = [a + b for a, b in zip(x, x_res)]

        return _linear(x, sd["wte"])    # weight-tied lm_head

    # ── train ─────────────────────────────────────────────────────────────────

    def train_step(self, tokens: list, lr_t: float) -> float:
        """One forward + backward + Adam step. Returns loss as Python float."""
        n    = min(self.block_size, len(tokens) - 1)
        keys = [[] for _ in range(self.n_layer)]
        vals = [[] for _ in range(self.n_layer)]

        losses = []
        for pos_id in range(n):
            tid, tgt = tokens[pos_id], tokens[pos_id + 1]
            logits   = self._gpt(tid, pos_id, keys, vals)
            probs    = _softmax(logits)
            losses.append(-probs[tgt].log())

        loss = sum(losses) / n
        loss.backward()

        self._t += 1
        b1, b2, eps = 0.9, 0.95, 1e-8
        for i, p in enumerate(self.params):
            self._m[i] = b1 * self._m[i] + (1 - b1) * p.grad
            self._v[i] = b2 * self._v[i] + (1 - b2) * p.grad ** 2
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p.data -= lr_t * mh / (vh ** 0.5 + eps)
            p.grad  = 0

        return loss.data

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(self, bos_token: int, vocab_size: int, temperature: float = 1.0):
        """Autoregressively generate token ids until BOS reappears."""
        keys = [[] for _ in range(self.n_layer)]
        vals = [[] for _ in range(self.n_layer)]
        tid  = bos_token
        out  = []

        for pos_id in range(self.block_size):
            logits = self._gpt(tid, pos_id, keys, vals)
            probs  = _softmax([l / temperature for l in logits])
            tid    = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid == bos_token:
                break
            out.append(tid)

        return out
