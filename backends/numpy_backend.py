"""
backends/numpy_backend.py — microgpt with NumPy.

Same algorithm as Karpathy's scalar version. What changes:
  • All ops run on full [T, C] matrices — vectorised over the time axis.
  • Gradients are derived analytically (matrix calculus) instead of
    building a computation graph node-by-node.

This is Karpathy's "Everything else is just for efficiency" made concrete:
the scalar autograd computes exactly the same values, one cell at a time.

Notation used in math comments:
  T  = sequence length
  C  = n_embd  (model width)
  H  = n_head
  D  = head_dim = C / H
  4C = MLP hidden width
  V  = vocab_size
"""

import numpy as np


class MicroGPTNumpy:

    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size, seed=42):
        rng = np.random.default_rng(seed)

        self.V        = vocab_size
        self.C        = n_embd
        self.n_layer  = n_layer
        self.n_head   = n_head
        self.T_max    = block_size
        self.head_dim = n_embd // n_head

        def mat(r, c, std=0.02):
            return (rng.standard_normal((r, c)) * std).astype(np.float64)

        self.p = {
            "wte": mat(vocab_size, n_embd),
            "wpe": mat(block_size, n_embd),
        }
        for i in range(n_layer):
            self.p[f"l{i}.wq"]  = mat(n_embd,      n_embd)
            self.p[f"l{i}.wk"]  = mat(n_embd,      n_embd)
            self.p[f"l{i}.wv"]  = mat(n_embd,      n_embd)
            self.p[f"l{i}.wo"]  = mat(n_embd,      n_embd,     std=0)
            self.p[f"l{i}.fc1"] = mat(4 * n_embd,  n_embd)
            self.p[f"l{i}.fc2"] = mat(n_embd,      4 * n_embd, std=0)

        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    # ── primitive ops ─────────────────────────────────────────────────────────

    @staticmethod
    def _rmsnorm(x):
        """
        y = x / sqrt(mean(x^2) + eps)

        Returns (y, rms) where rms has shape [T, 1] — needed for backward.
        """
        rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
        return x / rms, rms

    @staticmethod
    def _rmsnorm_bwd(dy, x_pre, rms):
        """
        Backward through  y_i = x_i / rms,  rms = sqrt(mean(x^2) + eps).

        Derivation (chain rule on s = mean(x^2) + eps,  y_i = x_i * s^{-1/2}):

          dL/dx_i = dL/dy_i / rms
                    - x_i * sum_j(dL/dy_j * x_j) / (C * rms^3)

        Vectorised over [T, C]:
          dx = dy / rms  -  x_pre * (dy . x_pre).sum(-1, keepdim) / (C * rms^3)

        Args:
          dy    : upstream gradient  [T, C]
          x_pre : input  to rmsnorm  [T, C]  (pre-normalisation)
          rms   : cached rms values  [T, 1]
        """
        C   = x_pre.shape[-1]
        dot = (dy * x_pre).sum(axis=-1, keepdims=True)   # [T, 1]
        return dy / rms - x_pre * dot / (C * rms ** 3)

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax over last axis."""
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    # ── forward pass ──────────────────────────────────────────────────────────

    def _forward(self, tokens):
        """
        Full-sequence forward pass.

        tokens : list[int] of length T+1
                 (T context tokens, each paired with the next token as target)

        Returns (loss: float, cache: dict)
        The cache stores every intermediate value required for the backward pass.
        """
        inputs  = np.array(tokens[:-1], dtype=np.intp)   # [T]
        targets = np.array(tokens[1:],  dtype=np.intp)   # [T]
        T       = len(inputs)

        cache = {
            "layers":  [{} for _ in range(self.n_layer)],
            "T":       T,
            "inputs":  inputs,
            "targets": targets,
        }

        # ── embeddings ────────────────────────────────────────────────────────
        tok_emb = self.p["wte"][inputs]      # [T, C]
        pos_emb = self.p["wpe"][:T]          # [T, C]
        x_emb   = tok_emb + pos_emb          # [T, C]

        # initial RMSNorm — applied once before the transformer layers
        # (same as the scalar version: x = rmsnorm(tok_emb + pos_emb))
        x, rms0 = self._rmsnorm(x_emb)

        cache["x_emb"] = x_emb   # pre-norm (needed for rmsnorm backward)
        cache["rms0"]  = rms0

        # ── transformer layers ────────────────────────────────────────────────
        for li in range(self.n_layer):
            lc = cache["layers"][li]

            # ────────────────────────────── Attention sublayer ────────────────
            lc["x_res_a"] = x                           # residual branch
            xn, rms_a     = self._rmsnorm(x)
            lc["rms_a"]   = rms_a
            lc["xn_a"]    = xn   # normalised input; also == x_pre for rmsnorm_bwd

            # QKV projections:  Q = xn @ wq.T,  etc.   [T, C]
            Q = xn @ self.p[f"l{li}.wq"].T
            K = xn @ self.p[f"l{li}.wk"].T
            V = xn @ self.p[f"l{li}.wv"].T
            lc["Q_flat"] = Q
            lc["K_flat"] = K
            lc["V_flat"] = V

            # reshape to multi-head view:  [T, C] → [H, T, D]
            # T captured by closure via default arg — safe even inside the loop
            Q = Q.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            K = K.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            V = V.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            lc["Q"] = Q   # [H, T, D]
            lc["K"] = K
            lc["V"] = V

            # scaled dot-product scores:  score[h,i,j] = Q[h,i] . K[h,j] / sqrt(D)
            scale        = self.head_dim ** -0.5
            scores       = (Q @ K.transpose(0, 2, 1)) * scale    # [H, T, T]

            # causal mask — upper triangle (future positions) → -inf
            mask          = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[:, mask] = -1e9   # large negative avoids true -inf edge cases

            attn          = self._softmax(scores)                 # [H, T, T]
            attn[:, mask] = 0.0                                   # clean masked weights
            lc["attn"]    = attn
            lc["mask"]    = mask
            lc["scale"]   = scale

            # weighted sum over values
            # out[h, t] = sum_s  attn[h, t, s] * V[h, s]
            out         = (attn @ V).transpose(1, 0, 2).reshape(T, self.C)  # [T, C]
            lc["attn_out"] = out

            # output projection + residual
            x = out @ self.p[f"l{li}.wo"].T + x    # x is still x_res_a here

            # ────────────────────────────── MLP sublayer ──────────────────────
            lc["x_res_m"] = x
            xn, rms_m     = self._rmsnorm(x)
            lc["rms_m"]   = rms_m
            lc["xn_m"]    = xn

            h_pre         = xn @ self.p[f"l{li}.fc1"].T   # [T, 4C]
            h_relu        = np.maximum(0.0, h_pre)         # ReLU gate
            h             = h_relu ** 2                    # squared ReLU (gist)

            lc["h_pre"]  = h_pre
            lc["h_relu"] = h_relu
            lc["h"]      = h

            x = h @ self.p[f"l{li}.fc2"].T + x     # projection + residual

        # ── lm_head — weight-tied with wte ───────────────────────────────────
        cache["x_final"] = x
        logits = x @ self.p["wte"].T                             # [T, V]

        # ── cross-entropy loss ────────────────────────────────────────────────
        probs = self._softmax(logits)                            # [T, V]
        # clamp to avoid log(0); numerically equivalent for probs near 1
        loss  = -np.log(probs[np.arange(T), targets] + 1e-12).mean()

        cache["probs"] = probs

        return float(loss), cache

    # ── backward pass ─────────────────────────────────────────────────────────

    def _backward(self, cache):
        """
        Analytical backward pass through the full computation graph.

        Every gradient formula is derived from first principles.
        Returns a gradient dict with the same keys as self.p.
        """
        T       = cache["T"]
        probs   = cache["probs"]
        targets = cache["targets"]
        grads   = {k: np.zeros_like(v) for k, v in self.p.items()}

        # ── dL/dlogits ────────────────────────────────────────────────────────
        # Softmax + cross-entropy have a clean combined gradient:
        #
        #   dL/dlogit[t, i] = (prob[t, i] - 1{i == y_t}) / T
        #
        # Proof: let p = softmax(z), L = -log(p[y]).
        #   dL/dz_i = dp_y/dz_i * (-1/p_y)
        #           = p_y*(delta_{yi} - p_i) * (-1/p_y)
        #           = p_i - delta_{yi}
        dlogits                        = probs.copy()          # [T, V]
        dlogits[np.arange(T), targets] -= 1.0
        dlogits                        /= T

        # ── lm_head:  logits = x_final @ wte.T ───────────────────────────────
        # If  Y = X @ W.T  then  dX = dY @ W,  dW += dY.T @ X
        x_final       = cache["x_final"]
        dx             = dlogits @ self.p["wte"]               # [T, C]
        grads["wte"]  += dlogits.T @ x_final                  # [V, C]

        # ── transformer layers — reversed ────────────────────────────────────
        for li in reversed(range(self.n_layer)):
            lc = cache["layers"][li]

            # ────────────────────────────── MLP backward ──────────────────────
            # Forward was:  x = h @ fc2.T  +  x_res_m
            # So gradient splits: one copy through the MLP, one through residual.
            dx_res = dx.copy()

            # fc2 backward:  x_mlp = h @ fc2.T
            #   dh     = dx @ fc2          [T, 4C]
            #   d_fc2 += dx.T @ h          [C, 4C]
            h    = lc["h"]
            xn_m = lc["xn_m"]
            dh   = dx @ self.p[f"l{li}.fc2"]                   # [T, 4C]
            grads[f"l{li}.fc2"] += dx.T @ h                    # [C, 4C]

            # squared ReLU backward:  h = relu(h_pre)^2
            # Let r = relu(h_pre) = max(0, h_pre)
            #   dh/dh_pre = 2 * r * 1{h_pre > 0}   (chain rule on r^2 then relu)
            h_relu = lc["h_relu"]
            h_pre  = lc["h_pre"]
            dh_pre = dh * (2.0 * h_relu) * (h_pre > 0)        # [T, 4C]

            # fc1 backward:  h_pre = xn_m @ fc1.T
            #   dxn_m  = dh_pre @ fc1      [T, C]
            #   d_fc1 += dh_pre.T @ xn_m   [4C, C]
            dxn_m = dh_pre @ self.p[f"l{li}.fc1"]              # [T, C]
            grads[f"l{li}.fc1"] += dh_pre.T @ xn_m             # [4C, C]

            # RMSNorm backward then merge residual
            # x_res_m is both the residual value AND the pre-norm input
            dx = self._rmsnorm_bwd(dxn_m, lc["x_res_m"], lc["rms_m"]) + dx_res

            # ────────────────────────────── Attention backward ─────────────────
            # Forward was:  x = attn_out @ wo.T  +  x_res_a
            dx_res = dx.copy()

            # wo backward:  x_wo = attn_out @ wo.T
            attn_out   = lc["attn_out"]
            d_attn_out = dx @ self.p[f"l{li}.wo"]              # [T, C]
            grads[f"l{li}.wo"] += dx.T @ attn_out              # [C, C]

            # reshape [T, C] → [H, T, D] for per-head ops
            d_out = d_attn_out.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)

            Q, K, V = lc["Q"], lc["K"], lc["V"]
            attn    = lc["attn"]
            mask    = lc["mask"]
            scale   = lc["scale"]

            # weighted-sum backward:  out = attn @ V
            #   dV     = attn.T @ d_out      [H, T, D]
            #   d_attn = d_out @ V.T         [H, T, T]
            dV      = attn.transpose(0, 2, 1) @ d_out          # [H, T, D]
            d_attn  = d_out @ V.transpose(0, 2, 1)             # [H, T, T]

            # Softmax backward (Jacobian of softmax applied to d_attn):
            #
            #   d_score[t,s] = attn[t,s] * (d_attn[t,s]
            #                               - sum_k attn[t,k] * d_attn[t,k])
            #
            # This is the standard result: ds = p * (g - (p . g)) where p = attn.
            d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
            d_scores[:, mask] = 0.0   # masked positions carry no gradient
            d_scores          *= scale    # undo the 1/sqrt(D) scaling from forward

            # score = Q @ K.T  →  dQ, dK
            #   dQ = d_scores @ K          [H, T, D]
            #   dK = d_scores.T @ Q        [H, T, D]
            dQ = d_scores @ K                                   # [H, T, D]
            dK = d_scores.transpose(0, 2, 1) @ Q               # [H, T, D]

            # merge heads back:  [H, T, D] → [T, C]
            dQ = dQ.transpose(1, 0, 2).reshape(T, self.C)
            dK = dK.transpose(1, 0, 2).reshape(T, self.C)
            dV = dV.transpose(1, 0, 2).reshape(T, self.C)

            # wq/wk/wv backward:  Q = xn_a @ wq.T
            #   d_wq  += dQ.T @ xn_a   [C, C]
            #   dxn_a  = dQ @ wq  +  dK @ wk  +  dV @ wv   [T, C]
            # Note: compute dxn_a first (uses self.p, which is not yet updated),
            # then accumulate weight grads.
            xn_a = lc["xn_a"]
            dxn_a = (
                dQ @ self.p[f"l{li}.wq"]
                + dK @ self.p[f"l{li}.wk"]
                + dV @ self.p[f"l{li}.wv"]
            )                                                   # [T, C]

            grads[f"l{li}.wq"] += dQ.T @ xn_a                 # [C, C]
            grads[f"l{li}.wk"] += dK.T @ xn_a
            grads[f"l{li}.wv"] += dV.T @ xn_a

            # RMSNorm backward then merge residual
            # x_res_a is both the residual value AND the pre-norm input
            dx = self._rmsnorm_bwd(dxn_a, lc["x_res_a"], lc["rms_a"]) + dx_res

        # ── initial embedding + RMSNorm backward ─────────────────────────────
        # dx here is gradient w.r.t. x after the initial rmsnorm(x_emb).
        x_emb  = cache["x_emb"]
        dx_emb = self._rmsnorm_bwd(dx, x_emb, cache["rms0"])   # [T, C]

        # wte[inputs] gets gradient from token lookups (scatter-add because
        # the same character can appear multiple times in one document)
        np.add.at(grads["wte"], cache["inputs"], dx_emb)

        # wpe gets gradient from positional embeddings
        grads["wpe"][:T] += dx_emb

        return grads

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _adam(self, grads, lr_t, b1=0.9, b2=0.95, eps=1e-8):
        self._t += 1
        t = self._t
        for k in self.p:
            self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * grads[k] ** 2
            mh = self._m[k] / (1 - b1 ** t)
            vh = self._v[k] / (1 - b2 ** t)
            self.p[k] -= lr_t * mh / (np.sqrt(vh) + eps)

    # ── public API ────────────────────────────────────────────────────────────

    def train_step(self, tokens: list, lr_t: float) -> float:
        loss, cache = self._forward(tokens)
        grads       = self._backward(cache)
        self._adam(grads, lr_t)
        return loss

    def generate(
        self,
        bos_token: int,
        vocab_size: int,
        temperature: float = 1.0,
        rng=None,
    ):
        """
        Autoregressively generate token ids by running the full forward pass
        on the growing context at each step.
        """
        if rng is None:
            rng = np.random.default_rng()

        context = [bos_token]
        out     = []

        for _ in range(self.T_max - 1):
            # forward on current context; last element is a dummy target
            _, cache = self._forward(context + [bos_token])

            # take the logit for the last real position
            logits  = cache["x_final"][-1] @ self.p["wte"].T   # [V]
            logits  = logits / temperature
            logits -= logits.max()
            probs   = np.exp(logits)
            probs  /= probs.sum()

            tid = int(rng.choice(vocab_size, p=probs))
            if tid == bos_token:
                break
            context.append(tid)
            out.append(tid)

        return out
