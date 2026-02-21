"""
backends/torch_backend.py — microgpt with PyTorch.

Same algorithm, same hyperparameters.  PyTorch's autograd handles
backpropagation, and torch.optim.Adam handles the parameter update.
Pass device='cpu' or device='cuda' to switch targets.

The RTX 5080 (Blackwell, sm_120) is supported from PyTorch 2.6+.
"""

import numpy as np
import torch
import torch.nn.functional as F


class MicroGPTTorch:

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_layer,
        n_head,
        block_size,
        device="cpu",
        seed=42,
    ):
        torch.manual_seed(seed)

        self.V        = vocab_size
        self.C        = n_embd
        self.n_layer  = n_layer
        self.n_head   = n_head
        self.T_max    = block_size
        self.head_dim = n_embd // n_head
        self.device   = torch.device(device)

        def mat(r, c, std=0.02):
            t = torch.randn(r, c, dtype=torch.float32, device=self.device) * std
            return t.requires_grad_(True)

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

        # standard Adam — betas/eps match Karpathy's gist
        self.optimizer = torch.optim.Adam(
            list(self.p.values()),
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rmsnorm(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()

    # ── core transformer ──────────────────────────────────────────────────────

    def _transformer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : LongTensor [T]
        Returns logits [T, V]   (weight-tied, same as numpy/scalar versions)
        """
        T  = inputs.shape[0]
        p  = self.p
        dv = self.device

        tok_emb = p["wte"][inputs]
        pos_emb = p["wpe"][:T]
        x = self._rmsnorm(tok_emb + pos_emb)

        for li in range(self.n_layer):
            # ── attention ─────────────────────────────────────────────────────
            x_res = x
            xn    = self._rmsnorm(x)

            Q = xn @ p[f"l{li}.wq"].T                           # [T, C]
            K = xn @ p[f"l{li}.wk"].T
            V = xn @ p[f"l{li}.wv"].T

            # [T, C] → [H, T, D]
            Q = Q.view(T, self.n_head, self.head_dim).transpose(0, 1)
            K = K.view(T, self.n_head, self.head_dim).transpose(0, 1)
            V = V.view(T, self.n_head, self.head_dim).transpose(0, 1)

            scale  = self.head_dim ** -0.5
            scores = (Q @ K.transpose(-2, -1)) * scale          # [H, T, T]

            # causal mask — upper triangle
            mask   = torch.triu(
                torch.ones(T, T, device=dv, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn   = F.softmax(scores, dim=-1)

            out = (attn @ V).transpose(0, 1).reshape(T, self.C)
            x   = out @ p[f"l{li}.wo"].T + x_res

            # ── MLP ───────────────────────────────────────────────────────────
            x_res = x
            xn    = self._rmsnorm(x)
            h     = F.relu(xn @ p[f"l{li}.fc1"].T) ** 2        # squared ReLU
            x     = h  @ p[f"l{li}.fc2"].T + x_res

        # weight-tied lm_head
        return x @ p["wte"].T                                   # [T, V]

    # ── train ─────────────────────────────────────────────────────────────────

    def train_step(self, tokens: list, lr_t: float) -> float:
        """Forward + backward + Adam. Returns scalar loss."""
        # linear LR decay applied per-step
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr_t

        inputs  = torch.tensor(tokens[:-1], dtype=torch.long,  device=self.device)
        targets = torch.tensor(tokens[1:],  dtype=torch.long,  device=self.device)

        self.optimizer.zero_grad()
        logits = self._transformer(inputs)
        loss   = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(self, bos_token: int, vocab_size: int, temperature: float = 1.0):
        """Autoregressively generate token ids, stopping at BOS."""
        context = [bos_token]
        out     = []

        with torch.no_grad():
            for _ in range(self.T_max - 1):
                inp    = torch.tensor(context, dtype=torch.long, device=self.device)
                logits = self._transformer(inp)[-1] / temperature   # last position
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                tid    = int(np.random.choice(vocab_size, p=probs))
                if tid == bos_token:
                    break
                context.append(tid)
                out.append(tid)

        return out
