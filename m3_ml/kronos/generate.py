"""Autoregressive sampling that keeps every path (PLAN.md section 6).

Differences from the official auto_regressive_inference:
1. every sampled path is returned (the official code returns only their mean);
2. the fine (s2) token is decoded with the full-length sibling sequence, as in
   training: position i carries the coarse token of token i + 1, the last
   position the newly sampled one, and only the last output is used. The
   official code passes a length-1 query, so its rotary encoding treats every
   key as position 0, unlike training.

sample_paths recomputes the whole sequence at every step (the reference
implementation). sample_paths_cached gives the same logits with key/value
caching: each origin's context is run once and shared by its paths, and each
step processes one new token per path (PLAN.md bug log: the full recomputation
ran out of GPU memory and crashed the computer).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import LOOKBACK
from .vendor.kronos import sample_from_logits


@torch.no_grad()
def sample_paths(tokenizer, model, x, stamps_ctx, stamps_future, T, top_p, paths, device, seed):
    """x: (origins, LOOKBACK, 6) normalized; stamps: (origins, LOOKBACK, 5) and
    (origins, horizon, 5). Returns normalized decoded closes, shape
    (origins, paths, horizon + 1): position 0 is the origin bar, then h = 1..horizon."""
    torch.manual_seed(seed)
    n, horizon = x.shape[0], stamps_future.shape[1]
    rep = lambda a: torch.as_tensor(a, device=device).repeat_interleave(paths, dim=0)
    x, stamps = rep(x), torch.cat([rep(stamps_ctx), rep(stamps_future)], dim=1)
    s1, s2 = tokenizer.encode(x.float(), half=True)
    for i in range(horizon):
        length = s1.shape[1]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            s1_logits, context = model.decode_s1(s1, s2, stamps[:, :length])
        new_s1 = sample_from_logits(s1_logits[:, -1, :].float(), temperature=T, top_k=0, top_p=top_p)
        sibling = torch.cat([s1[:, 1:], new_s1], dim=1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            s2_logits = model.decode_s2(context, sibling)
        new_s2 = sample_from_logits(s2_logits[:, -1, :].float(), temperature=T, top_k=0, top_p=top_p)
        s1, s2 = torch.cat([s1, new_s1], dim=1), torch.cat([s2, new_s2], dim=1)
    decoded = tokenizer.decode([s1, s2], half=True).float()       # (n * paths, LOOKBACK + horizon, 6)
    close = decoded[:, LOOKBACK - 1:, 3].reshape(n, paths, horizon + 1)
    return close.cpu().numpy()


def _heads(attn, x):
    """(B, n, d) -> (B, heads, n, head_dim)."""
    return x.view(x.shape[0], x.shape[1], attn.n_heads, attn.head_dim).transpose(1, 2)


def _merge(x):
    """(B, heads, n, head_dim) -> (B, n, d)."""
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def _rope(x, start, inv_freq):
    """The vendor's rotary encoding of x (B, heads, n, head_dim) at positions start, start + 1, ..."""
    t = torch.arange(start, start + x.shape[2], device=x.device).type_as(inv_freq)
    emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return x * emb.cos() + torch.cat((-x2, x1), dim=-1) * emb.sin()


class _KV:
    """Keys and values of one attention layer for every path, preallocated to the final length."""

    def __init__(self, k, v, paths, total):
        b, h, n, d = k.shape
        self.k, self.v = k.new_empty(b * paths, h, total, d), v.new_empty(b * paths, h, total, d)
        self.k[:, :, :n] = k.repeat_interleave(paths, dim=0)
        self.v[:, :, :n] = v.repeat_interleave(paths, dim=0)
        self.n = n

    def add(self, k, v):
        self.k[:, :, self.n], self.v[:, :, self.n] = k[:, :, 0], v[:, :, 0]
        self.n += 1
        return self.view()

    def view(self):
        return self.k[:, :, :self.n], self.v[:, :, :self.n]


def _block(block, x, kv, start):
    """One vendor TransformerBlock (eval mode) on new positions start, start + 1, ...
    Without a cache: causal attention among them. With a cache (one new position):
    its key/value is appended and it attends to every position so far."""
    attn = block.self_attn
    h = block.norm1(x)
    q, k, v = (_heads(attn, proj(h)) for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
    q, k = _rope(q, start, attn.rotary.inv_freq), _rope(k, start, attn.rotary.inv_freq)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True) if kv is None else \
        F.scaled_dot_product_attention(q, *kv.add(k, v))
    x = x + attn.out_proj(_merge(out))
    return x + block.ffn(block.norm2(x)), k, v


class CachedDecoder:
    """Incremental decoding for the vendor Kronos predictor in eval mode. Gives the
    same logits as decode_s1 / decode_s2 on the full sequence
    (ml/tests/test_kronos_inference.py)."""

    def __init__(self, model, s1, s2, stamps, paths, total):
        """Run the context (origins x n tokens) once and copy its caches to every path."""
        m = self.m = model
        x = m.embedding([s1, s2]) + m.time_emb(stamps)
        self.self_kv = []
        for block in m.transformer:
            x, k, v = _block(block, x, None, 0)
            self.self_kv.append(_KV(k, v, paths, total))
        h = m.norm(x)
        self.cross_kv = _KV(*self._cross(h, 0), paths, total)
        self.last = h[:, -1:].repeat_interleave(paths, dim=0)
        self.s1_logits = m.head(self.last)[:, -1]

    def _cross(self, h, start):
        """Keys and values of the fine-token cross-attention for hidden states h."""
        attn = self.m.dep_layer.cross_attn
        return _rope(_heads(attn, attn.k_proj(h)), start, attn.rotary.inv_freq), _heads(attn, attn.v_proj(h))

    def s2_logits(self, s1):
        """Fine-token logits for the next position given its sampled coarse token s1
        (paths x 1): the query sits at the last position and sees every position so far."""
        m = self.m
        attn = m.dep_layer.cross_attn
        q = _rope(_heads(attn, attn.q_proj(m.embedding.emb_s1(s1))), self.cross_kv.n - 1, attn.rotary.inv_freq)
        out = attn.out_proj(_merge(F.scaled_dot_product_attention(q, *self.cross_kv.view())))
        return m.head.cond_forward(m.dep_layer.norm(self.last + out))[:, -1]

    def append(self, s1, s2, stamp):
        """Add the sampled token (paths x 1) and compute the coarse logits for the position after it."""
        m = self.m
        start = self.cross_kv.n
        x = m.embedding([s1, s2]) + m.time_emb(stamp)
        for block, kv in zip(m.transformer, self.self_kv):
            x = _block(block, x, kv, start)[0]
        h = m.norm(x)
        self.cross_kv.add(*self._cross(h, start))
        self.last = h
        self.s1_logits = m.head(h)[:, -1]


@torch.no_grad()
def sample_paths_cached(tokenizer, model, x, stamps_ctx, stamps_future, T, top_p, paths, device, seed):
    """Same inputs, outputs and sampling as sample_paths, with CachedDecoder."""
    torch.manual_seed(seed)
    n, horizon = x.shape[0], stamps_future.shape[1]
    amp = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda')
    stamps = torch.cat([torch.as_tensor(stamps_ctx, device=device), torch.as_tensor(stamps_future, device=device)], dim=1)
    s1, s2 = tokenizer.encode(torch.as_tensor(x, device=device).float(), half=True)
    length = s1.shape[1]
    with amp():
        dec = CachedDecoder(model, s1, s2, stamps[:, :length], paths, length + horizon)
    stamps = stamps.repeat_interleave(paths, dim=0)
    s1, s2 = [s1.repeat_interleave(paths, dim=0)], [s2.repeat_interleave(paths, dim=0)]
    for i in range(horizon):
        new_s1 = sample_from_logits(dec.s1_logits.float(), temperature=T, top_k=0, top_p=top_p)
        with amp():
            s2_logits = dec.s2_logits(new_s1)
        new_s2 = sample_from_logits(s2_logits.float(), temperature=T, top_k=0, top_p=top_p)
        s1.append(new_s1)
        s2.append(new_s2)
        if i < horizon - 1:
            with amp():
                dec.append(new_s1, new_s2, stamps[:, length + i:length + i + 1])
    # The tokenizer decodes the entire history; release predictor KV buffers
    # before its full-sequence activations are allocated on the same GPU.
    del dec
    decoded = tokenizer.decode([torch.cat(s1, dim=1), torch.cat(s2, dim=1)], half=True).float()
    close = decoded[:, LOOKBACK - 1:, 3].reshape(n, paths, horizon + 1)
    return close.cpu().numpy()


def return_quantiles(close_norm, mean, std, horizons, taus):
    """Quantiles of the h-day log ratio C_{t+h} / C_t per origin, from normalized
    decoded closes (origins, paths, horizon + 1) and the context close mean/std.
    The base C_t is each path's decoded origin close, which cancels the tokenizer's
    reconstruction bias. Non-positive ratios are floored at 0.001 before the log."""
    close = close_norm * (std[:, None, None] + 1e-5) + mean[:, None, None]
    out = {}
    for h in horizons:
        ratio = close[:, :, h] / close[:, :, 0]
        q = np.quantile(ratio, taus, axis=1).T
        out[h] = np.log(np.maximum(q, 1e-3))
    return out
