"""Tiny GPT in MLX: pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings.

RoPE takes explicit per-row positions so batched decoding can left-pad.
"""
import math
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab: int = 4355
    dim: int = 384
    layers: int = 6
    heads: int = 6
    ffn_mult: float = 8 / 3
    max_len: int = 384
    dropout: float = 0.0
    rope_base: float = 10000.0
    compute: str = "bfloat16"  # matmul dtype; master weights stay fp32
    prefix: bool = False  # prefix-LM: bidirectional attention over the english prompt

    def dict(self):
        return asdict(self)


class Linear(nn.Module):
    """fp32 master weight, matmul in compute dtype (mixed precision)."""

    def __init__(self, i, o, dt):
        super().__init__()
        self.weight = mx.random.uniform(-i ** -0.5, i ** -0.5, (o, i))
        self.dt = dt

    def __call__(self, x):
        return x.astype(self.dt) @ self.weight.astype(self.dt).T


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = mx.ones((d,))

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), 1e-5)


def rope(x, pos, base):
    # x: (B, H, T, D)  pos: (B, T) or None (= 0..T-1, fused kernel)
    if pos is None:
        return mx.fast.rope(x, x.shape[-1], traditional=False, base=base, scale=1.0, offset=0)
    d = x.shape[-1] // 2
    inv = mx.exp(-math.log(base) * mx.arange(d, dtype=mx.float32) / d)
    ang = pos[:, None, :, None].astype(mx.float32) * inv  # B,1,T,d
    cos, sin = mx.cos(ang).astype(x.dtype), mx.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h, self.hd, self.base = c.heads, c.dim // c.heads, c.rope_base
        dt = getattr(mx, c.compute)
        self.qkv = Linear(c.dim, 3 * c.dim, dt)
        self.out = Linear(c.dim, c.dim, dt)

    def __call__(self, x, pos, mask, cache=None):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = rope(q, pos, self.base), rope(k, pos, self.base)
        if cache is not None:
            if cache[0] is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            cache[0], cache[1] = k, v
        if isinstance(mask, mx.array):
            mask = mask.astype(q.dtype)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        h = int(c.dim * c.ffn_mult / 64 + 0.999) * 64
        dt = getattr(mx, c.compute)
        self.gate, self.up, self.down = Linear(c.dim, h, dt), Linear(c.dim, h, dt), Linear(h, c.dim, dt)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn, self.mlp = Attention(c), MLP(c)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x, pos, mask, cache=None):
        x = x + self.drop(self.attn(self.n1(x), pos, mask, cache))
        return x + self.drop(self.mlp(self.n2(x)))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.dim)
        self.blocks = [Block(c) for _ in range(c.layers)]
        self.norm = RMSNorm(c.dim)
        self.dt = getattr(mx, c.compute)
        self.drop = nn.Dropout(c.dropout)
        # GPT-2 style init: small embeddings, residual projections scaled by depth
        self.emb.weight = mx.random.normal(self.emb.weight.shape) * 0.02
        for b in self.blocks:
            for lin in (b.attn.out, b.mlp.down):
                lin.weight = lin.weight * (1 / math.sqrt(2 * c.layers))

    def __call__(self, x, pos=None, mask="causal", cache=None):
        B, T = x.shape
        h = self.drop(self.emb(x).astype(self.dt))
        for i, b in enumerate(self.blocks):
            h = b(h, pos, mask, None if cache is None else cache[i])
        h = self.norm(h)
        return h @ self.emb.weight.astype(self.dt).T

    def n_params(self):
        from mlx.utils import tree_flatten
        return sum(v.size for _, v in tree_flatten(self.parameters()))


def quantize(model: GPT, bits: int = 8, group_size: int = 64):
    """Swap every Linear for an MLX QuantizedLinear (inference only)."""
    def q(lin):
        ref = nn.Linear(lin.weight.shape[1], lin.weight.shape[0], bias=False)
        ref.weight = lin.weight
        return nn.QuantizedLinear.from_linear(ref, group_size=group_size, bits=bits)
    for b in model.blocks:
        b.attn.qkv, b.attn.out = q(b.attn.qkv), q(b.attn.out)
        b.mlp.gate, b.mlp.up, b.mlp.down = q(b.mlp.gate), q(b.mlp.up), q(b.mlp.down)
    return model
