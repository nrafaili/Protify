import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from typing import Optional, Tuple, Union


Linear = partial(nn.Linear, bias=False)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False, _inplace=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        """
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(hidden_size), Linear(hidden_size, hidden_size * 3)
        )
        self.out_proj = Linear(hidden_size, hidden_size)
        self.q_ln = nn.LayerNorm(hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        qkv = self.layernorm_qkv(x) # (bs, seq_len, d_model * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (bs, seq_len, hidden_size)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class AttentionPooler(nn.Module):
    """
    Cross-attention mechanism for pooling (b, L, d) -> (b, n_tokens, d_pooled)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int = 1,
            n_heads: int = 16,
    ):
        super(AttentionPooler, self).__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.d_head = hidden_size // n_heads
        self.n_heads = n_heads
        self.Q = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Wq = Linear(hidden_size, hidden_size)
        self.Wv = Linear(hidden_size, hidden_size)
        self.Wk = Linear(hidden_size, hidden_size)
        self.Wo = Linear(hidden_size, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q = self.Wq(self.Q).expand(x.size(0), -1, -1)  # (b, n_tokens, d)
        v = self.Wv(x)  # (b, L, d)
        k = self.Wk(x)  # (b, L, d)
        q, k, v = map(self.reshaper, (q, k, v))  # (b, n_heads, n_tokens, d_head) (b, n_heads, L, d_head) (b, n_heads, L, d_head)
        if output_attentions:
            # Manually compute attention scores
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (b, n_heads, n_tokens, L)
            
            if attention_mask is not None:
                scores = scores + attention_mask
                
            attention_probs = F.softmax(scores, dim=-1)
            context_layer = torch.matmul(attention_probs, v)  # (b, n_heads, n_tokens, d_head)
            context_layer = rearrange(context_layer, "b h s d -> b s (h d)")  # (b, n_tokens, n_heads * d_head)
            output = self.Wo(context_layer)  # (b, n_tokens, d_pooled)
            
            return output, attention_probs
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False
            ) # (b, n_heads, n_tokens, d_head)
            attn = rearrange(attn, "b h s d -> b s (h d)")  # (b, n_tokens, n_heads * d_head)
            return self.Wo(attn)  # (b, n_tokens, d_pooled)
