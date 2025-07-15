from .common import xp
from .scaled_dot_product_attention import scaled_dot_product_attention
from .position_wise_ffn import PositionWiseFeedForwardNetwork
from .residual_layer_norm import ResidualLayerNorm
import math

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.dk        = d_model // num_heads
        limit = math.sqrt(6/(d_model + d_model))
        # inicialização Xavier simples
        self.W_q = xp.random.rand(d_model, d_model, dtype=xp.float32) * limit
        self.W_k = xp.random.rand(d_model, d_model, dtype=xp.float32) * limit
        self.W_v = xp.random.rand(d_model, d_model, dtype=xp.float32) * limit
        self.W_o = xp.random.rand(d_model, d_model, dtype=xp.float32) * limit

    def _split_heads(self, x: xp.ndarray):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.dk)
        return x.transpose(0, 2, 1, 3)  # (B, nh, T, dk)

    def _combine_heads(self, x: xp.ndarray):
        B, nh, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, nh * dk)

    def forward(self, x: xp.ndarray, mask: xp.ndarray | None = None):
        # x: (B, T, D)
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # atenção escalada
        context = scaled_dot_product_attention(q, k, v, mask)
        concat  = self._combine_heads(context)
        return concat @ self.W_o     # (B, T, D)

class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.norm1     = ResidualLayerNorm(d_model)
        self.norm2     = ResidualLayerNorm(d_model)
        self.dropout   = dropout

    def forward(self, x: xp.ndarray, mask: xp.ndarray | None = None):
        attn_out = self.self_attn.forward(x, mask)               # Self‑Attention
        x1       = self.norm1.forward(x, attn_out)               # Residual + LayerNorm
        ffn_out  = self.ffn.forward(x1)                          # Feed‑Forward
        return self.norm2.forward(x1, ffn_out)                   # Residual + LayerNorm
