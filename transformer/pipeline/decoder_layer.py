from .common import xp
import math
from .softmax import softmax
from .scaled_dot_product_attention import scaled_dot_product_attention
from .multi_head_attention import MultiHeadAttention
from .position_wise_ffn import PositionWiseFeedForwardNetwork
from .residual_layer_norm import ResidualLayerNorm

class DecoderLayer:
    """
    Uma camada do decoder do Transformer, com:
      1) Masked Multi-Head Self-Attention
      2) Multi-Head Cross-Attention (encoder-decoder)
      3) Feed-Forward ponto-a-ponto
    Cada subcamada é seguida de conexão residual + LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        # Self-attention mascarada (impede olhar o futuro)
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        # Cross-attention: queries do decoder, keys/values do encoder
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Rede ponto-a-ponto
        self.ffn        = PositionWiseFeedForwardNetwork(d_model, d_ff)
        # Normalizações (aplicadas após cada residual)
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.norm3 = ResidualLayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, memory, tgt_mask, src_mask, training: bool = True):
        cache = {}
        # Salva as entradas da camada no cache
        cache['x'] = x
        cache['memory'] = memory

        # Masked Self-Attention
        attn1, cache['sa_cache'] = self.self_attn.forward(x, x, x, mask=tgt_mask)
        x1, cache['norm1_cache'] = self.norm1.forward(x, attn1, training=training)
        
        # Cross-Attention
        attn2, cache['ca_cache'] = self.cross_attn.forward(x1, memory, memory, mask=src_mask)
        x2, cache['norm2_cache'] = self.norm2.forward(x1, attn2, training=training)

        # Feed-Forward
        ffn_out, cache['ffn_cache'] = self.ffn.forward(x2)
        out, cache['norm3_cache'] = self.norm3.forward(x2, ffn_out, training=training)

        return out, cache

    def backward(self, dout, cache):
        # Backprop pela norm3 e FFN
        dx2_ffn, dffn_out, norm3_grads = self.norm3.backward(dout, cache['norm3_cache'])
        dx2_ffn_b, ffn_grads = self.ffn.backward(dffn_out, cache['ffn_cache'])
        dx2 = dx2_ffn + dx2_ffn_b

        # Backprop through norm2
        dx1_attn, dattn2, norm2_grads = self.norm2.backward(dx2, cache['norm2_cache'])
        # Backprop through cross-attention
        dx1_attn_b, d_memory, cross_attn_grads = self.cross_attn.backward_cross(dattn2, cache['ca_cache'])
        dx1 = dx1_attn + dx1_attn_b

        # Backprop through norm1
        dx_attn, dattn1, norm1_grads = self.norm1.backward(dx1, cache['norm1_cache'])
        # Backprop through self-attention
        dx_attn_b, self_attn_grads = self.self_attn.backward(dattn1, cache['sa_cache'])
        dx = dx_attn + dx_attn_b
        
        # Combine all gradients
        grads = {**self_attn_grads, **cross_attn_grads, **ffn_grads, **norm1_grads, **norm2_grads, **norm3_grads}
        return dx, d_memory, grads

