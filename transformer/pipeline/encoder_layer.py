from .common import xp
from .multi_head_attention import MultiHeadAttention
from .position_wise_ffn import PositionWiseFeedForwardNetwork
from .residual_layer_norm import ResidualLayerNorm
import math

class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        # Agora self.self_attn é uma instância da classe importada e reutilizável
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        # Atenção: Passe o dropout para o ResidualLayerNorm
        self.norm1 = ResidualLayerNorm(d_model, dropout=dropout)
        self.norm2 = ResidualLayerNorm(d_model, dropout=dropout)
        # O self.dropout aqui não é mais necessário
        # self.dropout = dropout

    def forward(self, x: xp.ndarray, mask: xp.ndarray | None = None, training: bool = True):
        # Cache para backward
        cache = {}
        cache['x'] = x
        
        # CORREÇÃO: A chamada ao forward agora usa a assinatura genérica (query, key, value)
        # Para self-attention, todos são 'x'.
        attn_out, cache['attn_cache'] = self.self_attn.forward(x, x, x, mask=mask)

        # Passa o flag 'training' para a camada de normalização (para controlar o dropout)
        x1, cache['norm1_cache'] = self.norm1.forward(x, attn_out, training=training)
        
        ffn_out, cache['ffn_cache'] = self.ffn.forward(x1)
        out, cache['norm2_cache'] = self.norm2.forward(x1, ffn_out, training=training)
        
        return out, cache

    def backward(self, dout, cache):
        # Backprop pela norm2
        dx1_ffn, dffn_out, norm2_grads = self.norm2.backward(dout, cache['norm2_cache'])
        
        # Backprop pela FFN
        dx1_ffn_b, ffn_grads = self.ffn.backward(dffn_out, cache['ffn_cache'])
        dx1 = dx1_ffn + dx1_ffn_b # Acumula gradientes
        
        # Backprop pela norm1
        dx_attn, dattn_out, norm1_grads = self.norm1.backward(dx1, cache['norm1_cache'])
        
        # Backprop pela self-attention
        # Como é self-attention, o gradiente de entrada é a soma dos gradientes de query, key e value.
        dx_attn_b, attn_grads = self.self_attn.backward(dattn_out, cache['attn_cache'])
        x = cache['x']
        dx = dx_attn + dx_attn_b # Acumula gradientes
        
        # Junta todos os gradientes
        grads = {**attn_grads, **ffn_grads, **norm1_grads, **norm2_grads}
        return dx, grads