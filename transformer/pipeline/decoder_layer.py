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

    def forward(self,
                x: xp.ndarray,
                memory: xp.ndarray,
                tgt_mask: xp.ndarray | None = None,
                src_mask: xp.ndarray | None = None) -> xp.ndarray:
        # 1) Masked Self-Attention
        # Q=K=V=x, máscara triangular tgt_mask
        attn1 = self.self_attn.forward(x, x, x, mask=tgt_mask)
        x1    = self.norm1.forward(x, attn1)

        # 2) Encoder-Decoder Attention
        # Q=x1, K=V=memory, máscara src_mask
        attn2 = self.cross_attn.forward(x1, memory, memory, mask=src_mask)
        x2    = self.norm2.forward(x1, attn2)

        # 3) Feed-Forward
        ffn_out = self.ffn.forward(x2)
        # última conexão residual + norm
        return self.norm3.forward(x2, ffn_out)
