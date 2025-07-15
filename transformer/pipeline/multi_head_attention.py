from .common import xp
import math
from .scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadAttention:
    """
    Implementação da camada de Multi-Head Self-Attention.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Inicializador da classe.

        Args:
            embed_dim (int): A dimensionalidade dos embeddings de entrada. (d_model)
            num_heads (int): O número de cabeças de atenção a serem usadas.
        """
        # Garante que a dimensão do embedding pode ser dividida igualmente entre as cabeças
        assert embed_dim % num_heads == 0, "A dimensão do embedding deve ser divisível pelo número de cabeças."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Xavier uniform initialization para W_q, W_k, W_v, W_o
        limit = math.sqrt(6 / (embed_dim + embed_dim))
        self.W_q = xp.random.uniform(-limit, limit, (embed_dim, embed_dim), dtype=xp.float32)
        self.W_k = xp.random.uniform(-limit, limit, (embed_dim, embed_dim), dtype=xp.float32)
        self.W_v = xp.random.uniform(-limit, limit, (embed_dim, embed_dim), dtype=xp.float32)
        self.W_o = xp.random.uniform(-limit, limit, (embed_dim, embed_dim), dtype=xp.float32)

    def _split_heads(self, x: xp.ndarray) -> xp.ndarray:
        """
        Divide a última dimensão em (num_heads, head_dim) e transpõe para o cálculo de atenção.

        Entrada x: (batch_size, seq_length, embed_dim)
        Saída:     (batch_size, num_heads, seq_length, head_dim)
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: xp.ndarray) -> xp.ndarray:
        """
        Reverte _split_heads:
        Entrada x: (batch_size, num_heads, seq_length, head_dim)
        Saída:     (batch_size, seq_length, embed_dim)
        """
        B, nh, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, nh * dh)

    def forward(self,
                query: xp.ndarray,
                key:   xp.ndarray,
                value: xp.ndarray,
                mask:  xp.ndarray | None = None) -> xp.ndarray:
        """
        Passo forward da camada.
        Para self-attention, query, key e value serão o mesmo tensor de entrada.

        Args:
            query: xp.ndarray, shape (batch_size, seq_length, embed_dim)
            key:   xp.ndarray, shape (batch_size, seq_length, embed_dim)
            value: xp.ndarray, shape (batch_size, seq_length, embed_dim)
            mask:  xp.ndarray or None, shape broadcastable to (batch_size, num_heads, seq_length, seq_length)

        Returns:
            xp.ndarray de saída, shape (batch_size, seq_length, embed_dim)
        """
        # 1. Projeção linear
        q = query @ self.W_q   # (B, T, D)
        k = key   @ self.W_k
        v = value @ self.W_v

        # 2. Dividir em múltiplas cabeças
        q = self._split_heads(q)  # (B, nh, T, dk)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3. Scaled dot-product attention
        context = scaled_dot_product_attention(q, k, v, mask)  # (B, nh, T, dk)

        # 4. Concatenar cabeças e aplicar W_o
        concat = self._combine_heads(context)  # (B, T, D)
        output = concat @ self.W_o             # (B, T, D)

        return output


# Exemplo de uso (opcional)
if __name__ == "__main__":
    batch_size = 2
    seq_length = 5
    embed_dim  = 16
    num_heads  = 4

    # Simula dados aleatórios
    x = xp.random.randn(batch_size, seq_length, embed_dim).astype(xp.float32)

    mha = MultiHeadAttention(embed_dim, num_heads)
    out = mha.forward(x, x, x, mask=None)
    print("Output shape:", out.shape)  # deve ser (batch_size, seq_length, embed_dim)
