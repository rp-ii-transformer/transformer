from .common import xp
import math

class PositionWiseFeedForwardNetwork:
    """
    Rede Feed‑Forward ponto‑a-ponto como no artigo "Attention Is All You Need".
    Consiste de duas camadas lineares com ReLU no meio.
    """
    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-6):
        """
        d_model: dimensão de entrada/saída
        d_ff:    dimensão interna da camada oculta
        """
        # Xavier uniform para W1
        limit1 = math.sqrt(6 / (d_model + d_ff))
        self.W1 = xp.random.uniform(-limit1, limit1, (d_model, d_ff), dtype=xp.float32)
        self.b1 = xp.zeros((d_ff,), dtype=xp.float32)

        # Xavier uniform para W2
        limit2 = math.sqrt(6 / (d_ff + d_model))
        self.W2 = xp.random.uniform(-limit2, limit2, (d_ff, d_model), dtype=xp.float32)
        self.b2 = xp.zeros((d_model,), dtype=xp.float32)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """
        x: xp.ndarray de forma (batch, seq_len, d_model)
        retorna: xp.ndarray de forma (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        # achata para (B*T, d_model)
        flat = x.reshape(-1, D)

        # 1ª camada + ReLU
        hidden = xp.dot(flat, self.W1) + self.b1
        hidden = xp.maximum(0, hidden)

        # 2ª camada
        out = xp.dot(hidden, self.W2) + self.b2

        # retorna à forma original
        return out.reshape(B, T, D)