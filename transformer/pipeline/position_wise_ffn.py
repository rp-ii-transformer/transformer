from .common import xp
import math

class PositionWiseFeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int):
        """
        Feed‑forward: duas camadas lineares com ReLU.
        W1: (d_model, d_ff), b1: (d_ff,)
        W2: (d_ff, d_model), b2: (d_model,)
        """
        # Xavier uniforme
        lim1 = math.sqrt(6 / (d_model + d_ff))
        self.W1 = xp.random.uniform(-lim1, lim1, (d_model, d_ff), dtype=xp.float32)
        self.b1 = xp.zeros((d_ff,), dtype=xp.float32)

        lim2 = math.sqrt(6 / (d_ff + d_model))
        self.W2 = xp.random.uniform(-lim2, lim2, (d_ff, d_model), dtype=xp.float32)
        self.b2 = xp.zeros((d_model,), dtype=xp.float32)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """
        x: (batch, seq_len, d_model)
        retorna: (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)                 # (B*T, d_model)
        # camada 1 + ReLU
        h = xp.dot(flat, self.W1) + self.b1     # (B*T, d_ff)
        h = xp.maximum(h, 0)
        # camada 2
        out = xp.dot(h, self.W2) + self.b2      # (B*T, d_model)
        return out.reshape(B, T, D)
