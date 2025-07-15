from .common import xp
import math

def get_positional_encoding(max_len: int, d_model: int) -> xp.ndarray:
    """
    Retorna um array de shape (1, max_len, d_model), para poder fazer:
      x + pe[:, :T, :]
    onde x é (B, T, d_model).
    """
    pe = xp.zeros((max_len, d_model), dtype=xp.float32)
    position = xp.arange(max_len, dtype=xp.float32)[:, xp.newaxis]          # (max_len, 1)
    div_term = xp.exp( xp.arange(0, d_model, 2, dtype=xp.float32) *
                      (-math.log(10000.0) / d_model) )                      # (d_model/2,)

    pe[:, 0::2] = xp.sin(position * div_term)
    pe[:, 1::2] = xp.cos(position * div_term)

    return pe[None, :, :]  # shape (1, max_len, d_model)


def add_positional_encoding(x: xp.ndarray, pe: xp.ndarray) -> xp.ndarray:
    """
    x: (B, T, d_model)
    pe: (1, max_len, d_model)
    """
    return x + pe[:, : x.shape[1], :]
