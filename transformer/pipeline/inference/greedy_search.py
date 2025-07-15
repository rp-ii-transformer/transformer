import numpy as onp
from ..common  import xp
from ..softmax import softmax

def greedy_decode(model, src, src_mask, max_len, start_idx):
    memory = model.encode(src, src_mask)
    batch = src.shape[0]
    ys = onp.full((batch, 1), start_idx, dtype=int)

    for _ in range(max_len - 1):
        out = model.decode(ys, memory)
        logits = model.project(out[:, -1, :])
        next_tokens = xp.argmax(softmax(xp.asarray(logits)), axis=-1).get()  # converte p/ NumPy
        ys = onp.concatenate([ys, next_tokens[:, None]], axis=1)
    return ys
