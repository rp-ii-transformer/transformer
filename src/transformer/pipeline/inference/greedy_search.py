import numpy as np
from ..pipeline.softmax import softmax

def greedy_decode(model, src, src_mask, max_len, start_idx):
    memory = model.encode(src, src_mask)  # (batch, src_len, d_model)
    batch = src.shape[0]
    ys = np.full((batch, 1), start_idx, dtype=int)

    for _ in range(max_len-1):
        out = model.decode(ys, memory)
        logits = model.project(out[:, -1, :])   # (batch, vocab_size)
        next_tokens = np.argmax(softmax(logits), axis=-1, keepdims=True)
        ys = np.concatenate([ys, next_tokens], axis=1)
    return ys
