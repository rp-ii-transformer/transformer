from transformer.pipeline.common import xp
from ..pipeline.softmax import softmax

def greedy_decode(model, src, src_mask, max_len, start_idx):
    memory = model.encode(src, src_mask)  # (batch, src_len, d_model)
    batch = shape[0]
    ys = xp.full((batch, 1), start_idx, dtype=int)

    for _ in range(max_len-1):
        out = model.decode(ys, memory)
        logits = model.project(out[:, -1, :])   # (batch, vocab_size)
        next_tokens = xp.argmax(softmax(logits), axis=-1, keepdims=True)
        ys = xp.concatenate([ys, next_tokens], axis=1)
    return ys
