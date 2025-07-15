from transformer.pipeline.common import xp
from ..pipeline.softmax import softmax

def beam_search(model, src, src_mask, max_len, start_idx, beam_size):
    memory = model.encode(src, src_mask)
    sequences = [([start_idx], 0.0)]  # (seq, score)

    for _ in range(max_len-1):
        all_candidates = []
        for seq, score in sequences:
            ys = xp.array(seq)[None, :]
            out = model.decode(ys, memory)
            logits = model.project(out[:, -1, :])[0]  # (vocab,)
            log_probs = xp.log(softmax(logits))
            topk = xp.argsort(-log_probs)[:beam_size]
            for k in topk:
                cand_seq = seq + [int(k)]
                cand_score = score + float(log_probs[k])
                all_candidates.append((cand_seq, cand_score))
        # seleciona melhores beam_size
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    return sequences
