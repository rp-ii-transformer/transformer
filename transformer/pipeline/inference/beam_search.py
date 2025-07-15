from ..common  import xp
from ..softmax import softmax

def beam_search(model, src, src_mask, max_len, start_idx, beam_size):
    """
    Args:
        model: objeto com métodos encode(src, src_mask) e project(...)
        src:   xp.ndarray (batch, src_len)
        src_mask: xp.ndarray broadcastável para as atenções cruzadas
        max_len: comprimento máximo da sequência de saída
        start_idx: índice de token <sos>
        beam_size: tamanho do beam

    Returns:
        Lista de tuplas (sequence, score) com as top-k hipóteses.
    """
    # 1) codifica a fonte
    memory = model.encode(src, src_mask)  # (B, src_len, d_model)

    # 2) inicializa o beam com apenas o token <sos>
    sequences = [([start_idx], 0.0)]

    for _ in range(max_len - 1):
        all_candidates = []
        for seq, score in sequences:
            # transforma a sequência atual em array (1, cur_len)
            ys = xp.array(seq, dtype=xp.int64)[None, :]
            # decodifica até aqui
            out = model.decode(ys, memory)         # (1, cur_len, d_model)
            # projeta para logits_vocab
            logits = model.project(out[:, -1, :])[0]  # (vocab_size,)
            # log-softmax
            log_probs = xp.log(softmax(logits))
            # top-k
            topk = xp.argsort(-log_probs)[:beam_size]
            for k in topk:
                cand_seq   = seq + [int(k)]
                cand_score = score + float(log_probs[k])
                all_candidates.append((cand_seq, cand_score))

        # mantém apenas os beam_size melhores
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    return sequences
