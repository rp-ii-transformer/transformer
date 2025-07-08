import numpy as np
import src.transformer.pipeline.softmax as log_softmax


def _smooth_targets(targets, vocab_size, epsilon, pad_idx):
    """
    Constrói a distribuição alvo suavizada:
      (1 - ε) na posição correta, ε/(V-1) nas demais.
    Zera todas as posições onde targets == pad_idx.
    """
    batch, seq_len = targets.shape
    # preenche com ε/(V-1)
    smooth = np.full((batch, seq_len, vocab_size),
                     fill_value=epsilon/(vocab_size-1),
                     dtype=np.float32)
    # atribui (1-ε) na posição certa
    for b in range(batch):
        for t in range(seq_len):
            idx = targets[b, t]
            if idx == pad_idx:
                smooth[b, t, :] = 0.
            else:
                smooth[b, t, idx] = 1.0 - epsilon
    return smooth

def label_smoothing_loss(logits, targets, pad_idx, epsilon=0.1):
    """
    logits: np.ndarray (batch, seq_len, vocab_size)
    targets: np.ndarray (batch, seq_len) com índices [0..vocab_size-1]
    pad_idx: int
    epsilon: float, valor de label smoothing (padrão 0.1)
    Retorna: escalar (float) — perda média por token (ignorando pads).
    """
    batch, seq_len, V = logits.shape

    # 1) log-softmax estável
    log_probs = log_softmax(logits)  # shape (batch, seq_len, V)

    # 2) construo distribuição alvo suavizada
    true_dist = _smooth_targets(targets, V, epsilon, pad_idx)

    # 3) perda pontual: −∑ p_true · log p_pred
    #    e máscara para descartar pads
    loss_all = -np.sum(true_dist * log_probs, axis=-1)           # (batch, seq_len)
    mask = (targets != pad_idx).astype(np.float32)               # (batch, seq_len)

    # 4) média somente sobre tokens válidos
    total_loss = np.sum(loss_all * mask)
    total_tokens = np.sum(mask)
    return total_loss / total_tokens
