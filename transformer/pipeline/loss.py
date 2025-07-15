from .common import xp
from .softmax import log_softmax, softmax

def _smooth_targets(targets: xp.ndarray,
                    vocab_size: int,
                    epsilon: float,
                    pad_idx: int) -> xp.ndarray:
    """
    Constrói a distribuição alvo suavizada:
      (1 - ε) na posição correta, ε/(V-1) nas demais.
    Zera todas as posições onde targets == pad_idx.
    Retorna shape (B, T, V).
    """
    B, T = targets.shape
    smooth = xp.full((B, T, vocab_size),
                     fill_value=epsilon / (vocab_size - 1),
                     dtype=xp.float32)
    for b in range(B):
        for t in range(T):
            idx = int(targets[b, t])
            if idx == pad_idx:
                smooth[b, t, :] = 0.0
            else:
                smooth[b, t, idx] = 1.0 - epsilon
    return smooth


def label_smoothing_loss(logits: xp.ndarray,
                         targets: xp.ndarray,
                         pad_idx: int,
                         epsilon: float = 0.1) -> float:
    """
    logits:  (B, T, V)
    targets: (B, T) índices [0..V-1]
    pad_idx: índice de padding
    """
    B, T, V = logits.shape
    # 1) log-probs estáveis
    log_probs = log_softmax(logits)           # (B, T, V)
    # 2) distribuição alvo suavizada
    true_dist = _smooth_targets(targets, V, epsilon, pad_idx)
    # 3) perda de cross-entropy
    loss_all = -xp.sum(true_dist * log_probs, axis=-1)      # (B, T)
    mask     = (targets != pad_idx).astype(xp.float32)      # (B, T)
    total_loss   = xp.sum(loss_all * mask)
    total_tokens = xp.sum(mask)
    return float(total_loss / total_tokens)


def label_smoothing_grad(logits: xp.ndarray,
                         targets: xp.ndarray,
                         pad_idx: int,
                         epsilon: float = 0.1) -> xp.ndarray:
    """
    Retorna gradiente dL/dlogits, shape (B, T, V), para usar no backward.
    Usa a mesma distribuição suavizada de targets.
    """
    B, T, V = logits.shape
    # 1) p_pred = softmax(logits)
    p_pred = softmax(logits)                           # (B, T, V)
    # 2) p_true suavizada
    p_true = _smooth_targets(targets, V, epsilon, pad_idx)
    # 3) máscara para pads
    mask = (targets != pad_idx).astype(xp.float32)      # (B, T)
    # 4) diferença e normalização pela soma de tokens válidos
    diff = p_pred - p_true                              # (B, T, V)
    # aplica máscara nos tempos
    diff = diff * mask[:, :, None]
    total_tokens = xp.sum(mask)
    return diff / total_tokens                          # (B, T, V)
