from .common import xp
from .softmax import log_softmax, softmax

def _smooth_targets(targets: xp.ndarray,
                    vocab_size: int,
                    epsilon: float,
                    pad_idx: int) -> xp.ndarray:
    """
    Versão vetorizada e eficiente em memória que constrói a distribuição alvo suavizada.
    Esta função é projetada para funcionar com vocabulários grandes sem estourar a VRAM.
    """
    B, T = targets.shape
    
    # --- ECONOMIZAR MEMÓRIA ACONTECE AQUI AGR ---
    # 1. Em vez de criar uma matriz identidade de (vocab_size, vocab_size) que consumiria >12GB,
    #    nós criamos a matriz one-hot diretamente no formato final (B, T, vocab_size).
    #    Isso consome muito menos memória (apenas alguns MB).
    one_hot = xp.zeros((B, T, vocab_size), dtype=xp.float32)

    # 2. Cria arrays de índices para as duas primeiras dimensões (batch e tempo)
    b_idx = xp.arange(B)[:, None]
    t_idx = xp.arange(T)[None, :]
    
    # 3. Usa indexação avançada para colocar '1's nos lugares corretos.
    #    Isso faz a mesma coisa que put_along_axis, mas com funções mais básicas.
    one_hot[b_idx, t_idx, targets] = 1.0
    # --- FIM DA CORREÇÃO ---

    #suavizavao
    smooth_dist = one_hot * (1.0 - epsilon) + (1.0 - one_hot) * epsilon / (vocab_size - 1)
    pad_mask = (targets == pad_idx)[:, :, None]
    smooth_dist = xp.where(pad_mask, 0.0, smooth_dist)
    
    return smooth_dist

def label_smoothing_loss(logits: xp.ndarray,
                         targets: xp.ndarray,
                         pad_idx: int,
                         epsilon: float = 0.1) -> float:
    """
    [cite_start]Calcula a perda com label smoothing. [cite: 224]

    Args:
        logits (xp.ndarray): Saída bruta do modelo (B, T, V).
        targets (xp.ndarray): Índices alvo corretos (B, T).
        pad_idx (int): O índice do token de padding a ser ignorado.
        [cite_start]epsilon (float): O fator de suavização. [cite: 224]

    Returns:
        float: O valor da perda, normalizado pelo número de tokens.
    """
    B, T, V = logits.shape
    log_probs = log_softmax(logits)
    true_dist = _smooth_targets(targets, V, epsilon, pad_idx)
    
    # Calcula a cross-entropy negativa
    loss_all = -xp.sum(true_dist * log_probs, axis=-1)
    
    # Cria uma máscara para ignorar o padding no cálculo da perda
    mask = (targets != pad_idx).astype(xp.float32)
    
    # Aplica a máscara e normaliza pelo número total de tokens não-padding
    total_loss = xp.sum(loss_all * mask)
    total_tokens = xp.sum(mask)
    
    # Adiciona uma pequena constante para evitar divisão por zero
    return float(total_loss / (total_tokens + 1e-9))

def label_smoothing_grad(logits: xp.ndarray,
                         targets: xp.ndarray,
                         pad_idx: int,
                         epsilon: float = 0.1) -> xp.ndarray:
    """
    Calcula o gradiente da perda em relação aos logits (dL/dlogits).
    """
    B, T, V = logits.shape
    p_pred = softmax(logits) # Probabilidades previstas
    p_true = _smooth_targets(targets, V, epsilon, pad_idx) # Probabilidades alvo suavizadas
    
    mask = (targets != pad_idx).astype(xp.float32)
    
    # O gradiente é a diferença entre a distribuição prevista e a distribuição alvo
    diff = p_pred - p_true
    
    # Zera o gradiente nas posições de padding e normaliza
    diff = diff * mask[:, :, None]
    total_tokens = xp.sum(mask)
    
    return diff / (total_tokens + 1e-9)