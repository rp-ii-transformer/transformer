import math

# ── Implementação PyTorch ───────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

class TokenEmbedding(nn.Module):
    """
    Embedding layer em PyTorch, com Xavier-Uniform initialization.
    """
    def __init__(self, vocab_size: int, d_model: int, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model

        # Xavier-Uniform
        limit = math.sqrt(6 / (vocab_size + d_model))
        weights = (torch.rand(vocab_size, d_model, device=device) * 2 - 1) * limit
        self.embedding_weights = nn.Parameter(weights)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        token_ids: LongTensor of shape (batch, seq_len)
        retorna:   Tensor of shape (batch, seq_len, d_model)
        """
        embedded = self.embedding_weights[token_ids]
        return embedded * math.sqrt(self.d_model)


# ── Implementação NumPy/CuPy ────────────────────────────────────────────
from .common import xp  # xp → numpy ou cupy

def create_embedding(vocab_size: int, d_model: int) -> xp.ndarray:
    """
    Cria matriz de embedding (vocab_size, d_model) Xavier-Uniform.
    """
    limit = math.sqrt(6 / (vocab_size + d_model))
    return (xp.random.rand(vocab_size, d_model, dtype=xp.float32) * 2 - 1) * limit

def embed_tokens(token_ids: xp.ndarray, W: xp.ndarray) -> xp.ndarray:
    """
    token_ids: array int32 (batch, seq_len)
    W:         array float32 (vocab_size, d_model)
    retorna:   array float32 (batch, seq_len, d_model)
    """
    B, T = token_ids.shape
    D    = W.shape[1]
    out  = xp.zeros((B, T, D), dtype=xp.float32)
    for i in range(B):
        for j in range(T):
            out[i, j] = W[token_ids[i, j]]
    return out * math.sqrt(D)


def visualizar_embedding(model, vocab_stoi, vocab_itos, frase: str, tipo='src', max_len=20):
    """
    Visualiza entrada, pesos e saída da camada de embedding para uma frase dada.
    Usa o vocabulário carregado manualmente (vocab_stoi e vocab_itos).
    """
    import numpy as np

    pad_idx = vocab_stoi['<pad>']
    tokens = frase.lower().split()
    token_ids = [vocab_stoi['<sos>']] + [vocab_stoi.get(tok, vocab_stoi['<unk>']) for tok in tokens] + [
        vocab_stoi['<eos>']]
    token_ids += [pad_idx] * (max_len - len(token_ids))
    token_ids = token_ids[:max_len]

    tokens_completos = [vocab_itos[i] for i in token_ids]

    # Seleciona embedding
    W = model.Wemb_src if tipo == 'src' else model.Wemb_tgt

    # Calcula embedding
    embedded = embed_tokens(xp.array([token_ids]), W)

    # Converte para numpy (se for cupy)
    if hasattr(W, "get"):
        W = W.get()
        embedded = embedded.get()

    # Exibe tudo
    print(f"> Frase: \"{frase}\"\n")
    print(f"> Tokens:    {tokens_completos}")
    print(f"> Token IDs: {token_ids}\n")

    print(f"> Primeiros vetores da matriz de embedding ({'Wemb_src' if tipo == 'src' else 'Wemb_tgt'}):")
    print(np.round(W[:5], 3))

    print(f"\n>️  Saída da camada de embedding:")
    print(np.round(embedded[0], 3))

    return embedded[0]


vocab_size = 5
d_model = 4
token_ids = np.array([[0, 2, 4], [3, 1, 0]])

W = create_embedding(vocab_size, d_model)
print(np.round(W, 3))

embedded = embed_tokens(token_ids, W)
print(np.round(embedded, 3))
