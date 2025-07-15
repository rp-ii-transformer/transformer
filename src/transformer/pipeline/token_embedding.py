import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # matriz de embeddings (distribuição Xavier Uniform)

        #  definir o intervalo de inicialização dos pesos da matriz de embeddings
        limit = math.sqrt(6 / (vocab_size + d_model))
        # gera uma matriz aleatória (entre 0 e 1) e multiplica pelos pesos
        weights = (torch.rand(vocab_size, d_model, device=device) * 2 - 1) * limit
        # transforma os pesos em um parametro treinável
        self.embedding_weights = nn.Parameter(weights)

    def forward(self, token_ids):
        # relaciona cada token com o vetor de pesos d_model
        embedded = self.embedding_weights[token_ids]
        # aplica escala
        return embedded * math.sqrt(self.d_model)
