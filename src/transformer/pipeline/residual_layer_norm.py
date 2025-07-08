import torch
import torch.nn as nn


class ResidualLayerNorm(nn.Module):
    # A equação de Layer Normalization vem da seção 3.1 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que toma como referência outro artigo:
    # "Layer Normalization" (Ba, J. L., Kiros, J. R., & Hinton, G. E., 2016),
    # a fórmula é LayerNorm(x+Sublayer(x)).
    #
    # Para o dropout foi usado como referência a seção 5.4 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que menciona o uso de dropout antes da normalização da camada.
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer é uma função,
        # caso sublayer seja um módulo já chamado mudar para:
        # (x + self.dropout(sublayer))
        return self.norm(x + self.dropout(sublayer(x)))
