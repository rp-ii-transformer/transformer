from .common import xp

class ResidualLayerNorm:
    # A equação de Layer Normalization vem da seção 3.1 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que toma como referência outro artigo:
    # "Layer Normalization" (Ba, J. L., Kiros, J. R., & Hinton, G. E., 2016),
    # a fórmula é LayerNorm(x+Sublayer(x)).
    #
    # Para o dropout foi usado como referência a seção 5.4 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que menciona o uso de dropout antes da normalização da camada.
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6):
        """
        d_model: dimensão dos embeddings
        dropout: probabilidade de zerar ativações da subcamada
        eps: termo de estabilidade numérica
        """
        self.dropout = dropout
        self.eps     = eps
        # parâmetros aprendíveis γ e β
        self.gamma = xp.ones((d_model,), dtype=xp.float32)
        self.beta  = xp.zeros((d_model,), dtype=xp.float32)

    def forward(self, x: xp.ndarray, sublayer_out: xp.ndarray) -> xp.ndarray:
        """
        Aplica conexão residual + dropout + layer norm.

        Args:
            x:           tensor de entrada (batch, seq_len, d_model)
            sublayer_out: saída da subcamada (mesma forma que x)
        Returns:
            tensor normalizado, mesma forma de x
        """
        # aplica dropout à saída da subcamada
        if self.dropout > 0.0:
            mask = (xp.random.rand(*sublayer_out.shape) >= self.dropout).astype(xp.float32)
            sub = sublayer_out * mask / (1.0 - self.dropout)
        else:
            sub = sublayer_out

        # conexão residual
        y = x + sub

        # cálculo do LayerNorm
        mean = xp.mean(y, axis=-1, keepdims=True)
        var  = xp.mean((y - mean) ** 2, axis=-1, keepdims=True)
        y_norm = (y - mean) / xp.sqrt(var + self.eps)

        # escala e shift
        return self.gamma * y_norm + self.beta
