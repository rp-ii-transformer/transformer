import torch.nn as nn
from transformer.pipeline.token_embedding import TokenEmbedding
from transformer.pipeline.common import xp

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, device='cpu'):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, device=device)
        # camada linear que projeta o vetor de saída (do decoder) de volta para o espaço de vocabulário
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.embedding_weights
        print("🔹 Saída do Token Embedding:", self.output_projection.weight)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        print("🔹 Treinamento: Saída do Token Embedding:", embedded)

        # TODO positional encoding
        positional = embedded
        # positional = embedded + self.positional_encoding(embedded)
        # print("🔹 Treinamento: Saída do Positional Encoding:", embedded)

        out = self.output_projection(positional)
        print("🔹 Treinamento: Saída do Transformer:", out)
        return out


def transformer_start(model_name, device, tokenizer):
    vocab_size = tokenizer.vocab_size
    model = CustomTransformer(vocab_size=vocab_size, device=device).to(device)
    print(f"TRANSFORMER: Modelo manual carregado: {model_name} {type(model)}")
    return model
