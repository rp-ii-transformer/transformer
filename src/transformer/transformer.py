import torch
from transformer_manual import forward_pass_transformer_manual

class CustomTransformer(torch.nn.Module):
    def __init__(self, vocab_size=10000, d_model=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids):
        return forward_pass_transformer_manual(token_ids, self.vocab_size, self.d_model)

def transformer_start(model_name, device):
    model = CustomTransformer().to(device)
    print(f"TRANSFORMER: Modelo manual carregado: {type(model)}")
    return model
