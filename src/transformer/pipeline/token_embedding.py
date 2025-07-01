import torch
import math

def token_embedding_start(vocab_size, d_model, token_ids):
    embedding_weights = create_token_embedding(vocab_size, d_model)
    return embed_tokens(token_ids, embedding_weights, d_model)

def create_token_embedding(vocab_size, d_model, device='cpu'):
    limit = math.sqrt(6 / (vocab_size + d_model))
    embedding_weights = (torch.rand(vocab_size, d_model, device=device) * 2 - 1) * limit
    embedding_weights.requires_grad = True
    return embedding_weights

def embed_tokens(token_ids, embedding_weights, d_model):
    batch_size, seq_len = token_ids.shape
    embedded = torch.zeros((batch_size, seq_len, d_model), device=embedding_weights.device)

    for i in range(batch_size):
        for j in range(seq_len):
            token_id = token_ids[i, j].item()
            embedded[i, j] = embedding_weights[token_id]

    return embedded * math.sqrt(d_model)

if __name__ == "__main__":
    vocab_size = 10000
    d_model = 512
    batch_size = 2
    seq_len = 5

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print("Token IDs:")
    print(token_ids)

    output = token_embedding_start(vocab_size, d_model, token_ids)

    # Esperado: (2, 5, 512)
    print("Shape dos vetores embutidos:", output.shape)
