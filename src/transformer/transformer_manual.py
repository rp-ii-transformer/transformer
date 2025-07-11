import torch

from src.transformer.pipeline.token_embedding import token_embedding_start


def forward_pass_transformer_manual(token_ids, vocab_size, d_model):
    print("\n🔹 Iniciando Token Embedding...")
    embedded_tokens = token_embedding_start(vocab_size, d_model, token_ids)

    print("🔹 Saída do Token Embedding:", embedded_tokens.shape)

    # TODO Positional Encoding MOCK
    encoded_with_position = embedded_tokens
    # TODO Encoder/Decoder MOCK
    decoded_output = encoded_with_position
    # TODO Linear + Softmax para simular logits MOCK
    batch_size, seq_len, _ = decoded_output.shape
    mock_logits = torch.rand((batch_size, seq_len, vocab_size), device=decoded_output.device)

    return mock_logits
