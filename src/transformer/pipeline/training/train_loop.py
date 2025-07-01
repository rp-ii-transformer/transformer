import numpy as np
from .loss import cross_entropy_loss
from .optimizer_scheduler import Adam, noam_schedule

def train_epoch(model, data_loader, pad_idx, d_model, warmup, params):
    """
    model: instância do seu Transformer (versão NumPy)
    data_loader: itera batches (src_tokens, tgt_tokens)
    params: lista de parâmetros (np.ndarray) do model
    """
    optimizer = Adam(params, lr=1.0, betas=(0.9,0.98))
    total_loss = 0
    step = 0

    for src, tgt in data_loader:
        step += 1
        # preparação dos inputs
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        # forward
        logits = model(src, tgt_input)           # (batch, seq_len-1, vocab_size)
        # loss
        loss = cross_entropy_loss(logits, tgt_output, pad_idx)
        total_loss += loss
        # backward (você precisa implementar gradiente manual ou aproximado)
        grads = model.backward()                  # stub: retorna grads para cada param
        # update lr do Adam
        lr = noam_schedule(d_model, warmup, step)
        optimizer.lr = lr
        # otimização
        optimizer.step(grads)

    return total_loss / len(data_loader)
