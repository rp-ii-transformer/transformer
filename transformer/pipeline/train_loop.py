from .loss                import label_smoothing_loss
from .optimizer_scheduler import Adam, noam_schedule
from .common              import xp

def train_epoch(model, data_loader, pad_idx, d_model, warmup):
    """
    Executa uma época de treinamento do Transformer puro NumPy/CuPy.
    """
    params = list(model.get_parameters_dict().values())
    optim  = Adam(params, lr=1.0, betas=(0.9,0.98))

    total_loss   = 0.0
    step         = 0
    batch_count  = 0

    for src, tgt in data_loader:
        step += 1
        batch_count += 1

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model.forward(src, tgt_input, None, None)
        loss   = label_smoothing_loss(logits, tgt_output, pad_idx)
        total_loss += float(loss)

        grads      = model.backward(logits, tgt_output, pad_idx)
        optim.lr   = noam_schedule(d_model, warmup, step)
        optim.step(grads)

        # libera memória cupy (se for o caso)
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # retorna perda média por batch
    return total_loss / batch_count
