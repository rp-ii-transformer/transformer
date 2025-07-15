from transformer.pipeline.common import is_gpu
from transformer.pipeline.transformer_manual import Transformer

def transformer_start(vocab_size, **kwargs) -> Transformer:
    model = Transformer(vocab_size=vocab_size, **kwargs)
    device = "GPU" if is_gpu() else "CPU"
    print(f"► Transformer manual inicializado em {device}")
    return model
