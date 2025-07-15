import os
import sys
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import AutoTokenizer
from transformer.pipeline.common            import xp
from transformer.pipeline.transformer_manual import Transformer

# ─────── CONFIGURAÇÃO ───────────────────────────────────────────────
DEBUG = False

if DEBUG:
    D_MODEL    = 128
    N_LAYERS   = 2
    N_HEADS    = 4
    D_FF       = 512
    MAX_LEN    = 64
    CHECKPOINT = "checkpoints_debug/epoch3.npz" 
else:
    D_MODEL    = 512
    N_LAYERS   = 6
    N_HEADS    = 8
    D_FF       = 2048
    MAX_LEN    = 64
    CHECKPOINT = "checkpoints/epoch10.npz"
# ───────────────────────────────────────────────────────────────────

# 1) Carrega tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    trust_remote_code=True
)
vocab_size = tokenizer.vocab_size

# 2) Instancia o Transformer
model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN
)

# 3) Carrega checkpoint NumPy
ckpt = onp.load(CHECKPOINT)
model.Wemb_src = xp.asarray(ckpt["Wemb_src"])
model.Wemb_tgt = xp.asarray(ckpt["Wemb_tgt"])
model.Wout     = xp.asarray(ckpt["Wout"])
# Se tiver treinado mais parametros na versão FULL:
# ex: model.enc[0].self_attn.W_q = xp.asarray(ckpt["enc_0_Wq"]) etc.

print(f"→ Usando {'DEBUG' if DEBUG else 'FULL'} checkpoint: {CHECKPOINT}")
print("GPU disponível?", hasattr(xp, "get_default_memory_pool"))

# 4) Função de geração *greedy*
def greedy_generate(text: str) -> str:
    # tokenização
    enc = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])                 # (1, S)
    pad_id   = tokenizer.pad_token_id

    # --- Construção da máscara para cross-attention ---
    # queremos uma máscara que seja (B, 1, 1, S) para broadcast até (B, nh, T, S)
    is_pad   = (src_ids == pad_id)                         # (1, S)
    mask_vals = xp.where(is_pad, -1e9, 0.0).astype(xp.float32)  # (1, S)
    src_mask = mask_vals[:, None, None, :]                  # (1, 1, 1, S)

    # --- Prepara SOS/EOS ---
    sos = (tokenizer.bos_token_id or
           tokenizer.cls_token_id or pad_id)
    eos = (tokenizer.eos_token_id or
           tokenizer.sep_token_id or sos)

    # começa a sequência alvo com só [SOS]
    tgt = xp.array([[sos]], dtype=xp.int32)                 # (1, 1)

    # gera token a token
    for _ in range(MAX_LEN - 1):
        logits = model.forward(src_ids, tgt, src_mask, None)  # (1, T, V)
        next_id = int(xp.argmax(logits[0, -1]))               # escolhe próximo
        tgt = xp.concatenate(
            [tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1
        )
        if next_id == eos:
            break

    # decodifica removendo tokens especiais
    out_ids = [int(i) for i in tgt[0, 1:].tolist()]
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# 5) Exemplos
for prompt in [
    "Hello, how are you?",
    "This is a test of the Transformer implementation.",
    "Machine translation is fun!"
]:
    print(f"> {prompt}")
    print(f"< {greedy_generate(prompt)}\n")
