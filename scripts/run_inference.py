import os
import sys
import numpy as onp
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import AutoTokenizer
from transformer.pipeline.common import xp
from transformer.pipeline.transformer_manual import Transformer

# --- 1. CONFIGURAÇÃO ---
DEBUG = False

if DEBUG:
    # Parâmetros para modelo treinado com 'train_debug.py'
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 512
    MAX_LEN = 64
    CHECKPOINT = "checkpoints_debug/epoch10.npz"
else:
    # Parâmetros para modelo treinado com 'train.py'
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    MAX_LEN = 64
    CHECKPOINT = "checkpoints/epoch58.npz"

# --- 2. CARREGAR TOKENIZER E MODELO ---
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    trust_remote_code=True
)
vocab_size = tokenizer.vocab_size

model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN
)

# --- 3. CARREGAR CHECKPOINT ---
print(f"-> Carregando checkpoint: {CHECKPOINT}")
ckpt = onp.load(CHECKPOINT)
top_level_params = ["Wemb_src", "Wemb_tgt", "Wout"]

for name in model.get_parameters_dict().keys():
    if name in ckpt:
        if name in top_level_params:
            setattr(model, name, xp.asarray(ckpt[name]))
        else:
            parts = name.split('_')
            container_path_parts = parts[:3]
            param_name = '_'.join(parts[3:])
            target_obj = model
            for part in container_path_parts:
                if part.isdigit():
                    target_obj = target_obj[int(part)]
                else:
                    attr_map = {'sa': 'self_attn', 'ca': 'cross_attn', 'ffn': 'ffn'}
                    if 'norm' in part:
                        target_obj = getattr(target_obj, part)
                    else:
                        target_obj = getattr(target_obj, attr_map.get(part, part))
            setattr(target_obj, param_name, xp.asarray(ckpt[name]))
    else:
        print(f"!! Atenção: Parâmetro '{name}' não encontrado no checkpoint.")

print("-> Modelo carregado com sucesso.")
print(f"→ Usando {'DEBUG' if DEBUG else 'FULL'} checkpoint: {CHECKPOINT}\n")

# --- 4. FUNÇÃO DE GERAÇÃO GREEDY ---
def greedy_generate(text: str) -> str:
    # Tokeniza entrada em inglês
    enc = tokenizer(
        [text], return_tensors="np", padding="max_length",
        truncation=True, max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])
    pad_id = tokenizer.pad_token_id

    # Máscara de padding da fonte (mesmo formato do treino)
    src_mask = (src_ids == pad_id)[:, None, None, :]

    # Define tokens especiais
    sos_id = tokenizer.bos_token_id
    if sos_id is None:
        sos_id = tokenizer.convert_tokens_to_ids("<s>") if "<s>" in tokenizer.get_vocab() else pad_id
    eos_id = tokenizer.eos_token_id

    # Inicia target com <sos>
    tgt = xp.array([[sos_id]], dtype=xp.int32)

    for _ in range(MAX_LEN - 1):
        tgt_len = tgt.shape[1]

        # Máscara de padding do target + máscara causal
        tgt_pad_mask = (tgt == pad_id)[:, None, None, :]
        causal_mask = xp.triu(xp.ones((1, 1, tgt_len, tgt_len), dtype=xp.bool_), k=1)
        tgt_mask = tgt_pad_mask | causal_mask

        # Forward sem dropout
        logits = model.forward(src_ids, tgt, src_mask, tgt_mask, training=False)

        next_id = int(xp.argmax(logits[0, -1]))
        tgt = xp.concatenate([tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1)

        if eos_id is not None and next_id == eos_id:
            break

    # Remove o SOS inicial e decodifica
    out_ids = [int(i) for i in tgt[0, 1:].tolist()]
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# --- 5. EXEMPLOS ---
for prompt in [
    "Hello, how are you?",
    "This is a test of the Transformer implementation.",
    "Machine translation is fun!",
    "Do you know my name",
    "Name",
    "My car",
    "Everybody says my name is Nabucodonosor",
    "Qual é o seu nome?",
    "Qual é o seu projeto?",
    "O lobo diz AUUUUUUUUUUUUUUUUUUUUUU AU AU AUUUUUUUUUUUUUUUUUU",
    "O lobo",
    "O papagaio",
    "O imortal",
    "O gato",
    "O cachorro"
]:
    print(f"> {prompt}")
    print(f"< {greedy_generate(prompt)}\n")
