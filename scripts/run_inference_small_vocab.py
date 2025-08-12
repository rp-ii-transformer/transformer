import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer.pipeline.transformer_manual import Transformer
from transformer.pipeline.common import is_gpu, xp

# --- 1. CONFIGURAÇÃO (deve ser a mesma do treino) ---

CHECKPOINT_DIR = "checkpoints_small_vocab"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "epoch50.npz")
VOCAB_FILE = os.path.join(CHECKPOINT_DIR, "vocab.npy")

D_MODEL = 64
N_LAYERS = 2
N_HEADS = 2
D_FF = 128
MAX_LEN = 20

# --- 2. CARREGAR VOCABULÁRIO E MODELO ---

print(f"Carregando vocabulário de: {VOCAB_FILE}")
vocab_stoi = np.load(VOCAB_FILE, allow_pickle=True).item()
vocab_itos = {i: s for s, i in vocab_stoi.items()}
vocab_size = len(vocab_stoi)
pad_idx = vocab_stoi['<pad>']

print(f"Carregando modelo do checkpoint: {CHECKPOINT_FILE}")
model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN
)

ckpt = np.load(CHECKPOINT_FILE)
top_level_params = ["Wemb_src", "Wemb_tgt", "Wout"]

for name in model.get_parameters_dict().keys():
    if name in ckpt:
        if name in top_level_params:
            setattr(model, name, xp.asarray(ckpt[name]))
        else:
            # --- CORREÇÃO AQUI ---
            # Lógica de parsing mais robusta para o nome do parâmetro
            parts = name.split('_')
            # O caminho para o objeto que contém o parâmetro (ex: ['enc', '0', 'sa'])
            container_path_parts = parts[:3]
            # O nome final do parâmetro (ex: 'W_q', 'b1', 'norm1_gamma')
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
            
            # Define o atributo no objeto container encontrado
            setattr(target_obj, param_name, xp.asarray(ckpt[name]))
            # --- FIM DA CORREÇÃO ---
    else:
        print(f"Atenção: Parâmetro '{name}' não encontrado no checkpoint.")

print("Modelo carregado com sucesso.")

# --- 3. FUNÇÃO DE GERAÇÃO (INFERÊNCIA) ---

def greedy_translate(text: str) -> str:
    tokens = [vocab_stoi['<sos>']]
    for word in text.lower().split():
        tokens.append(vocab_stoi.get(word, vocab_stoi['<unk>']))
    tokens.append(vocab_stoi['<eos>'])

    padded_tokens = tokens + [pad_idx] * (MAX_LEN - len(tokens))
    src_ids = xp.asarray([padded_tokens[:MAX_LEN]], dtype=xp.int32)
    
    src_mask = (src_ids == pad_idx)[:, None, None, :]
    tgt = xp.array([[vocab_stoi['<sos>']]], dtype=xp.int32)

    for _ in range(MAX_LEN - 1):
        tgt_len = tgt.shape[1]
        tgt_causal_mask = xp.triu(xp.ones((1, 1, tgt_len, tgt_len), dtype=xp.bool_), k=1)
        
        logits = model.forward(src_ids, tgt, src_mask, tgt_causal_mask, training=False)
        
        next_id = int(xp.argmax(logits[0, -1]))
        tgt = xp.concatenate([tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1)
        
        if next_id == vocab_stoi['<eos>']:
            break
            
    output_ids = [int(i) for i in tgt[0].tolist()]
    words = [vocab_itos[i] for i in output_ids if i not in [pad_idx, vocab_stoi['<sos>'], vocab_stoi['<eos>']]]
    
    return " ".join(words)

# --- 4. TESTE A TRADUÇÃO ---

test_sentences = [
    "hello world hello world world hello",
    "how are you",
    "this is a test",
    "the book is on the table",
    "the book table",
    "i love",
    "machine",
    "translation",
    "i love machine translation",
    "hello!!",
]

print("\n--- INICIANDO TRADUÇÃO ---")
for sentence in test_sentences:
    translation = greedy_translate(sentence)
    print(f"> {sentence}")
    print(f"< {translation}\n")