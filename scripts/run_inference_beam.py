import os
import sys
import numpy as onp
# Adiciona o diretório raiz ao path para encontrar o módulo 'transformer'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from transformer.pipeline.common import xp
from transformer.pipeline.transformer_manual import Transformer
# 1. IMPORTAÇÃO MODULAR DO BEAM SEARCH
from transformer.pipeline.inference.beam_search import beam_search

# --- CONFIGURAÇÃO ---
DEBUG = False

if DEBUG:
    # Parâmetros para debug
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 512
    MAX_LEN = 64
    CHECKPOINT = "checkpoints_debug/epoch10.npz"
else:
    # Parâmetros do modelo principal
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    MAX_LEN = 64
    CHECKPOINT = "checkpoints/modelo_epoch_58.npz"

# --- CARREGAR TOKENIZER E MODELO ---
print("-> Carregando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    trust_remote_code=True
)
vocab_size = tokenizer.vocab_size

print("-> Construindo o modelo Transformer...")
model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=0.0 # Dropout é desativado na inferência
)

# --- 3. CARREGAR CHECKPOINT ---
print(f"-> Carregando checkpoint: {CHECKPOINT}")
ckpt = onp.load(CHECKPOINT)
model_params = model.get_parameters_dict()

# Mapeamento de abreviações para nomes de atributos reais
attr_map = {
    'sa': 'self_attn',
    'ca': 'cross_attn',
    'ffn': 'ffn'
}

for name in model_params.keys():
    if name in ckpt:
        # Primeiro, trata os parâmetros de nível raiz, como antes
        if hasattr(model, name):
            setattr(model, name, xp.asarray(ckpt[name]))
            continue

        parts = name.split('_')
        
        # 1. O caminho para o container tem sempre as 3 primeiras partes.
        # Ex: para 'enc_0_sa_W_q', o caminho é ['enc', '0', 'sa'].
        # Ex: para 'enc_0_norm1_gamma', o caminho é ['enc', '0', 'norm1'].
        path_parts = parts[:3]
        
        # 2. O nome do parâmetro final é todo o resto, unido de volta por '_'.
        # Ex: para 'enc_0_sa_W_q', o nome final é 'W_q'.
        # Ex: para 'enc_0_norm1_gamma', o nome final é 'gamma'.
        param_name = '_'.join(parts[3:])

        # Navega até o objeto container usando o caminho
        target_obj = model
        for part in path_parts:
            if part.isdigit():
                target_obj = target_obj[int(part)]
            else:
                attr_name = attr_map.get(part, part)
                target_obj = getattr(target_obj, attr_name)

        # Define o atributo no objeto container usando o nome final
        setattr(target_obj, param_name, xp.asarray(ckpt[name]))
    else:
        print(f"!! Atenção: Parâmetro '{name}' não encontrado no checkpoint.")


# 2. FUNÇÃO "HELPER" PARA TRADUÇÃO
def translate(text: str, beam_size: int = 4, alpha: float = 0.6) -> str:
    """
    Orquestra a tokenização, chamada do beam_search e decodificação.
    """
    # Tokeniza a sentença de entrada
    enc = tokenizer(
        [text], return_tensors="np", padding="max_length",
        truncation=True, max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])
    
    # Cria a máscara de padding da entrada
    pad_id = tokenizer.pad_token_id
    src_mask = (src_ids == pad_id)[:, None, None, :]

    # Define tokens especiais de início e fim
    sos_id = tokenizer.bos_token_id or tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.eos_token_id

    # Chama a função importada de beam_search
    result_tokens = beam_search(
        model=model,
        src_ids=src_ids,
        src_mask=src_mask,
        max_len=MAX_LEN,
        sos_id=sos_id,
        eos_id=eos_id,
        beam_size=beam_size,
        alpha=alpha
    )
    
    # Decodifica o resultado para texto
    return tokenizer.decode(result_tokens, skip_special_tokens=True)


# 3. EXECUÇÃO DA INFERÊNCIA
if __name__ == "__main__":
    prompts = [
        "Hello, how are you?",
        "This is a test of the Transformer implementation.",
        "Machine translation is fun!",
        "Do you know my name",
        "My car",
        "Everybody says my name is Nabucodonosor",
        "The cat sat on the book.",
        "What is your project about?",
        "The wolf",
        "The parrot",
        "The immortal",
        "The cat",
        "The dog",
        "To be or not to be, that is the question.",
        "I have been learning to play the guitar for three years.",
        
        
        "The paper 'Attention Is All You Need' is foundational for modern NLP.",
        "I would like to order a pizza with extra cheese, please.",
        "Can you please tell me where the nearest train station is?",
        "Never look a gift horse in the mouth.",
        "The book is on the table."
        
        
        
        "Good morning, I would like a cup of coffee.",
        "Where is the nearest train station?",
        "I don't understand what you are saying.",
        "How much does this cost?",

        # Sentenças com diferentes tempos verbais
        "I will travel to Brazil next year.",
        "She had already finished her homework when I called.",
        "He is learning how to code in Python.",

        # Sentenças mais complexas
        "Although it was raining, we decided to go for a walk in the park.",
        "The attention mechanism is one of the key components of this model.",
        
        "A bright light appeared in the sky.",
    ]

    print("--- Iniciando Traduções com Beam Search ---\n")
    for prompt in prompts:
        translation = translate(prompt)
        print(f"> EN: {prompt}")
        print(f"< PT: {translation}\n")