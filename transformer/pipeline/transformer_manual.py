from .common               import xp
from .token_embedding      import create_embedding, embed_tokens
from .positional_encoding import get_positional_encoding, add_positional_encoding
from .encoder_layer        import EncoderLayer
from .decoder_layer        import DecoderLayer
from .loss                 import label_smoothing_grad
import math

class Transformer:
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 d_ff: int,
                 max_len: int,
                 dropout: float = 0.1):  # Dropout como parâmetro do modelo

        # --- parâmetros aprendíveis ---
        self.Wemb_src = create_embedding(vocab_size, d_model)             # (V, D)
        self.Wemb_tgt = create_embedding(vocab_size, d_model)             # (V, D)
        self.pe       = get_positional_encoding(max_len, d_model)         # (1, L, D)
        self.enc = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.dec = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.Wout = xp.random.randn(d_model, vocab_size, dtype=xp.float32) / math.sqrt(d_model)

        # cache para forward/backward
        self._cache = {}

    def forward(self,
                src_ids:  xp.ndarray,                  # (B, Lsrc)
                tgt_ids:  xp.ndarray,                  # (B, Ltgt)
                src_mask: xp.ndarray | None = None,    # (B, Lsrc, 1) ou None
                tgt_mask: xp.ndarray | None = None,    # (B, Ltgt, Ltgt) ou None
                training: bool = True                  # se True, aplica dropout
               ) -> xp.ndarray:                        # (B, Ltgt, V)
        B, Ls = src_ids.shape
        _, Lt = tgt_ids.shape

        # 1) Embeddings + Positional Encoding
        Es = embed_tokens(src_ids, self.Wemb_src) * math.sqrt(self.Wemb_src.shape[1])
        Es = Es + self.pe[:, :Ls, :]
        Et = embed_tokens(tgt_ids, self.Wemb_tgt) * math.sqrt(self.Wemb_tgt.shape[1])
        Et = Et + self.pe[:, :Lt, :]

        # armazena valores iniciais no cache para o backward
        self._cache['src_ids'] = src_ids
        self._cache['tgt_ids'] = tgt_ids
        self._cache['Es'] = Es
        self._cache['Et'] = Et

        # 2) Pilha do Encoder
        xs = Es
        for i, layer in enumerate(self.enc):
            # CORREÇÃO: Passa o flag 'training' e armazena o cache da camada
            xs, layer_cache = layer.forward(xs, src_mask, training=training)
            self._cache[f'enc_layer_{i}_cache'] = layer_cache

        memory = xs # A saída final do encoder é a memória para o decoder

        # 3) Pilha do Decoder
        xt = Et
        for i, layer in enumerate(self.dec):
            # CORREÇÃO: Passa 'memory', o flag 'training' e armazena o cache da camada
            xt, layer_cache = layer.forward(xt, memory, tgt_mask, src_mask, training=training)
            self._cache[f'dec_layer_{i}_cache'] = layer_cache

        # 4) Projeção Final
        self._cache['xt_final'] = xt
        flat = xt.reshape(-1, xt.shape[-1])   # (B*Lt, D)
        logits = flat @ self.Wout             # (B*Lt, V)
        return logits.reshape(B, Lt, -1)      # (B, Lt, V)

    def get_parameters_dict(self) -> dict[str, xp.ndarray]:
        """
        Coleta TODOS os parâmetros aprendíveis do modelo de forma hierárquica.
        """
        params = {
            "Wemb_src": self.Wemb_src,
            "Wemb_tgt": self.Wemb_tgt,
            "Wout": self.Wout,
        }

        # Coleta parâmetros de todas as camadas do Encoder
        for i, layer in enumerate(self.enc):
            for p_name, p_val in layer.self_attn.get_parameters().items():
                params[f"enc_{i}_sa_{p_name}"] = p_val
            for p_name, p_val in layer.ffn.get_parameters().items():
                params[f"enc_{i}_ffn_{p_name}"] = p_val
            params[f"enc_{i}_norm1_gamma"] = layer.norm1.gamma
            params[f"enc_{i}_norm1_beta"] = layer.norm1.beta
            params[f"enc_{i}_norm2_gamma"] = layer.norm2.gamma
            params[f"enc_{i}_norm2_beta"] = layer.norm2.beta

        # Coleta parâmetros de todas as camadas do Decoder
        for i, layer in enumerate(self.dec):
            for p_name, p_val in layer.self_attn.get_parameters().items():
                params[f"dec_{i}_sa_{p_name}"] = p_val
            for p_name, p_val in layer.cross_attn.get_parameters().items():
                params[f"dec_{i}_ca_{p_name}"] = p_val
            for p_name, p_val in layer.ffn.get_parameters().items():
                params[f"dec_{i}_ffn_{p_name}"] = p_val
            params[f"dec_{i}_norm1_gamma"] = layer.norm1.gamma
            params[f"dec_{i}_norm1_beta"] = layer.norm1.beta
            params[f"dec_{i}_norm2_gamma"] = layer.norm2.gamma
            params[f"dec_{i}_norm2_beta"] = layer.norm2.beta
            params[f"dec_{i}_norm3_gamma"] = layer.norm3.gamma
            params[f"dec_{i}_norm3_beta"] = layer.norm3.beta

        return params

    '''def get_parameters_dict_old(self) -> dict[str, xp.ndarray]:
        params = {
            "Wemb_src": self.Wemb_src,
            "Wemb_tgt": self.Wemb_tgt,
            "Wout":     self.Wout,
        }
        # Encoder layers
        for i, layer in enumerate(self.enc):
            a = layer.self_attn
            params[f"enc_{i}_Wq"]  = a.W_q
            params[f"enc_{i}_Wk"]  = a.W_k
            params[f"enc_{i}_Wv"]  = a.W_v
            params[f"enc_{i}_Wo"]  = a.W_o
            f = layer.ffn
            params[f"enc_{i}_ffn_W1"] = f.W1
            params[f"enc_{i}_ffn_b1"] = f.b1
            params[f"enc_{i}_ffn_W2"] = f.W2
            params[f"enc_{i}_ffn_b2"] = f.b2

        # Decoder layers
        for i, layer in enumerate(self.dec):
            sa, ca = layer.self_attn, layer.cross_attn
            params[f"dec_{i}_sa_Wq"] = sa.W_q
            params[f"dec_{i}_sa_Wk"] = sa.W_k
            params[f"dec_{i}_sa_Wv"] = sa.W_v
            params[f"dec_{i}_sa_Wo"] = sa.W_o
            params[f"dec_{i}_ca_Wq"] = ca.W_q
            params[f"dec_{i}_ca_Wk"] = ca.W_k
            params[f"dec_{i}_ca_Wv"] = ca.W_v
            params[f"dec_{i}_ca_Wo"] = ca.W_o
            f = layer.ffn
            params[f"dec_{i}_ffn_W1"] = f.W1
            params[f"dec_{i}_ffn_b1"] = f.b1
            params[f"dec_{i}_ffn_W2"] = f.W2
            params[f"dec_{i}_ffn_b2"] = f.b2

        return params'''

    def backward(self,
                 logits: xp.ndarray,  # (B, Lt, V)
                 tgt_out: xp.ndarray,  # (B, Lt)
                 pad_idx: int,
                 epsilon: float = 0.1
                ) -> list[xp.ndarray]:
        """
        Backprop completo, calculando todos os gradientes via chain rule.
        """
        # 1. Gradiente inicial na saída do modelo (dL/dlogits)
        dlogits = label_smoothing_grad(logits, tgt_out, pad_idx, epsilon) # (B, Lt, V)

        # 2. Gradientes da camada de projeção final (dL/dWout e dL/dxt_final)
        xt_final = self._cache['xt_final']  # (B, Lt, D)
        B, Lt, D = xt_final.shape
        _, _, V = logits.shape

        flat_xt = xt_final.reshape(-1, D)    # (B*Lt, D)
        flat_dlog = dlogits.reshape(-1, V)   # (B*Lt, V)

        grad_Wout = flat_xt.T @ flat_dlog    # (D, V)
        dxt = flat_dlog @ self.Wout.T        # (B*Lt, D)
        dxt = dxt.reshape(B, Lt, D)          # (B, Lt, D) -> Gradiente na saída do decoder stack

         # Inicializa dicionário de gradientes
        params_keys = self.get_parameters_dict().keys()
        grads = {name: xp.zeros_like(self.get_parameters_dict()[name]) for name in params_keys}
        grads["Wout"] = grad_Wout

        # 3. Retropropagação pelo Decoder
        d_memory_total = xp.zeros_like(self._cache['Es']) # Gradiente para a memória do encoder
        for i in reversed(range(len(self.dec))):
            layer_cache = self._cache[f'dec_layer_{i}_cache']
            dxt, d_memory_from_dec, dec_layer_grads = self.dec[i].backward(dxt, layer_cache)
            d_memory_total += d_memory_from_dec
            for name, grad in dec_layer_grads.items():
                grads[f"dec_{i}_{name}"] = grad

        # 4. Retropropagação pelo Embedding do Target
        grad_Wemb_tgt = xp.zeros_like(self.Wemb_tgt)
        d_Et = dxt * math.sqrt(self.Wemb_tgt.shape[1])
        xp.add.at(grad_Wemb_tgt, self._cache['tgt_ids'].reshape(-1), d_Et.reshape(-1, D))
        grads["Wemb_tgt"] = grad_Wemb_tgt

        # 5. Retropropagação pelo Encoder
        dxs = d_memory_total
        for i in reversed(range(len(self.enc))):
            layer_cache = self._cache[f'enc_layer_{i}_cache']
            dxs, enc_layer_grads = self.enc[i].backward(dxs, layer_cache)
            for name, grad in enc_layer_grads.items():
                grads[f"enc_{i}_{name}"] = grad
        
        # 6. Retropropagação pelo Embedding da Source
        grad_Wemb_src = xp.zeros_like(self.Wemb_src)
        d_Es = dxs * math.sqrt(self.Wemb_src.shape[1])
        xp.add.at(grad_Wemb_src, self._cache['src_ids'].reshape(-1), d_Es.reshape(-1, D))
        grads["Wemb_src"] = grad_Wemb_src
        
        # Retorna a lista de gradientes na mesma ordem dos parâmetros
        return [grads[name] for name in params_keys]