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
                 d_model:     int,
                 n_layers:    int,
                 n_heads:     int,
                 d_ff:        int,
                 max_len:     int):
        # --- parâmetros aprendíveis ---
        self.Wemb_src = create_embedding(vocab_size, d_model)             # (V, D)
        self.Wemb_tgt = create_embedding(vocab_size, d_model)             # (V, D)
        self.pe       = get_positional_encoding(max_len, d_model)         # (1, L, D)
        self.enc      = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.dec      = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.Wout     = xp.random.randn(d_model, vocab_size, dtype=xp.float32) / math.sqrt(d_model)

        # cache para forward/backward
        self._cache = {}

    def forward(self,
                src_ids:  xp.ndarray,                  # (B, Lsrc)
                tgt_ids:  xp.ndarray,                  # (B, Ltgt)
                src_mask: xp.ndarray | None = None,    # (B, Lsrc, 1) ou None
                tgt_mask: xp.ndarray | None = None     # (B, Ltgt, Ltgt) ou None
               ) -> xp.ndarray:                        # (B, Ltgt, V)
        B, Ls = src_ids.shape
        _, Lt = tgt_ids.shape

        # 1) Embeddings + Positional Encoding
        Es = embed_tokens(src_ids, self.Wemb_src) * math.sqrt(self.Wemb_src.shape[1])
        Es = Es + self.pe[:, :Ls, :]
        Et = embed_tokens(tgt_ids, self.Wemb_tgt) * math.sqrt(self.Wemb_tgt.shape[1])
        Et = Et + self.pe[:, :Lt, :]

        # armazena no cache
        self._cache['src_ids']  = src_ids
        self._cache['tgt_ids']  = tgt_ids
        self._cache['src_mask'] = src_mask
        self._cache['tgt_mask'] = tgt_mask
        self._cache['Es']       = Es
        self._cache['Et']       = Et

        # 2) Encoder Stack
        xs = Es
        for i, layer in enumerate(self.enc):
            xs = layer.forward(xs, src_mask)
            self._cache[f'enc_out_{i}'] = xs

        # 3) Decoder Stack
        xt = Et
        # armazena saída antes de qualquer decoder para backward de embeddings
        self._cache['g_before_decoder'] = xt
        for i, layer in enumerate(self.dec):
            xt = layer.forward(xt, xs, tgt_mask, src_mask)
            self._cache[f'dec_out_{i}'] = xt

        # 4) Projeção final
        self._cache['xt_final'] = xt
        flat = xt.reshape(-1, xt.shape[-1])   # (B*Lt, D)
        logits = flat @ self.Wout             # (B*Lt, V)
        return logits.reshape(B, Lt, -1)      # (B, Lt, V)

    def get_parameters_dict(self) -> dict[str, xp.ndarray]:
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

        return params

    def backward(self,
                 logits:   xp.ndarray,  # (B, Lt, V)
                 tgt_out:  xp.ndarray,  # (B, Lt)
                 pad_idx:  int,
                 epsilon:  float = 0.1
                ) -> list[xp.ndarray]:
        """
        Backprop completo:
          1) dL/dlogits via label_smoothing_grad
          2) dL/dWout  = xt^T · dlogits
          3) dL/dxt     = dlogits · Wout^T
          4) retorna lista de grads na mesma ordem de get_parameters_dict().values()
        """
        B, Lt, V = logits.shape
        D = self.Wout.shape[0]

        # 1) gradiente dL/dlogits
        dlogits = label_smoothing_grad(logits, tgt_out, pad_idx, epsilon)  # (B, Lt, V)

        # 2) gradiente Wout
        xt = self._cache['xt_final']                   # (B, Lt, D)
        flat_xt   = xt.reshape(-1, D)                  # (B*Lt, D)
        flat_dlog = dlogits.reshape(-1, V)             # (B*Lt, V)
        grad_Wout = flat_xt.T @ flat_dlog              # (D, V)

        # 3) constrói dicionário de grads
        params = self.get_parameters_dict()
        grads  = { name: xp.zeros_like(val) for name, val in params.items() }
        grads["Wout"] = grad_Wout

        # 4) devolve lista de grads na ordem
        return [grads[name] for name in params.keys()]
