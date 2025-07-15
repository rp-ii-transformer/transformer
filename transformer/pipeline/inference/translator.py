from .greedy_search import greedy_decode
from .beam_search   import beam_search
from ..common       import xp

class Translator:
    def __init__(self, model, stoi, itos, pad_idx, sos_idx, eos_idx):
        self.model  = model
        self.stoi   = stoi
        self.itos   = itos
        self.pad    = pad_idx
        self.sos    = sos_idx
        self.eos    = eos_idx

    def translate(self, sentence, method="greedy", **kw):
        # 1) tokeniza + transforma em índices com xp
        tokens = sentence.split()
        idxs = [ self.stoi.get(t, self.stoi.get("<unk>", self.pad)) for t in tokens ]
        # cria array (1, seq_len)
        src_idxs = xp.array(idxs, dtype=xp.int64)[None, :]
        # máscara de padding (1, seq_len, 1)
        src_mask = (src_idxs != self.pad)[..., None]

        # 2) escolhe método
        if method == "greedy":
            # greedy_decode espera numpy, então converta de volta:
            import numpy as _onp
            np_src = _onp.asarray(src_idxs.get() if hasattr(src_idxs, "get") else src_idxs)
            np_mask = _onp.asarray(src_mask.get() if hasattr(src_mask, "get") else src_mask)
            out_idxs = greedy_decode(self.model, np_src, np_mask,
                                     max_len=kw.get("max_len", 50),
                                     start_idx=self.sos)
        else:
            import numpy as _onp
            np_src = _onp.asarray(src_idxs.get() if hasattr(src_idxs, "get") else src_idxs)
            np_mask = _onp.asarray(src_mask.get() if hasattr(src_mask, "get") else src_mask)
            beams = beam_search(self.model, np_src, np_mask,
                                max_len=kw.get("max_len", 50),
                                start_idx=self.sos,
                                beam_size=kw.get("beam_size", 5))
            out_idxs = beams[0][0]  # melhor beam

        # 3) converte índices em tokens, parando no <eos>
        tokens_out = []
        # out_idxs shape: (1, tgt_len) ou lista de listas
        for idx in (out_idxs[0] if method=="greedy" else out_idxs):
            if idx == self.eos:
                break
            tokens_out.append(self.itos[idx])
        return " ".join(tokens_out)
