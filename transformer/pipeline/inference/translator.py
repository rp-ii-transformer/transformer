class Translator:
    def __init__(self, model, stoi, itos, pad_idx, sos_idx, eos_idx):
        self.model = model
        self.stoi  = stoi
        self.itos  = itos
        self.pad   = pad_idx
        self.sos   = sos_idx
        self.eos   = eos_idx

    def translate(self, sentence, method="greedy", **kw):
        # 1) tokeniza + indices
        tokens = sentence.split()
        src_idxs = xp.array([self.stoi.get(t, self.stoi["<unk>"]) for t in tokens])[None, :]
        src_mask = (src_idxs != self.pad)[..., None]  # ou máscara customizada

        # 2) escolhe método
        if method == "greedy":
            out_idxs = greedy_decode(self.model, src_idxs, src_mask,
                                     max_len=kw.get("max_len", 50),
                                     start_idx=self.sos)
        else:
            beams = beam_search(self.model, src_idxs, src_mask,
                                max_len=kw.get("max_len", 50),
                                start_idx=self.sos,
                                beam_size=kw.get("beam_size", 5))
            out_idxs = beams[0][0]  # melhor beam

        # 3) converte para tokens e para em <eos>
        tokens_out = []
        for idx in out_idxs[0] if method=="greedy" else out_idxs:
            if idx == self.eos: break
            tokens_out.append(self.itos[idx])
        return " ".join(tokens_out)
