from src.transformer.pipeline.common import xp
from src.transformer.pipeline.training.loss import label_smoothing_loss

def main():
    # Parâmetros do teste
    vocab_size = 5
    batch      = 2
    seq_len    = 3
    epsilon    = 0.1
    pad_idx    = -1  # não há padding neste teste

    # logits uniformes ⇒ devem gerar ln(vocab_size)
    logits = xp.zeros((batch, seq_len, vocab_size), dtype=xp.float32)

    # targets qualquer, aqui 0,1,2 repetidos
    targets = xp.tile(xp.arange(seq_len, dtype=xp.int32)[None, :], (batch, 1))

    loss = label_smoothing_loss(logits, targets, pad_idx, epsilon)
    expected = float(xp.log(vocab_size))

    print(f"Loss calculada: {float(loss):.6f}")
    print(f"Loss esperada : {expected:.6f}")

    # Check simples
    assert abs(float(loss) - expected) < 1e-5, "🚨 label_smoothing_loss diverge do esperado!"

    print("✅ Teste passou!")

if __name__ == "__main__":
    main()
