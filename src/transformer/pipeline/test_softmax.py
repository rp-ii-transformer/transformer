import numpy as np
from softmax import softmax, cross_entropy_loss

def main():
    logits = [1, 2, 3]              # Saída do modelo (antes do softmax)
    probs = softmax(logits)         # Converte para probabilidades
    predicted_idx = np.argmax(probs)

    target_idx = 2  # valor correto índice 2 = (o valor 3)

    loss = cross_entropy_loss(probs, target_idx)

    print()
    print("Logits (pontuações):", ", ".join(str(x) for x in logits))
    print("Softmax (probabilidades): [", ", ".join(f"{p}" for p in probs), "]")
    print("Soma das probabilidades:", np.sum(probs))
    print(f"A maior probabilidade foi: {np.max(probs)}")
    print(f"Logit com maior probabilidade: {logits[predicted_idx]}")
    # print(f"Target correto (esperado): {logits[target_idx]}")
    print(f"Loss (erro): {loss:.4f}")
    print()
    print(f"A probabilidade do valor {logits[target_idx]} ser o correto é: {probs[target_idx] * 100:.2f}%")
    print()

if __name__ == "__main__":
    main()
