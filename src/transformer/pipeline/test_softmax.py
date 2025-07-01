import numpy as np
from softmax import softmax

def main():
    logits = [1, 2, 3]
    probs = softmax(logits)
    idx = np.argmax(probs)

    print("Logits (pontuações):", ", ".join(str(x) for x in logits))
    print("Softmax (probabilidades): [", ", ".join(str(p) for p in probs), "]")
    print("Soma das probabilidades:", np.sum(probs))
    print("A maior probabilidade foi:", np.max(probs))
    print(f"Logit com maior probabilidade: {logits[idx]}")


if __name__ == "__main__":
    main()
