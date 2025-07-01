import numpy as np
from softmax import softmax

def main():
    logits = [10, 5, 1, 12]
    probs = softmax(logits)

    print("Logits (pontuações):", ", ".join(str(x) for x in logits))
    print("Softmax (probabilidades): [", ", ".join(str(p) for p in probs), "]")
    print("Soma das probabilidades:", np.sum(probs))

if __name__ == "__main__":
    main()
