import numpy as np
from softmax import softmax

def main():
    array = [10, 5, 1, 12]
    probs = softmax(array)

    print("Array (pontuações):", ", ".join(str(x) for x in array))
    print("Softmax (probabilidades): [", ", ".join(str(p) for p in probs), "]")
    print("Soma das probabilidades:", np.sum(probs))

if __name__ == "__main__":
    main()
