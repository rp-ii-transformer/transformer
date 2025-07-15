try:
    import cupy as xp
    _GPU = True
except ImportError:
    import numpy as xp
    _GPU = False

def is_gpu():
    return _GPU
