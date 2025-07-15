from .common import xp
import math
from .softmax import softmax

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention as per 'Attention is All You Need':
    Computes dot products of query with all keys, divide each by sqrt(d_k),
    and apply a softmax function to obtains the weights of the values.

    Args:
        query (numpy.ndarray): Query matrix, shape (batch_size, seq_length_query, d_k).
        key (numpy.ndarray): Key matrix, shape (batch_size, seq_length_key, d_k).
        value (numpy.ndarray): Value matrix, shape (batch_size, seq_length_key, d_v).
        mask (numpy.ndarray, optional): Mask for attention scores, shape (batch_size, seq_length_query, seq_length_key).

    Returns:
        numpy.ndarray: Attention output, shape (batch_size, seq_length_query, d_v).
    """
    # last dimension size
    d_k = query.shape[-1]

    # (..., seq_len_q, seq_len_k)
    scores = xp.matmul(query, key.swapaxes(-1, -2))

    # scale
    scores = scores / math.sqrt(d_k)

    # apply mask (e.g. for padding or causal masking)
    if mask is not None:
        scores = scores + mask

    # softmax over key sequence length
    weights = softmax(scores)

    # weighted sum of values: (..., seq_len_q, d_v)
    output = xp.matmul(weights, value)
    return output