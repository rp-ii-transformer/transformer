from .common import xp
import math
from .softmax import softmax

def scaled_dot_product_attention(q, k, v, mask=None):
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

    Note:        
    Calcula a atenção do tipo "scaled dot-product".
    q, k, v: (B, nh, T, dk)
    mask: (B, 1, T, T) ou (B, 1, 1, T) - broadcastable
    Retorna (context, attn_weights)
    """
    # last dimension size
    dk = q.shape[-1]

    # (..., seq_len_q, seq_len_k)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dk)  # (B, nh, T, T)

    # apply mask (e.g. for padding or causal masking)
    if mask is not None:
        scores = xp.where(mask, -1e9, scores)

    attn_weights = softmax(scores)  # (B, nh, T, T)
    context = attn_weights @ v      # (B, nh, T, dv)

    # Cache para o backward pass
    cache = (q, k, v, attn_weights)
    return context, cache



def scaled_dot_product_attention_backward(d_context, cache):
    """
    Retropropagação para a scaled_dot_product_attention.
    d_context: (B, nh, T, dv)
    cache: tupla com (q, k, v, attn_weights)
    Retorna (dq, dk, dv)
    """
    q, k, v, attn_weights = cache
    dk = q.shape[-1]

    # Gradiente em relação a v
    # dL/dv = attn_weights^T * dL/d_context
    dv = attn_weights.transpose(0, 1, 3, 2) @ d_context

    # Gradiente em relação a attn_weights
    # dL/d_attn_weights = dL/d_context * v^T
    d_attn_weights = d_context @ v.transpose(0, 1, 3, 2)

    # Gradiente em relação a scores (passando pelo softmax)
    # dL/d_scores = (dL/d_attn_weights - sum(dL/d_attn_weights * attn, axis=-1)) * attn
    d_scores = (d_attn_weights - xp.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)) * attn_weights

    # Gradiente em relação a k
    # dL/dk = (dL/d_scores^T * q) / sqrt(dk)
    d_k = (d_scores.transpose(0, 1, 3, 2) @ q) / math.sqrt(dk)

    # Gradiente em relação a q
    # dL/dq = (dL/d_scores * k) / sqrt(dk)
    d_q = (d_scores @ k) / math.sqrt(dk)

    return d_q, d_k, dv
