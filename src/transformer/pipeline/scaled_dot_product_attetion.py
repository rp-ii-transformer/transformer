from softmax import softmax
import torch
import math

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
    # dimension of key (from last key)
    d_k = query.shape[-1]

    # matrix multiplication. key is transposed for correct dot-product
    scores = torch.matmul(query, key.transpose(-2, -1))

    # scale the dot_product
    scaled_scores = scores / math.sqrt(d_k)

    # apply mask if it exists
    if mask is not None:
        scaled_scores = scaled_scores + mask

    attention_weights = softmax(scaled_scores)

    output = torch.matmul(attention_weights, value)

    return output
