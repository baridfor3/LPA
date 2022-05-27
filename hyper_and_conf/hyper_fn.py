# encoding=utf8

import tensorflow as tf
import numpy as np
import math
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def get_position_encoding(length,
                          hidden_size,
                          min_timescale=1.0,
                          max_timescale=1.0e4):
    """Return positional encoding.
  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


# def get_angles(pos, i, d_model):
#     angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#     return pos * angle_rates
#
#
# def get_position_encoding(position, d_model):
#     angle_rads = get_angles(
#         np.arange(position)[:, np.newaxis],
#         np.arange(d_model)[np.newaxis, :], d_model)
#
#     # apply sin to even indices in the array; 2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#     # apply cos to odd indices in the array; 2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#     pos_encoding = angle_rads[np.newaxis, ...]
#
#     return tf.cast(pos_encoding, dtype=tf.float32)


def get_learning_rate(learning_rate,
                      hidden_size,
                      step=1,
                      learning_rate_warmup_steps=16000):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    warmup_steps = float(learning_rate_warmup_steps)
    step = max(1.0, step)

    learning_rate *= (hidden_size**-0.5)
    learning_rate *= min(1, step / warmup_steps)
    learning_rate *= (0.5 / np.sqrt(max(step, warmup_steps)))
    return learning_rate


def scaled_dot_product_attention(q, k, v, mask=None, dropout=0, heads=1):
    """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """
 
    matmul_qk = tf.matmul(q, k,
                          transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)
    if dropout != 1 or dropout != 0:
        attention_weights = tf.nn.dropout(attention_weights, dropout)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def local_perm(inputs, is_masked, perm_size, seq_len, leak_ratio):
    # Generate permutation indices
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random.shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    # non-functional tokens
    non_func_tokens = tf.logical_not(
        tf.logical_or(tf.equal(inputs, 1), tf.equal(inputs, 0)))
    masked_tokens = tf.logical_and(is_masked, non_func_tokens)
    non_masked_or_func_tokens = tf.logical_not(masked_tokens)

    smallest_index = -2 * tf.ones([seq_len], dtype=tf.int64)

    # Similar to BERT, randomly leak some masked tokens
    if leak_ratio > 0:
        leak_tokens = tf.logical_and(
            masked_tokens,
            tf.random.uniform([seq_len], maxval=1.0) < leak_ratio)
        can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
    else:
        can_attend_self = non_masked_or_func_tokens
    to_index = tf.where(can_attend_self, smallest_index, index)
    from_index = tf.where(can_attend_self, to_index + 1, to_index)

    # For masked tokens, can attend if i > j
    # For context tokens, always can attend each other
    can_attend = from_index[:, None] > to_index[None, :]

    # In modeling, 1 indicates cannot attend. Hence, reverse the value here.
    perm_mask = 1.0 - tf.cast(can_attend, tf.float32)

    # Only masked tokens are included in the loss
    target_mask = tf.cast(masked_tokens, tf.float32)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = masked_tokens

    return perm_mask, target_mask, inputs_k, inputs_q
