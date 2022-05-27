# encoder=utf8
from hyper_and_conf import hyper_util
import tensorflow as tf
from tensorflow.python.keras import regularizers
from hyper_and_conf import hyper_fn
L2_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def split_heads(x, heads):
    """Split x into different heads, and transpose the resulting value.
The tensor is transposed to insure the inner dimensions hold the correct
values during the matrix multiplication.
Args:
  x: A tensor with shape [batch_size, length, num_units]
Returns:
  A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
"""
    with tf.name_scope("split_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        num_units = tf.shape(x)[-1]
        # Calculate depth of last dimension after it has been split.
        depth = (num_units // heads)

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x, num_units):
    """Combine tensor that has been split.
Args:
  x: A tensor [batch_size, num_heads, length, num_units/num_heads]
Returns:
  A tensor with shape [batch_size, length, num_units]
"""
    with tf.name_scope("combine_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x,
                         [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, num_units])


def scaled_dot_product_attention(q, k, v, heads=0, mask=None, dropout=0):
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
    num_units = tf.cast(tf.shape(q)[-1], tf.float32)
    # scale q
    if heads != 0:
        # scaled q
        q *= (num_units // heads) ** -0.5
    else:
        q *= num_units ** -0.5

    scaled_attention_logits = tf.matmul(q, k,
                                        transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    # scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)
    if dropout != 1 and dropout != 0:
        attention_weights = tf.nn.dropout(attention_weights, dropout)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class LPA(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads, pool, stride, dropout):
        """Initialize Attention.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(num_units, num_heads))

        super(LPA, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_weights = 0
        self.MP = tf.keras.layers.MaxPooling2D(pool, stride)
        self.input_dense_layer = tf.keras.layers.Conv1D(self.num_units, 1)
        self.att_layer = SelfAttention(
            self.num_units, self.num_heads, self.dropout, linear=False)
        self.LN = LayerNorm()

    def call(self, x, bias, training, pos=0, **kwargs):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        h = tf.shape(x)[2]
        w = tf.shape(x)[3]
        c = tf.shape(x)[4]
        # org_x = x
        MP_stage = self.MP(tf.reshape(x, [-1, h, w, c]))
        att_input = tf.reshape(
            MP_stage, [batch_size, length, -1])
        if pos == 0:
            pos = hyper_fn.get_position_encoding(
                length, self.num_units)
        res = self.input_dense_layer(att_input) + pos
        att_input = self.LN(res)
        att_output = self.att_layer(att_input, bias, training)
        if training:
            att_output = tf.nn.dropout(att_output, self.dropout)
        return att_output + res, self.att_layer.get_attention_weights()


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, num_units, num_heads, dropout, linear=True):
        """Initialize Attention.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(num_units, num_heads))

        super(Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_weights = 0
        self.linear = linear

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        if self.linear:
            self.q_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                       use_bias=False,
                                                       name="q")
            self.k_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                       use_bias=False,
                                                       name="k")
            self.v_dense_layer = tf.keras.layers.Dense(self.num_units,
                                                       use_bias=False,
                                                       name="v")
            self.output_dense_layer = tf.keras.layers.Dense(
                self.num_units, use_bias=False, name="output_transform")
        else:
            self.q_dense_layer = tf.keras.layers.Conv1D(self.num_units, 1)
            self.k_dense_layer = tf.keras.layers.Conv1D(self.num_units, 1)
            self.v_dense_layer = tf.keras.layers.Conv1D(self.num_units, 1)
            self.output_dense_layer = tf.keras.layers.Conv1D(self.num_units, 1)
        self.mask_x = tf.keras.layers.Masking(0)
        self.mask_y = tf.keras.layers.Masking(0)

        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, num_units]
    Returns:
      A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
    """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.num_units // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, num_units/num_heads]
    Returns:
      A tensor with shape [batch_size, length, num_units]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(
                x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.num_units])

    def call(self, x, y, bias, training, cache=None, **kwargs):
        """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, num_units]
      y: a tensor with shape [batch_size, length_y, num_units]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, num_units]
    """
        padding_bias = tf.expand_dims(
            tf.cast(tf.not_equal(tf.reduce_sum(x, -1), 0), tf.float32), -1)
        # x = self.mask_x(x)
        # y = self.mask_x(y)
        q = x
        k = y
        v = y
        # if self.linear:
        q = self.q_dense_layer(q)
        k = self.k_dense_layer(k)
        v = self.v_dense_layer(v)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat((cache["k"], k), axis=2)
            v = tf.concat((cache["v"], v), axis=2)

            # Update cache
            cache["k"] = k
            cache["v"] = v
        depth = (self.num_units // self.num_heads)
        q *= depth**-0.5
        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        self.attention_weights = weights
        if training:
            weights = tf.nn.dropout(weights, rate=self.dropout)
        with tf.name_scope('attention_output'):
            attention_output = tf.matmul(weights, v)
        attention_output = self.combine_heads(attention_output)
        return attention_output

    def get_attention_weights(self):
        return self.attention_weights


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, training, cache=None, **kwargs):
        return super(SelfAttention, self).call(x, x, bias, training, cache,
                                               **kwargs)


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, num_units, pad_id, name="embedding"):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      num_units: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.pad_id = pad_id
        self.shared_weights = self.add_weight(
            shape=[self.vocab_size, self.num_units],
            dtype="float32",
            name="shared_weights",
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.num_units**-0.5))
        self.dense_out = tf.keras.layers.Dense(self.vocab_size)
        # self.porjection = self.add_weight(
        #     shape=[self.vocab_size, self.num_units],
        #     dtype="float32",
        #     name="project",
        #     initializer=tf.random_normal_initializer(
        #         mean=0., stddev=self.num_units**-0.5))

    def build(self, input_shape):
        super(EmbeddingSharedWeights, self).build(input_shape)
        # self.build = True

    def call(self, inputs, linear=False):
        if linear:
            return self._linear(inputs)
        else:
            return self._embedding(inputs)

    def _embedding(self, inputs):
        embeddings = tf.gather(self.shared_weights, inputs)
        mask = tf.cast(tf.not_equal(inputs, self.pad_id), embeddings.dtype)
        embeddings *= tf.expand_dims(mask, -1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.num_units**0.5
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, num_units]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.num_units])
        logits = tf.matmul(inputs, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])
        # return self.dense_out(inputs)

    def product(self, inputs):
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.vocab_size])
        logits = tf.matmul(inputs, self.shared_weights)

        return tf.reshape(logits, [batch_size, length, self.num_units])

    def att_shared_weights(self, inputs):
        # projection = tf.matmul(
        #     self.project_weights, self.shared_weights, transpose_b=True)
        batch_size = tf.shape(input=inputs)[0]
        inputs = tf.reshape(inputs, [-1, self.num_units]) * 64**-0.5
        weights = tf.matmul(inputs, self.shared_weights, transpose_b=True)
        # weights = tf.nn.softmax(weights, -1)
        att = tf.matmul(weights, self.shared_weights)
        att = tf.reshape(inputs, [batch_size, -1, self.num_units])
        return att

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            'vocab_size': self.vocab_size,
            'num_units': self.num_units,
            'pad_id': self.pad_id
        }
        # config.update(c)
        return c


class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
        mode:
            add: ln(x) + x
            norm: ln(x)
    """

    def __init__(self,
                 epsilon=1e-6,
                 gamma_initializer="ones",
                 beta_initializer="zeros",
                 name='norm'):
        super(LayerNorm, self).__init__(name=name)
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gamma_kernel = self.add_weight(shape=(input_dim),
                                            name="gamma",
                                            initializer=self.gamma_initializer)
        self.beta_kernel = self.add_weight(shape=(input_dim),
                                           name="beta",
                                           initializer=self.beta_initializer)
        self.mask = tf.keras.layers.Masking(0)
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs, training=False):
        # inputs = self.mask(inputs)
        bias = hyper_util.zero_masking(inputs)
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)
        output = self.gamma_kernel * normalized + self.beta_kernel
        return output * bias

    def get_config(self):
        # config = super(LayerNorm, self).get_config()
        c = {'epsilon': self.epsilon}
        # config.update(c)
        return c


class NormBlock(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, dropout, add_mode=True):
        super(NormBlock, self).__init__()
        self.layer = layer
        # self.num_units = num_units
        self.dropout = dropout
        self.add_mode = add_mode

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNorm()
        self.mask = tf.keras.layers.Masking(0)
        super(NormBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, 'add_mode': self.add_mode}

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]
        x = self.mask(x)
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)
        # if isinstance(y,tuple):
        #     y = y[0]
        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.dropout)
        return y + x


class Feed_Forward_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, num_units, dropout, linear=True):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
        super(Feed_Forward_Network, self).__init__()
        self.num_units = num_units
        self.dropout = dropout
        self.linear = linear

    def build(self, input_shape):
        out_dim = input_shape[-1]
        # if self.linear:
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.num_units,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            out_dim, use_bias=True, name="output_layer")
        super(Feed_Forward_Network, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "dropout": self.dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_dense_layer(output)

        return output
