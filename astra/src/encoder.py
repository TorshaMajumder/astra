

import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
 


# 2. Scaled Dot-Product Attention (The core of the Transformer)
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape     == (..., seq_len_k, depth)
      v: value shape   == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k).  Defaults to None.

    Returns:
      output, attention_weights
    """
    assert mask.shape[0] == q.shape[0]
    matmul_qk = tf.matmul(q, k, transpose_b=True)  #(batch_size, num_heads, seq_len_q, seq_len_k)(2, 4, 30, 30)

    # print("7", matmul_qk.shape)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) #(batch_size, num_heads, seq_len_q, seq_len_k)(2, 4, 30, 30)

    # print("8", scaled_attention_logits.shape)

    # mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    # print("8", mask.shape)
    # add the mask to the scaled tensor.
    # if mask is not None:
    #   mask = tf.tile(mask[:, tf.newaxis, :, :], [1, tf.shape(scaled_attention_logits)[1], 1, 1])  #(batch_size, num_heads, seq_len, input_dim)(2, 4, 30, 1)
    #   # print("9", mask.shape)
    #   # inverted_mask = 1.0 - mask  # Invert the mask
    #   scaled_attention_logits += (mask * -1e9)  # Mask out padded tokens
    # if mask is not None:
    # The mask needs to broadcast to (batch_size, num_heads, seq_len_q, seq_len_k)
    # Assuming mask input shape is (batch_size, seq_len_k) or (batch_size, 1, seq_len_k)
    # Reshape mask to (batch_size, 1, 1, seq_len_k) for broadcasting

    mask = tf.cast(mask, dtype=scaled_attention_logits.dtype) # Ensure same dtype
    mask_shape = tf.shape(mask)
    if len(mask.shape) == 2: # (batch_size, seq_len_k)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
    elif len(mask.shape) == 3 and mask_shape[2] == 1: # (batch_size, seq_len_k, 1)
          mask = tf.transpose(mask, perm=[0, 2, 1])
          mask = mask[:, :, tf.newaxis, :] # Should become (batch_size, 1, 1, seq_len_k)
    elif len(mask.shape) == 4 and mask_shape[1:3] == [1, 1]: # Already correct shape
          pass
    else:
          # Handle potential unexpected mask shapes or raise an error
        tf.print("Warning: Unexpected mask shape in scaled_dot_product_attention:", mask_shape)
          # Attempt a reasonable reshape if possible, e.g., assuming last dim is seq_len_k
        mask = tf.reshape(mask, [mask_shape[0], 1, 1, mask_shape[-1]])

    scaled_attention_logits += (mask * -1e9)


    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  #(batch_size, num_heads, seq_len_q, seq_len_k)(2, 4, 30, 30)
    # print("10", attention_weights.shape)
    # print("11", attention_weights)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights




# 3. MultiHeadAttention Layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model) # Final linear layer after attention

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask):
        batch_size = tf.shape(x)[0]

        # 1 (2, 30, 512) (2, 30, 512) (2, 30, 512)
        # 2 (2, 4, 30, 128) (2, 4, 30, 128) (2, 4, 30, 128)
        # 7 (2, 4, 30, 30)
        # 8 (2, 4, 30, 30)
        # 9 (2, 4, 30, 1)
        # 10 (2, 4, 30, 30)
        # 11 Tensor("Softmax:0", shape=(2, 4, 30, 30), dtype=float32)
        # 3 (2, 4, 30, 128) (2, 4, 30, 30)
        # 4 (2, 30, 4, 128) (2, 4, 30, 30)
        # 5 (2, 30, 512)
        # 6 (2, 30, 512)

        q = self.wq(x)  # (batch_size, seq_len, d_model) (2, 30, 512)
        k = self.wk(x)  # (batch_size, seq_len, d_model) (2, 30, 512)
        v = self.wv(x)  # (batch_size, seq_len, d_model) (2, 30, 512)

        # print("1",q.shape, k.shape, v.shape)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) (2, 4, 30, 128)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth) (2, 4, 30, 128)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth) (2, 4, 30, 128)

        # print("2",q.shape, k.shape, v.shape)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth) (2, 4, 30, 128)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k) (2, 4, 30, 30)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # print("3", scaled_attention.shape, attention_weights.shape)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth) (2, 30, 4, 128)

        # print("4", scaled_attention.shape, attention_weights.shape)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model) (2, 30, 512)

        # print("5", concat_attention.shape)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model) (2, 30, 512)
        # print("6", output.shape)

        return output, attention_weights




def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_res=True, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.use_res = use_res

        # if use_res:
        #     self.reshape_res_1 = layers.Dense(d_model)
        #     self.reshape_res_2 = layers.Dense(d_model)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)



    def call(self, x, mask, training=True):

        attn_output, _ = self.mha(x, mask)

        attn_output = self.dropout1(attn_output, training=training)

        # if self.use_res:
        out1 = self.layernorm1(x + attn_output)
        # else:
        #     out1 = self.layernorm1(attn_output)


        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        # if self.use_res:
        out2 = self.layernorm2(out1 + ffn_output)
        # else:
        #     out2 = self.layernorm2(ffn_output)

        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, use_res=True, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)


        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = TimeSeriesEmbedding(d_model, base)
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, use_res=use_res) for _ in range(num_layers)]

        # self.dropout = layers.Dropout(rate)

    def call(self, x, mask, training=True):


        # x = self.embedding(x)

        # x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)

        return x
