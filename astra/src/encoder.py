# =========================================================
# Import all dependencies
# =========================================================
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
 

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.

    Parameters:
    -----------------------------------------------------------------------------
      q: query shape == (..., seq_len_q, depth)
      k: key shape     == (..., seq_len_k, depth)
      v: value shape   == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k).  Defaults to None.

    Returns:
    ------------------------------------------------------------------------------
      output, attention_weights
    """
    
    assert mask.shape[0] == q.shape[0]
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k) [e.g. (2, 4, 30, 30)]
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    #
    # scale matmul_qk
    #
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) 
    #
    # Ensure same dtype
    #
    mask = tf.cast(mask, dtype=scaled_attention_logits.dtype) 
    mask_shape = tf.shape(mask)
    # 
    # Broadcast mask shape if it differs (to (batch_size, 1, 1, seq_len_k))
    #
    if len(mask.shape) == 2: # Shape: (batch_size, seq_len_k)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
    
    elif len(mask.shape) == 3 and mask_shape[2] == 1: # Shape: (batch_size, seq_len_k, 1)
          mask = tf.transpose(mask, perm=[0, 2, 1])
          mask = mask[:, :, tf.newaxis, :] 
    
    elif len(mask.shape) == 4 and mask_shape[1:3] == [1, 1]: # Already correct shape
          pass
    
    else:
        # Handle potential unexpected mask shapes or raise an error
        tf.print("\nWarning: Unexpected mask shape in scaled_dot_product_attention:", mask_shape)
        # Attempt a reasonable reshape if possible, e.g., assuming last dim is seq_len_k
        mask = tf.reshape(mask, [mask_shape[0], 1, 1, mask_shape[-1]])
    # -------------------- end of broadcasting shapes --------------------------------------------
    # ---  Masking Convention: 
    #       0: Unmasked values (positions the model can attend to)
    #       1: Masked values (positions the model should ignore)
    # --- multiply the masked values with -1e9 to ignore
    #
    scaled_attention_logits += (mask * -1e9)
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k) (e.g. (2, 4, 30, 30))
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
    # Shape: (batch_size, num_heads, seq_len_q, depth) (e.g.(2, 4, 30, 128))
    output = tf.matmul(attention_weights, v)  

    return output, attention_weights



class AstraMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(AstraMultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        # Final linear layer after attention
        self.dense = layers.Dense(d_model) 

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask):
        batch_size = tf.shape(x)[0]
        q = self.wq(x)              # Shape: (batch_size, seq_len, d_model) (e.g. (2, 30, 512))
        k = self.wk(x)              # Shape: (batch_size, seq_len, d_model) (e.g. (2, 30, 512))
        v = self.wv(x)              # Shape: (batch_size, seq_len, d_model) (e.g. (2, 30, 512))
        # ------------------------------------------------------------------------------------------------------------
        q = self.split_heads(q, batch_size)  # Shape: (batch_size, num_heads, seq_len_q, depth) (e.g. (2, 4, 30, 128))
        k = self.split_heads(k, batch_size)  # Shape: (batch_size, num_heads, seq_len_q, depth) (e.g. (2, 4, 30, 128))
        v = self.split_heads(v, batch_size)  # Shape: (batch_size, num_heads, seq_len_q, depth) (e.g. (2, 4, 30, 128))
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # Shape: (batch_size, seq_len_q, num_heads, depth) (e.g. (2, 30, 4, 128))
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
        # Shape: (batch_size, seq_len_q, d_model) (e.g. (2, 30, 512))
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  
        # Shape: (batch_size, seq_len_q, d_model) (e.g. (2, 30, 512))
        output = self.dense(concat_attention)  

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    
    return tf.keras.Sequential([
                                layers.Dense(dff, activation='relu'),
                                layers.Dense(d_model)
                            ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_res=True, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        #
        # MLflow logging set : self.supports_masking = True
        self.supports_masking = True
        #
        # Components of the encoder
        #
        self.mha = AstraMultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.use_res = use_res

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)



    def call(self, x, mask, training=True):

        # Get the MHA output
        attn_output, attention_weights = self.mha(x, mask)
        # Apply dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Apply LN and RES
        out1 = self.layernorm1(x + attn_output)
        # Pass it to FFN
        ffn_output = self.ffn(out1)
        # Apply dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # Apply LN and RES
        out2 = self.layernorm2(out1 + ffn_output)
        # Get the final output and attention weights
        return out2, attention_weights


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, use_res=True, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        #
        # MLflow logging set : self.supports_masking = True
        self.supports_masking = True
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, use_res=use_res) for _ in range(num_layers)]


    
    def call(self, x, mask, training=True):
        #
        # Create a dictionary to store attention weights from each layer
        #
        attention_weights = {}

        for i in range(self.num_layers):
            x, block_attention_weights = self.enc_layers[i](x, mask, training=training)
            # Store the attention weights for the current layer
            attention_weights[f'encoder_layer_{i+1}_attention'] = block_attention_weights

        return x, attention_weights
