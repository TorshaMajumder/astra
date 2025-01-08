import tensorflow as tf
import numpy as np
from core.attention import MultiHeadAttention
from core.positional import positional_encoding
from core.masking import reshape_mask

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_leak=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.use_leak = use_leak
        if use_leak:
            self.reshape_leak_1 = tf.keras.layers.Dense(d_model)
            self.reshape_leak_2 = tf.keras.layers.Dense(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        if self.use_leak:
            out1 = self.layernorm1(self.reshape_leak_1(x) + attn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out1 = self.layernorm1(attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        if self.use_leak:
            out2 = self.layernorm2(self.reshape_leak_2(out1) + ffn_output) # (batch_size, input_seq_len, d_model)
        else:
            out2 = self.layernorm2(ffn_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, meta_shape, emd_dim, maxlen, num_heads, dff,
                 base=10000, rate=0.1, use_leak=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # self.d_model = d_model
        self.d_model = emd_dim+meta_shape
        self.meta_shape = meta_shape
        self.emd_dim = emd_dim
        self.maxlen = maxlen
        self.num_layers = num_layers
        self.base = base
        self.inp_transform = tf.keras.layers.Dense(emd_dim)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, use_leak)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, data, training=False):
        # adding embedding and position encoding.
        # x_pe = positional_encoding(data['times'], self.emd_dim, mjd=True)
        x_transformed = self.inp_transform(data["input"])
        reshaped_tensor1 = tf.tile(tf.expand_dims(data['meta'], 1), [1, self.maxlen, 1])
        # print(reshaped_tensor1.shape)
        # print(".........>>>>>>>", tf.shape(data['meta'][-1]))
        # print(data)
        # x_meta = tf.reshape(data['meta'], (1, 1, tf.shape(data['meta'])[-1]))
        # embedding_layer = tf.keras.layers.Dense(self.d_model)
        # embedding = embedding_layer(x_meta)  # Shape: (1, 1, 256)
        # # 2. Repeat to create a sequence of 200
        # repeated_embedding = tf.repeat(embedding, repeats=300, axis=1)  # Shape: (1, 200, 256)
        # x_pe = self.pe_emb(data['times'])
       
        # # x_meta = tf.reshape(data['meta'], (1, 1, tf.shape(data['meta'])[-1]))
        # x_meta = tf.tile(x_meta, [1, tf.shape(data['input'])[1], 1])
        # concatenated_tensor = tf.concat([data['input'], x_meta], axis=2)
        # x_input = tf.keras.layers.LayerNormalization()(concatenated_tensor)
        
        
        # inp_m = tf.reshape(data['meta'], (1, 1, tf.shape(data['meta'])[-1]))
        # inp_m = tf.reshape(data['meta'], [1, 54, 1]) 
        # print(tf.shape(data['meta']))
        # x_meta = self.inp_transform(inp_m)

        # transformed_input = x_transformed + x_pe 

        # Concatenate tensors
        # concatenated_tensor = tf.concat([transformed_input, reshaped_tensor1], axis=2)  # Shape: (1, 200, 310)
        concatenated_tensor = tf.concat([x_transformed, reshaped_tensor1], axis=2)  # Shape: (1, 200, 310)

        # print(concatenated_tensor.shape)  # Output: (1, 200, 310)   
        # transformed_input = tf.keras.layers.concatenate([transformed_input, x_meta], axis=1)
        # print(transformed_input.shape)
        
        # x = self.dropout(transformed_input, training=training)
        x = self.dropout(concatenated_tensor, training=training)
        # print(tf.shape(x))
        # exit
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, data['mask_in'])

        return x  # (batch_size, input_seq_len, d_model)
