import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense, GlobalAveragePooling1D


class ProjectionHead(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(128, name='Dense1')  # Replace 64 with your desired number of units
        self.bn1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(64, name='Dense2')  # Replace 32 with your desired number of units
        self.bn2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense3 = Dense(256, name='Dense3')
        # self.bn3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()




    
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)  # Apply ReLU activation after the first Dense layer
        x = self.dense2(inputs)
        x = self.bn2(x)
        x = tf.nn.relu(x)  # Apply ReLU activation after the second Dense layer
        x = self.dense3(x)
        # x = self.bn3(x)
        x = self.global_avg_pool(x)
        return x


class RegLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reg_layer = Dense(1, name='RegLayer')
        self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.bn_0(inputs)
        x = self.reg_layer(x)
        return x

class SauceLayer(tf.keras.layers.Layer):
    def init(self, shape,**kwargs):
        super(SauceLayer, self).init(**kwargs)
        self.supports_masking = True
        self.shape = shape

    def build(self, input_shape):
        self.scale = tf.Variable([1/self.shape for _ in range(self.shape)], trainable=True)

    def call(self, inputs):
        # Softmax normalized
        scale = tf.nn.softmax(self.scale)
        return tf.tensordot(scale, inputs, axes=1)
