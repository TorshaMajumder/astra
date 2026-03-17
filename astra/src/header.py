# =========================================================
# Import all dependencies
# =========================================================
import tensorflow as tf
from tensorflow.keras import layers




class DistilProjectionHead(layers.Layer):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, name="astra_ph_1", **kwargs):
        super(DistilProjectionHead, self).__init__(**kwargs)
        
        self.norm_last_layer = norm_last_layer
        
         
        self.mlp = tf.keras.Sequential(name="astra_ph_1_mlp")
        for i in range(nlayers - 1):
            self.mlp.add(layers.Dense(hidden_dim, use_bias=False, name=f"astra_ph_1_dense_{i}"))
            self.mlp.add(layers.Activation('gelu', name=f"astra_ph_1_gelu_{i}"))
            
        # Bottleneck layer
        self.mlp.add(layers.Dense(bottleneck_dim, use_bias=False, name=f"astra_ph_1_bottleneck"))
        
        # 2. L2 Normalization Layer
        self.last_layer_weight = self.add_weight(
                                                    shape=(bottleneck_dim, out_dim),
                                                    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                                    trainable=True,
                                                    name="astra_ph_1_last_layer_weight"
                                                )

    def call(self, x, training=False):
        # Pass through MLP
        x = self.mlp(x, training=training)
        
        # L2 Normalize features
        x = tf.math.l2_normalize(x, axis=1)
        
        # L2 Normalize the final weights (Weight Normalization)
        w = tf.math.l2_normalize(self.last_layer_weight, axis=0)
        # x = self.last_layer(x)
        # Dot product
        logits = tf.matmul(x, w)
        
        return logits




class ProjectionHead(layers.Layer):
    
    def __init__(self, d_model, projection_dim, name="projection_head", **kwargs):
        super(ProjectionHead, self).__init__(name=name, **kwargs)

        self.projection_layer = None
        self.projection_dim = projection_dim
        self.d_model = d_model 
        # GlobalAvgPooling is applied before this projection head, 
        # so, project it back to d_model first, and then final projection dim
        if self.projection_dim: 
            self.projection_layer = tf.keras.Sequential([
                                                            layers.Dense(self.d_model, activation='relu', name='projection_dense_1'), 
                                                            layers.Dense(self.projection_dim, name='projection_dense_2') 
                                                        ], name='projection_mlp')

    def call(self, x, training=False):

        if self.projection_layer:
            projected_output = self.projection_layer(x, training=training) 
            return projected_output 
        else:
            # If projection_dim is None or 0, just pass through the input (pooled output)
            return x