
import tensorflow as tf
from tensorflow.keras import layers


class ProjectionHead(layers.Layer):
    
    def __init__(self, d_model, projection_dim, name="projection_head", **kwargs):
        super(ProjectionHead, self).__init__(name=name, **kwargs)

        self.projection_layer = None
        self.projection_dim = projection_dim
        self.d_model = d_model # Store d_model if needed, or just use it below
        
        if self.projection_dim:
            self.projection_layer = tf.keras.Sequential([
                layers.Dense(self.d_model, activation='relu', name='projection_dense_1'), # Project back to d_model
                layers.Dense(self.projection_dim, name='projection_dense_2') # Final projection dim
            ], name='projection_mlp')

    def call(self, x, training=False):

        if self.projection_layer:
            projected_output = self.projection_layer(x, training=training) # Apply projection head
            return projected_output # Return projected output for loss calculation
        else:
            # If projection_dim is None or 0, just pass through the input (pooled output)
            return x