# =========================================================
# Import all dependencies
# =========================================================
import tensorflow as tf
from tensorflow.keras import layers


class AstraEmbedding(layers.Layer):

  """
  Embeds time series data using sequential, segment, and positional encodings.

  This layer combines three types of encodings to represent time series data:
  - Sequential encoding: Encodes the magnitude values using a dense layer.
  - Segment encoding: Encodes band information using a non-linear layer.
                      NOTE:  It projects up to the model dimension
  - Positional encoding: Encodes the temporal positions of observations.

  Parameters:
  -----------------------------------------------------------------------------------------------------------------
      d_model (int): Dimensionality of the embedding space.
      base (float): Base/wavelength for positional encoding. Defaults to 10000.
      rate (float): Dropout rate. Defaults to 0.1.
      use_band_info (bool): Whether to use band information and add segment encoding. Defaults to True.
      use_drop (bool): Whether to apply dropout. Defaults to False.
      mjd (bool): Whether to use Modified Julian Date (MJD) for positional encoding. Defaults to True.
      time_scaling (float): Scaling factor for time values when mjd is True. Defaults to 100.

  Returns:
  -----------------------------------------------------------------------------------------------------------------
      tf.Tensor: Embedded time series data.
  """
  def __init__(self, d_model, base=10000, rate=0.1, use_band_info=True, use_drop=False, name="astra_embedding", mjd=True, time_scaling=100):
    super(AstraEmbedding, self).__init__()

    self.mjd = mjd
    self.base = base
    self.d_model = d_model
    self.use_drop = use_drop
    self.time_scaling = time_scaling
    self.use_band_info = use_band_info
    # Embed magnitude feature (linear)
    self.seq_embedding = layers.Dense(d_model, name="sequence_embedding") 
    # if band information is used for embeddings
    if self.use_band_info:
      self.seg_embedding = tf.keras.Sequential([
                                                tf.keras.layers.Dense(32, activation='relu'),
                                                tf.keras.layers.Dense(d_model) 
                                            ], name="segment_embedding")
      # -----------------------------------------------------------------------
      # Uncomment if you want to use a linear projection
      # self.seg_embedding = layers.Dense(d_model, name="segment_embedding")
      # -----------------------------------------------------------------------
    # get the positional embeddings
    self.pos_encoding = self.build_positional_encoding() 
    # add the dropout layer
    self.dropout = layers.Dropout(rate)
      
  def build_positional_encoding(self):
    
    def positional_encoding(times):
      """
      Calculates positional encoding. This is implemented as in the original Transformer paper.
      Follow the link: http://nlp.seas.harvard.edu/2018/04/03/attention.html

      Parameters:
      -----------------------------------------------------------------
          times (tf.Tensor): Time values.

      Returns:
      -----------------------------------------------------------------
          tf.Tensor: Positional encoding tensor.
      """
      with tf.name_scope("positional_encoding") as scope:

        if self.mjd:
            indices = tf.cast(self.time_scaling, dtype=times.dtype)*times
        else:
            #
            # If MJD is False then the timestep will be np.arange(0, times.shape[1]/seq_len)
            #
            indices = tf.tile(tf.expand_dims(tf.range(tf.shape(times)[1], dtype=times.dtype), 0), [tf.shape(times)[0], 1])
            indices = tf.expand_dims(indices, 2)

        angle_rates = tf.exp((tf.range(self.d_model, dtype=times.dtype)) * (-tf.math.log(tf.cast(self.base, dtype=times.dtype))/tf.cast(self.d_model, times.dtype)))
        angle_rates = angle_rates[tf.newaxis, tf.newaxis, :]
        angle_rads = indices * angle_rates
        #
        # Use SIN and COSINE function for even and odd indices
        # Apply sin to even indices in the array; 2i
        sines = tf.sin(angle_rads[:, :, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.cos(angle_rads[:, :, 1::2])
        # Interleave sines and cosines
        # Get shape of angle_rads
        pos_encoding = tf.reshape(
                                    tf.stack([sines, cosines], axis=-1),
                                    [tf.shape(times)[0], tf.shape(times)[1], self.d_model]
                                )
        # ------------------------------------------------------------------------------------------
        # Handle odd d_model dimension if necessary (by padding or adjusting range)
        if self.d_model % 2 != 0:
            # Simple approach: repeat last element or handle based on original paper
            # For now, lets assume d_model is even for simplicity
            pass 
        # ------------------------------------------------------------------------------------------
        return tf.cast(pos_encoding, dtype=times.dtype)

    return positional_encoding

  
  def call(self, x, training=False):

    """
    Embeds the time series data.

    Parameters:
    --------------------------------------------------------------------------
        mag (tf.Tensor): Magnitude values.
        time (tf.Tensor): Time values.
        band (tf.Tensor): Band information. Defaults to None. 
                          Pass the band information if use_band_info is True.

    Returns:
    --------------------------------------------------------------------------
        tf.Tensor: Embedded time series data.
                   Shape: (batch_size, seq_len, d_model)
    """
    mag = x['input']            # Shape: (batch, seq_len, 1)
    time = x['times']           # Shape: (batch, seq_len, 1)
    #
    # Get the sequence embedding | Shape: (batch_size, seq_len, d_model)
    #
    emb = self.seq_embedding(mag) 
    #
    # Get the positional embedding and add it to the sequence embedding | Shape: (batch_size, seq_len, d_model)
    # 
    emb += self.pos_encoding(time) 
    #
    # Get the segment embedding and add it to the embedding | Shape: (batch_size, seq_len, d_model)
    # 
    if self.use_band_info and x.get('band_info') is not None:
      band_info = x['band_info']                      # Shape: (batch, seq_len, 1)
      band_embeddings = self.seg_embedding(band_info)
      emb += band_embeddings
    #
    # Apply dropout
    #
    if self.use_drop:
      emb = self.dropout(emb, training=training)
  
    return emb
