#
# Import all dependencies
#
import tensorflow as tf
from tensorflow.keras import layers


class TimeSeriesEmbedding(layers.Layer):

  """
  Embeds time series data using sequential, segment, and positional encodings.

  This layer combines three types of encodings to represent time series data:
  - Sequential encoding: Encodes the magnitude values using a dense layer.
  - Segment encoding: Encodes band information using a dense layer.
  - Positional encoding: Encodes the temporal positions of observations.

  Parameters:
  -----------------------------------------------------------------------------------------------------------------
      d_model (int): Dimensionality of the embedding space.
      base (float): Base/wavelength for positional encoding. Defaults to 10000.
      rate (float): Dropout rate. Defaults to 0.1.
      use_band_info (bool): Whether to use band information and add segment encoding. Defaults to True.
      use_drop (bool): Whether to apply dropout. Defaults to False.
      mjd (bool): Whether to use Modified Julian Date (MJD) for positional encoding. Defaults to True.

  Returns:
  -----------------------------------------------------------------------------------------------------------------
      tf.Tensor: Embedded time series data.
  """
  def __init__(self, d_model, base=10000, rate=0.1, use_band_info=True, use_drop=False, mjd=True):
      super(TimeSeriesEmbedding, self).__init__()

      self.mjd = mjd
      self.base = base
      self.d_model = d_model
      self.use_drop = use_drop
      self.seq_embedding = None
      self.seg_embedding = None
      self.dropout = layers.Dropout(rate)
      self.use_band_info = use_band_info


  def build(self, input_shape):
    """
    Builds the embedding layer.

    Initializes the sequential and segment encoding layers.

    Parameters:
    ---------------------------------------------------------
        input_shape (tuple): Shape of the input tensor.
    """

    self.seq_embedding = self.sequential_encoding(self.d_model)
    self.seg_embedding = self.segment_encoding(self.d_model)
    super(TimeSeriesEmbedding, self).build(input_shape)


  def sequential_encoding(self, d_model):
    """
    Creates a sequential encoding layer.

    Parameters:
    -----------------------------------------------------------
        d_model (int): Dimensionality of the encoding.

    Returns:
    -----------------------------------------------------------
        tf.keras.layers.Dense: A dense layer for sequential encoding.
        
    """
    with tf.name_scope("SequentialEncoding") as scope:
      return layers.Dense(d_model, activation=None)

  def segment_encoding(self, d_model):
    """
    Creates a segment encoding layer.

    Parameters:
    -----------------------------------------------------------------
        d_model (int): Dimensionality of the encoding.

    Returns:
    -----------------------------------------------------------------
        tf.keras.layers.Dense: A dense layer for segment encoding.
    """
    with tf.name_scope("SegmentEncoding") as scope:
      return layers.Dense(d_model, activation=None)

  def positional_encoding(self, times):
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

    with tf.name_scope("PositionalEncoding") as scope:
      if self.mjd:
        indices = times
      else:
        #
        # If MJD is False then the timestep will be np.arange(0, times.shape[1]/seq_len)
        #
        indices = tf.tile(tf.expand_dims(tf.range(tf.shape(times)[1], dtype=times.dtype), 0), [tf.shape(times)[0], 1])
        indices = tf.expand_dims(indices, 2)

      angle_rates = tf.exp((2*(tf.range(self.d_model, dtype=times.dtype)//2)) * (-tf.math.log(tf.cast(self.base, dtype=times.dtype))/tf.cast(self.d_model, times.dtype)))
      angle_rads = indices * angle_rates[tf.newaxis, tf.newaxis, :]
      #
      # Use SIN and COSINE function for even and odd indices
      #
      angle_rads = tf.where(tf.math.floormod(tf.range(self.d_model), 2) == 0,
                            tf.sin(angle_rads[:, :, :]),
                            tf.cos(angle_rads[:, :, :]))


      return tf.cast(angle_rads, dtype=times.dtype)





  def call(self, mag, time, band=None):

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
    #
    # Get the sequence embedding, Shape: (batch_size, seq_len, d_model)
    #
    x = self.seq_embedding(mag) 
    #
    # Get the positional embedding and add it to the sequence embedding, 
    # Shape: (batch_size, seq_len, d_model)
    #
    x += self.positional_encoding(time) 
    #
    # Get the segment embedding and add it to the embedding, 
    # Shape: (batch_size, seq_len, d_model)
    #
    if self.use_band_info and band is not None:
      band_info = self.seg_embedding(band) 
      x += band_info 
    #
    # Normalize the overall embeddings
    #
    x = tf.math.divide_no_nan(x, tf.math.sqrt(tf.cast(self.d_model, x.dtype)))
    #
    # Apply dropout
    #
    if self.use_drop:
      x = self.dropout(x)
  

    return x
