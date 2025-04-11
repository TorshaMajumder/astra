#
# Import all dependencies
#
import tensorflow as tf

@tf.function
def standardize(x, err):
    """
    Standardizes the input tensor 'x' using a weighted average based on the 'err' tensor.

    Parameters:
    ------------------------------------------------------------------------------------
        x: A TensorFlow tensor representing the magnitude of the light curves.
        err: A TensorFlow tensor representing the corresponding mag_err (uncertainties).

    Returns:
    ------------------------------------------------------------------------------------
        A TensorFlow tensor 'x_new' containing the standardized data.
    """
    #
    # Check for NaNs in the mag and magerr values and replace it with zeros and ones.
    #
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    err = tf.where(tf.math.is_nan(err), tf.ones_like(err), err)
    #
    # Calculate the weighted mean
    #
    weights = 1.0 / tf.square(err)
    weighted_sum = tf.reduce_sum(x * weights)
    sum_of_weights = tf.reduce_sum(weights)
    mean = weighted_sum / sum_of_weights
    #
    # Center the data by subtracting the weighted mean
    #
    x_new = x - mean

    return x_new , mean