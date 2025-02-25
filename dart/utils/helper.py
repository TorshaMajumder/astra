#
# Import all dependencies
#
import tensorflow as tf

# @tf.function
def standardize(x, err):
    """
    Standardizes the input tensor 'x' using a weighted average based on the 'err' tensor.

    Args:
        x: A TensorFlow tensor representing the data.
        err: A TensorFlow tensor representing the corresponding errors (uncertainties).

    Returns:
        A TensorFlow tensor 'x_' containing the standardized data.
    """

    # Replace NaN values in 'x' with zeros
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    # Replace NaN values in 'err' with ones (equivalent to no weight)
    err = tf.where(tf.math.is_nan(err), tf.ones_like(err), err)


    # Ensure err is not zero to avoid division by zero
    err = tf.where(tf.equal(err, 0), tf.ones_like(err), err) # replace every zero in err with 1 so that no nan are produced.

    # Calculate the weights (inverse of squared errors)
    weights = 1.0 / tf.square(err)

    # Calculate the weighted mean
    weighted_sum = tf.reduce_sum(x * weights)
    sum_of_weights = tf.reduce_sum(weights)

    mean = weighted_sum / sum_of_weights

    # Center the data by subtracting the weighted mean
    x_ = x - mean

    return x_ , mean