#
# Import all dependencies
#
import os
import numpy as np
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
    mean = tf.math.divide_no_nan(weighted_sum, sum_of_weights) 
    #
    # Center the data by subtracting the weighted mean
    #
    x_new = x - mean

    return x_new , mean


def generate_data_finetuning(path_to_read, path_to_store, objects_per_chunk=100, threshold=18.0):
    """
    Generates data for fine-tuning the model. 
    Consider brighter objects where weighted mean of the magnitude is less than 18.0

    Parameters:
    ------------------------------------------------------------------------------------
        path_to_read: Path to the input data file.
        path_to_store: Path to store the output data file.
        objects_per_chunk: Maximum number of light curves per chunk.
        threshold: Magnitude threshold for filtering light curves.

    Returns:
    ------------------------------------------------------------------------------------
        Creates files in the specified path.
    """
    #
    writer = None
    chunk_index = 0
    start_index = 0
    object_count = 0
    threshold = 18.0
    labels = list()
    chunks = list()
    filenames = list()
    weighted_mean = list()
    #
    # Create the finetuning directory if it doesn't exist
    #
    os.makedirs(os.path.join(path_to_store, "finetuning"), exist_ok=True)
    #
    for p in os.listdir(path_to_read):
        for lbl in os.listdir(path_to_read+p):
            for cnk in os.listdir(path_to_read+p+"/"+lbl):
                filenames.append(path_to_read+p+"/"+lbl+'/'+cnk)

    
    
    dataset = tf.data.TFRecordDataset(filenames)
    #
    #
    #
    for rec in dataset:
        #
        # Create a new writer for the current chunk
        #
        if object_count == 0:
            writer = tf.io.TFRecordWriter(path_to_store + f"finetuning/chunk_{chunk_index}.record")

        example = tf.train.SequenceExample()
        example.ParseFromString(rec.numpy())
        #
        # Convert to TensorFlow tensors
        # The columns of each lightcurve in ZTF is in the order: "mjd", "mag", "magerr", "band_sorted"
        #
        last_index = example.context.feature['last_index'].int64_list.value
        mag = tf.convert_to_tensor(example.feature_lists.feature_list['dim_1'].feature[0].float_list.value, dtype=tf.float64)
        magerr = tf.convert_to_tensor(example.feature_lists.feature_list['dim_2'].feature[0].float_list.value, dtype=tf.float64)
        #
        # Get the weighted mean of the light curve for each band
        #
        for i in range(len(last_index)):
            end_index = last_index[i] + 1  
            # Filter mag and magerr for the current segment
            mag_band = mag[start_index:end_index]
            magerr_band = magerr[start_index:end_index]
            # Apply standardization
            _, w_mean = standardize(mag_band, magerr_band)
            weighted_mean.append(w_mean)
            # Update start_index for the next segment
            start_index = end_index
        #
        # Check if the weighted mean is brighter than the threshold (18.0)
        #
        if any(w_mean < threshold for w_mean in weighted_mean):
            # 
            # Write the example to the new TFRecord file
            # Increment count if object is stored
            #
            writer.write(example.SerializeToString())
            object_count += 1  
        #
        if object_count == objects_per_chunk:
            #
            # Close the current writer and reset the count and increase the chunk index
            #
            writer.close()
            object_count = 0
            chunk_index += 1
    #
    # Close the last writer if it's still open
    #
    if writer is not None:
        writer.close()

    


