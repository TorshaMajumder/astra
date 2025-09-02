#
# Import all dependencies
#
import os
import json
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from astra.bands.bands import ztf_band



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
    filenames = list()
    weighted_mean = list()

    glob_pattern = os.path.join(source, 'partition_*', '*', 'chunk_*.record')
    print(f"Searching for inference files using pattern: {glob_pattern}")
    filenames_dataset = tf.data.Dataset.list_files(glob_pattern, shuffle=shuffle, seed=seed)

    num_files_found = tf.data.experimental.cardinality(filenames_dataset)
    if num_files_found == 0:
        raise ValueError(f"No TFRecord files found for inference matching pattern: {glob_pattern}")
    elif num_files_found != tf.data.UNKNOWN_CARDINALITY:
         print(f"Found {num_files_found} inference files.")

    dataset = filenames_dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=AUTO,
        num_parallel_calls=AUTO
    )
    
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
        label = example.context.feature['label'].bytes_list.value[0].decode('utf-8')
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





def load_hparams_from_event_file(run_directory):
    """
    Loads hyperparameters from a text summary in a TensorBoard event file.

    Args:
        run_directory (str): The path to the specific run directory
                             (e.g., '/path/to/run_YYYYMMDD_HHMMSS/').

    Returns:
        tuple: A tuple containing (model_params, training_params, data_params)
               dictionaries, or (None, None, None) if the data is not found.
    """
    print(f"Searching for hyperparameters in event file in: {run_directory}")
    
    try:
        # Initialize EventAccumulator to load text summaries (Tensors)
        ea = event_accumulator.EventAccumulator(
            run_directory,
            size_guidance={
                # Text summaries are often stored as Tensors
                event_accumulator.TENSORS: 10,
            }
        )
        ea.Reload()

        # The tag for hyperparameters was 'hyperparameters'
        hparam_tag = 'hyperparameters'
        
        # Check if the tag exists in the 'tensors' category, as text is stored there
        if hparam_tag not in ea.Tags()['tensors']:
            print(f"ERROR: Hyperparameter tag '{hparam_tag}' not found in the 'tensors' category of the event file.")
            print("Available tensor tags:", ea.Tags()['tensors'])
            return None, None, None

        # Retrieve the tensor event
        hparam_event = ea.Tensors(hparam_tag)[0] # Get the first (and likely only) event
        
        # The text is stored in the tensor_proto as a byte string
        # Convert the tensor proto to a numpy array, which will be an array of bytes
        hparam_bytes = tf.make_ndarray(hparam_event.tensor_proto).item()
        
        # Decode the byte string to a regular string
        hparam_string = hparam_bytes.decode('utf-8')
        
        # The string was saved with <pre> tags, remove them
        if hparam_string.startswith('<pre>'):
            hparam_string = hparam_string.replace('<pre>', '').replace('</pre>', '')
            
        # Parse the JSON string to get the dictionary
        log_data = json.loads(hparam_string)
        
        # Extract the specific parameter dictionaries
        hparams = log_data.get('hyperparameters', log_data) # Handle both nested/non-nested cases
        model_params = hparams.get('model_params', {})
        training_params = hparams.get('training_params', {})
        data_params = hparams.get('data_params', {})
        
        if not model_params or not data_params:
            print("ERROR: Parsed hyperparameters are missing 'model_params' or 'data_params' section.")
            return None, None, None

        print("Hyperparameters loaded successfully from event file.")
        return model_params, training_params, data_params

    except FileNotFoundError:
        print(f"ERROR: Log directory not found: {run_directory}")
        return None, None, None
    except IndexError:
         print(f"ERROR: Hyperparameter tag '{hparam_tag}' was found, but no event data was associated with it.")
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading hparams: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    


