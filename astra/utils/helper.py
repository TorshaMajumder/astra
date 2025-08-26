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
def get_sequence(current_serie, mask_serie, max_len, num_cols):
  #
  #
  #
  assert current_serie.shape[0] == mask_serie.shape[0]
  #
  pivot = 0
  serie_len = tf.shape(current_serie)[0]
  #
  # Check if the sequence is larger than "max_len"
  # If Yes then pick a window randomly
  # Else pad sero to the end to make its length as "max_len"
  #
  if serie_len > max_len:
    max_val = tf.maximum(serie_len - max_len, 0)
    #
    pivot = tf.random.uniform([],
                                 minval=tf.cast(0, tf.int32),
                                 maxval=tf.cast(max_val, tf.int32),
                                 dtype=tf.int32)
    #
    current_serie = tf.slice(current_serie, [pivot, 0], [max_len, -1])
    mask_serie = tf.slice(mask_serie, [pivot], [max_len])

  else:
    padding_rows = max_len - serie_len

    if padding_rows > 0:
      #
      zero_padding = tf.zeros([padding_rows, num_cols], dtype=current_serie.dtype)
      current_serie = tf.concat([current_serie, zero_padding], axis=0)
      mask_serie = tf.concat([mask_serie, tf.ones([padding_rows], dtype=mask_serie.dtype)], axis=0)

  return current_serie, mask_serie


@tf.function
def sequence_window(sequence, mask, last_index, bands_tensor, max_len):
    """
    Extracts random windows of lightcurves from the sequence if the sequence
    length is larger than max_len, and padding sequence shorter than max_len with zeros.

    Parameters:
    --------------------------------------------------------------------------------------------
      sequence: A tensor of shape [num_steps, num_features])
      mask: mask tensor
      last_index: a tensor of the indices of the last index of each band in the sequence.
      bands_tensor: the bands in the sequence.
      max_len: The maximum length of each band sequence after sliding window.

    Returns:
    --------------------------------------------------------------------------------------------
      result_series: An updated sequence with max_len.
      result_mask: An updated mask tensor with corresponding masks.
    """
    #
    #
    #
    series = []
    mask_series = []
    idx = tf.cast(0, dtype=tf.int64)
    num_cols = tf.shape(sequence)[1]
    #
    # Find the available bands in the sequence, otherwise return -1
    # If (g,i) - filters are available in the sequence, it will return {'g':0, 'r':-1, 'i':2}
    # Remember that the sequence is ordered wrt the filters/keys in the ztf_band dict.
    #
    band_indices = {band: tf.cond(
        tf.reduce_any(tf.equal(bands_tensor, band)),
        lambda: tf.cast(tf.where(tf.equal(bands_tensor, band))[0][0], tf.int64),
        lambda: tf.constant(-1, dtype=tf.int64)) for band in ztf_band.keys()}
    #
    for fil in ztf_band.keys():
      index_in_bands = band_indices[fil]
      is_in_bands = index_in_bands != -1
      #
      # If the band is available in the sequence then use sliding window
      # Else pad zero of "max_len" for that band
      #
      if is_in_bands:
          #
          # Extract the current band and adjust it to size "max_len"
          #
          current_serie = sequence[idx:last_index[index_in_bands] + 1]
          mask_serie = mask[idx:last_index[index_in_bands] + 1]
          current_serie, mask_serie = get_sequence(current_serie, mask_serie, max_len[fil], num_cols)
          #
          # Move to the next band in the sequence
          #
          idx = last_index[index_in_bands] + 1
      else:
          current_serie = tf.zeros((max_len[fil], num_cols), dtype=sequence.dtype)
          mask_serie = tf.ones(max_len[fil], dtype=mask.dtype)
      #
      series.append(current_serie)
      mask_series.append(mask_serie)
    #
    result_series, result_mask = tf.concat(series, axis=0), tf.concat(mask_series, axis=0)
    #
    return result_series, result_mask


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
    #
    # Create the finetuning directory if it doesn't exist
    #
    os.makedirs(os.path.join(path_to_store, "finetuning"), exist_ok=True)
    #
    #
    for root, _, files in os.walk(path_to_read):
      for file_ in files:
        filenames.append(os.path.join(root, file_))

    
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

    


