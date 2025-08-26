import os
import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from astra.utils.helper import standardize, sequence_window
from astra.bands.bands import ztf_band, ztf_mag


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
AUTO = tf.data.AUTOTUNE
os.system('clear')
tf.random.set_seed(1024)
np.random.seed(1024)

@tf.function
def get_window(current_serie, mask_serie, max_len, num_cols):
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
def sliding_window(sequence, mask, last_index, bands_tensor, max_len):
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
          current_serie, mask_serie = get_window(current_serie, mask_serie, max_len, num_cols)
          #
          # Move to the next band in the sequence
          #
          idx = last_index[index_in_bands] + 1
      else:
          current_serie = tf.zeros((max_len, num_cols), dtype=sequence.dtype)
          mask_serie = tf.ones(max_len, dtype=mask.dtype)
      #
      series.append(current_serie)
      mask_series.append(mask_serie)
    #
    result_series, result_mask = tf.concat(series, axis=0), tf.concat(mask_series, axis=0)
    #
    return result_series, result_mask



@tf.function
def binning(sequence, bin_width, drop_data):
  '''
  Bins the input sequence and randomly drops a fraction of the bins.
  It's implemented as in Section:4.1 of the paper: RAINBOW (arXiv - https://arxiv.org/pdf/2310.02916)

  Parameters:
  ------------------------------------------------------------------------------
    sequence: A TensorFlow tensor representing the input sequence.
    bin_width: The width of the bins.
    drop_data: The fraction of bins to drop.

  Returns:
  ------------------------------------------------------------------------------
    A tuple containing:
      - The modified sequence with dropped bins.
      - A mask indicating the dropped bins (1 for dropped/masked, 0 for kept/unmasked).
        The masking convention is matched with the original Transformer paper.

  '''
  #
  # If binning is True then 0.0<drop_data<=1
  #
  assert drop_data > 0.0 and drop_data <= 1.0 
  #
  # The light curve time span was divided into "bin_width" day long bins
  #
  time = sequence[:,0]
  #
  min_time = tf.reduce_min(time)
  max_time = tf.reduce_max(time)
  #
  bins = tf.range(min_time, max_time + bin_width, bin_width, dtype=time.dtype)
  #
  bin_index = tf.searchsorted(bins, time, side="left") # use left for lower-bound
  #
  # Find unique bin-index to drop
  #
  uniq_bin_index, time_index = tf.unique(bin_index)
  #
  num_bins_to_drop = tf.cast(tf.cast(tf.shape(uniq_bin_index)[0], tf.float32) * drop_data, tf.int32)
  #
  # Handle case where num_bins_to_drop is 0
  #
  num_bins_to_drop = tf.maximum(num_bins_to_drop, 1)
  #
  bin_index_drop = tf.random.shuffle(uniq_bin_index)[:num_bins_to_drop]
  #
  time_index_drop = tf.math.reduce_any(tf.equal(time_index[:, tf.newaxis], bin_index_drop), axis=1)
  #
  # Modify the sequence and the mask tensors with binned time-index to zero
  #
  mask_serie = tf.where(time_index_drop, tf.ones_like(time_index_drop, dtype=sequence.dtype),
                        tf.zeros_like(time_index_drop, dtype=sequence.dtype))
  new_serie = tf.where(tf.expand_dims(time_index_drop, axis=-1), tf.zeros_like(sequence), sequence)
  #
  return new_serie, mask_serie


@tf.function
def photometric_outlier(sequence, mask, mag_limit, mag_saturation):
  """
  Introduces photometric outliers to a sequence/magnitude.

  Parameters:
  ------------------------------------------------------------------------------
      sequence: The input TensorFlow tensor.
      mask: mask indicating the indices to be modified (where mask=1).
      mag_limit: The upper limit for magnitude.
      mag_saturation: The lower limit for magnitude.

  Returns:
  ------------------------------------------------------------------------------
      The modified sequence with added outliers.
  """
  #
  # Get the indices where the mask is 0
  #
  valid_indices = tf.where(tf.equal(mask, 0.0)) 
  random_index = tf.constant(-1, dtype=tf.int32)
  num_valid_indices = tf.shape(valid_indices)[0]
  #
  # Add outliers only when all(mask)!=0
  #
  if num_valid_indices != 0:

    random_index = tf.random.shuffle(valid_indices)[0]
    random_index = tf.squeeze(random_index)
    #
    # Filter the sequence/mag
    #
    mag = tf.gather(sequence, random_index)
    #
    # Lower the mag value below the detection limit
    #
    mag =  mag_saturation - tf.random.uniform([], 0, 0.5, dtype=sequence.dtype)
    mag = tf.cast(mag, sequence.dtype)
    indices = tf.reshape(random_index, [1])
    updates = tf.reshape(mag, [1])
    #
    # Update the sequence
    #
    sequence = tf.tensor_scatter_nd_update(sequence, tf.expand_dims(indices, axis=-1), updates)

  return sequence


@tf.function
def gaussian_noise(sequence, noise_level=0.1):
    """
    Adds white Gaussian noise to a sequence/magnitude for augmentation.

    Parameters:
    ----------------------------------------------------------------------------
        sequence: A TensorFlow tensor representing the sequence/magnitude.
        noise_level: The standard deviation of the Gaussian noise.

    Returns:
    ----------------------------------------------------------------------------
        A TensorFlow tensor representing the sequence with added noise.
    """
    #
    # Get the length of the sequence and reshape it for correct broadcasting
    #
    length = tf.shape(sequence)[0]
    sequence = tf.reshape(sequence, (length, 1))
    #
    # Generate Gaussian noise with mean 0 and specified standard deviation
    #
    noise = tf.random.normal(shape=(length, 1), mean=0.0, stddev=noise_level, dtype=sequence.dtype)
    #
    # Add the noise to the sequence
    #
    noisy_sequence = sequence + noise

    return noisy_sequence

@tf.function
def deserialize(sample):
    '''
    Deserialize the tf.records into an input dict format.
    The columns of each lightcurve in ZTF is in the order: "mjd", "mag", "magerr", "band_sorted"

    NOTE: "num_keys" param should be the total columns in each lightcurve.

    Parameters:
    ---------------------------------------------------------------------------------------------
    sample: tf.records sample

    Returns:
    ---------------------------------------------------------------------------------------------
    input_dict
    '''
    num_keys = 4
    input_dict = dict()
    sequence_features = dict()
    casted_inp_parameters = []

    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.string),
                        'bands': tf.io.VarLenFeature(dtype=tf.string),
                        'last_index': tf.io.VarLenFeature(dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.int64)}

    for i in range(num_keys):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float64)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict['id']   = tf.cast(context['id'], tf.int64)
    input_dict['last_index'] = tf.sparse.to_dense(context['last_index'])
    input_dict['label']  = tf.cast(context['label'], tf.string)
    input_dict['bands']  = tf.sparse.to_dense(context['bands'])


    for i in range(num_keys):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float64)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input_id'] = sequence

    return input_dict



@tf.function
def augmentation(data,
                 noise_level=0.1,
                 apply_white_noise=False,
                 apply_binning=False,
                 apply_outlier=False,
                 maxlen=400,
                 bin_width=5,
                 drop_data=0.50,
                ):
  '''
  Augments the input data with various photometric transformations.
  Convention: The original Transformer paper and most implementations follow this convention:

  0: Unmasked values (positions the model can attend to)
  1: Masked values (positions the model should ignore)


  Parameters:
  -------------------------------------------------------------------------------------------------------
    data: Input data in tf.records format.
    apply_white_noise (bool): Whether to apply Gaussian noise to the magnitude.
    apply_binning (bool): Whether to apply binning and random dropping of bins.
    apply_outlier (bool): Whether to introduce photometric outliers.
    maxlen (int): The maximum length of each band sequence after sliding window.
    bin_width (int): The width of the bins for binning.
    drop_data (float): The fraction of bins to drop during binning. Provide value 0.0 < drop_data <=1 .

  Returns:
  -------------------------------------------------------------------------------------------------------
    The augmented input data as a TensorFlow tensor.
  '''
  #
  input_seq = dict()
  if data is not None:
    #
    # Deserialize the data from tf.records
    #
    input_dict = deserialize(data)
    mag = input_dict['input_id'][:,1]
    magerr = input_dict['input_id'][:,2]
    #
    # Standardize the magnitude of the light curve 
    #
    new_mag, _ = standardize(mag, magerr)
    # input_seq['ori_mag'] = new_mag
    #
    # Create the mask_serie
    #
    mask = tf.zeros(tf.shape(new_mag), dtype=new_mag.dtype)
    #
    # Apply augmentation and add a masking tensor
    #
    if apply_white_noise:
      #
      new_mag = gaussian_noise(new_mag, noise_level)
    #
    new_input = tf.concat([
                            input_dict['input_id'][:, 0:1], #mjd
                            tf.reshape(new_mag, (-1,1)),
                            input_dict['input_id'][:, 2:] #magerr and band_sorted
                          ], axis=1)

    if apply_binning:
      new_input, mask = binning(new_input, bin_width, drop_data)
    #
    # Apply sliding_window for a fixed length sequence
    #
    new_input, mask = sliding_window(new_input, mask, input_dict['last_index'], input_dict['bands'], maxlen)
    #
    #
    #
    if apply_outlier:
      #
      # The mag_limit and mag_saturation are determined by examining 99% of the
      # standardized magnitude value of the largest dataset
      #
      mag_limit = ztf_mag['limit']
      mag_saturation = ztf_mag['saturation']
      #
      new_mag = photometric_outlier(new_input[:, 1], mask, mag_limit, mag_saturation)

      new_input = tf.concat([
                              new_input[:, 0:1], #mjd
                              tf.reshape(new_mag, (-1,1)),
                              new_input[:, 2:] #magerr and band_sorted
                            ], axis=1)

    input_seq['input'] = tf.expand_dims(new_input[:, 1], axis=-1)
    input_seq['times'] = tf.expand_dims(new_input[:, 0], axis=-1)
    input_seq['band_info'] = tf.expand_dims(new_input[:, 3], axis=-1)
    input_seq['mask'] = mask
    # input_seq['last_index'] = input_dict['last_index']
    # input_seq['bands'] = input_dict['bands']
    # input_seq['id'] = input_dict['id']
    # input_seq['label'] = input_dict['label']
    #
    return input_seq

def contrastive_data_loader(source,
                            seed=1024,
                            batch_size=100,
                            apply_white_noise=(False, True, True), 
                            noise_levels=(0.0, 0.1, 0.2), 
                            apply_binning=(False, False, True), 
                            apply_outlier=(False, False, True),
                            maxlens=(200, 100, 200), 
                            bin_widths=(5, 5, 5), 
                            drop_rates=(0.0, 0.30, 0.60), 
                            buffer_size=10000 
                           ):
    """
#     Data loader with augmentation. This method build the input format for the model.
#     The augmented data is in the sequence: anchor, positive, negative.

#     Parameters:
#     -----------------------------------------------------------------------------------
#         source (string): Record folder
                        #  NOTE: source is of the format - /train/partition_{n}/{class}/chunk_{n}_{m}.record
#         seed (int): Random seed.
#         batch_size (int): Batch size
#         num_model (int): Number of models for contrastive learning. We have considered a triplet model.
#         apply_white_noise (list of bool): Whether to apply Gaussian noise to the magnitude. Provide values for each model.
#         apply_binning (list of bool): Whether to apply binning and random dropping of bins. Provide values for each model.
#         apply_outlier (list of bool): Whether to introduce photometric outliers. Provide values for each model.
#         maxlen (list of int): The maximum length of each band sequence after sliding window. Provide values for each model.
#         bin_width (list of int): The width of the bins for binning. Provide values for each model.
#         drop_data (list of float): The fraction of bins to drop during binning. Provide valie between 0 and 1. Provide values for each model.

#     Returns:
#     -----------------------------------------------------------------------------------
#         Tensorflow Dataset: Iterator with augmented batches. 
#     """
    """Creates a tf.data.Dataset yielding (anchor, positive, negative) batches."""
    
    num_views = 3 # Anchor, Positive, Negative

    # Basic validation
    if not all(len(arg) == num_views for arg in [apply_white_noise, apply_binning, apply_outlier, maxlens, bin_widths, drop_rates]):
         raise ValueError("\n\nLength of all augmentation parameter lists/tuples must match num_views (3).")

    # --- File Discovery using Glob Pattern ---
    glob_pattern = os.path.join(source, 'partition_*', '*', 'chunk_*.record')
    print(f"\n\nSearching for TFRecord files using pattern: {glob_pattern}")
    # Keep shuffle=False here; we'll shuffle the dataset elements later
    filenames = tf.data.Dataset.list_files(glob_pattern, shuffle=False)
    # --- Check if files were found ---
    num_files_found = tf.data.experimental.cardinality(filenames)
    if num_files_found == 0:
        raise ValueError(f"No TFRecord files found matching the pattern: {glob_pattern}\n"
                         f"Please ensure the 'source' path ('{source}') is correct and files exist "
                         f"in the expected 'partition_*/CLASS/chunk_*.record' structure.")
    elif num_files_found == tf.data.UNKNOWN_CARDINALITY:
         print("\n\nWarning: Could not determine the exact number of files found (UNKNOWN_CARDINALITY). Proceeding anyway.")
        
    else:
        print(f"\n\nFound {num_files_found} TFRecord files.")
        # Optional: Print a few example filenames for verification
        print("\n\nExample filenames:\n\n")
        for f in filenames.take(2):
            print(f"- {f.numpy().decode()}")
    # --- End File Discovery and Check ---

    # Use interleave for better performance with multiple files
    dataset = filenames.interleave(tf.data.TFRecordDataset, cycle_length=AUTO, num_parallel_calls=AUTO)

    # Apply shuffle early if desired (can be slow for very large datasets)
    # dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

    loaders = []
    for i in range(num_views):
        # Using lambda function with default arguments to capture loop variables correctly
        aug_fn = lambda data, idx=i: augmentation(data,
                                                   apply_white_noise=apply_white_noise[idx],
                                                   noise_level=noise_levels[idx],
                                                   apply_binning=apply_binning[idx],
                                                   apply_outlier=apply_outlier[idx],
                                                   maxlen=maxlens[idx],
                                                   bin_width=bin_widths[idx],
                                                   drop_data=drop_rates[idx])
        view_loader = dataset.map(aug_fn, num_parallel_calls=AUTO)
        loaders.append(view_loader)

    # Zip the datasets for the different views
    zipped_dataset = tf.data.Dataset.zip(tuple(loaders))

    # Apply shuffle *after* zipping might be better if buffer_size is large
    # and memory is a concern, but shuffling before mapping ensures more randomness
    # across files earlier. Let's keep shuffle before mapping for now.
    # If shuffling after:
    shuffle_buffer_size = max(buffer_size // batch_size, 2)
    print(f"\n\nUsing shuffle buffer size: {shuffle_buffer_size} (Based on input buffer_size={buffer_size})\n\n")
    zipped_dataset = zipped_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)
    
    # The current `get_window` already does this. So, `batch` should be sufficient.
    final_loader = zipped_dataset.batch(batch_size)
    final_loader = final_loader.cache() # Cache after batching if memory allows
    final_loader = final_loader.prefetch(buffer_size=AUTO)
    #external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 276480000 exceeds 10% of free system memory.
    return final_loader





# Ensure your other data functions are defined:
# deserialize, standardize, sliding_window, get_window, ztf_band

def create_inference_loader(source,
                            batch_size,
                            maxlen, # The fixed sequence length the encoder expects
                            shuffle=False, # Shuffling is not needed for inference/evaluation
                            seed=None):
    """
    Creates a tf.data.Dataset for inference/evaluation.

    This loader reads TFRecords, applies basic preprocessing (standardization,
    fixed-length formatting), and yields batches of dictionaries containing
    both the model inputs and the associated metadata (IDs, labels).

    Args:
        source (str): Path to the directory containing TFRecord files.
        batch_size (int): The batch size for inference.
        maxlen (int): The target fixed sequence length for the model input,
                      matching the anchor view length from pre-training.
        shuffle (bool): Whether to shuffle the dataset. Defaults to False.
        seed (int): Random seed for shuffling if enabled.

    Returns:
        tf.data.Dataset: A dataset yielding batches of dictionaries.
    """

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

    # --- Mapping Function for Inference ---
    def preprocess_for_inference(data):
        input_dict = deserialize(data)
        # input_dict = deserialize(data)
        mags = input_dict['input_id'][:,1]
        magerrs = input_dict['input_id'][:,2]
        # features = input_dict['input_features']
        # mags = features[:, 1]
        # magerrs = features[:, 2]

        # 1. Standardize Magnitude (same way as during training)
        std_mags, _ = standardize(mags, magerrs)

        # 2. Reconstruct features with the standardized magnitude
        # processed_features = tf.stack([
        #     input_dict['input_id'][:, 0], # time
        #     std_mags,       # standardized magnitude
        #     input_dict['input_id'][:, 2:], # magerr & band_info
        # ], axis=1)
        processed_features = tf.concat([
                            input_dict['input_id'][:, 0:1], #mjd
                            tf.reshape(std_mags, (-1,1)),
                            input_dict['input_id'][:, 2:] #magerr and band_sorted
                          ], axis=1)

        # --- THIS IS THE KEY CHANGE ---
        # 3. Apply Padding/Cropping to the fixed `maxlen`.
        # Create a zero mask for the original length.
        initial_mask = tf.zeros(tf.shape(std_mags)[0], dtype=tf.float64)
        num_cols = tf.shape(processed_features)[1]
        # sliding_window(new_input, mask, input_dict['last_index'], input_dict['bands'], maxlen)
        # Use get_window to enforce the fixed length.
        final_features, final_mask = sequence_window(
            processed_features,
            initial_mask,
            input_dict['last_index'], 
            input_dict['bands'], 
            maxlen
        )
        # --- END OF KEY CHANGE ---

        # 3. Create the attention mask. It's all zeros (no masking) for now,
        # as padding will be handled by padded_batch later.
        # mask = tf.zeros(tf.shape(std_mags)[0], dtype=tf.float32)

        
        # 4. Prepare output dictionary with variable-length tensors
        output_dict = {}
        output_dict['input'] = tf.expand_dims(final_features[:, 1], axis=-1)
        output_dict['times'] = tf.expand_dims(final_features[:, 0], axis=-1)
        output_dict['band_info'] = tf.expand_dims(final_features[:, 3], axis=-1)
        output_dict['mask'] = tf.expand_dims(final_mask, axis=-1) # Mask is also variable length
        output_dict['id'] = input_dict['id']
        output_dict['label'] = input_dict['label']

        return output_dict
    # --- End Mapping Function ---

    processed_dataset = dataset.map(preprocess_for_inference, num_parallel_calls=AUTO)

    # --- Use padded_batch to handle variable lengths ---
    # Define the shapes to pad to. `None` in a dimension means it's variable
    # and will be padded to the maximum length of that dimension in each batch.
    # padded_shapes = {
    #     'input': tf.TensorShape([None, 1]),
    #     'times': tf.TensorShape([None, 1]),
    #     'band_info': tf.TensorShape([None, 1]),
    #     'mask': tf.TensorShape([None]),
    #     'id': tf.TensorShape([]),       # Scalar, no padding needed
    #     'label': tf.TensorShape([])   # Scalar, no padding needed
    # }

    # # Define the values to use for padding.
    # padding_values = {
    #     'input': tf.constant(0.0, dtype=tf.float32),
    #     'times': tf.constant(0.0, dtype=tf.float32),
    #     'band_info': tf.constant(0.0, dtype=tf.float32),
    #     'mask': tf.constant(1.0, dtype=tf.float32), # <-- CRUCIAL: Pad the mask with 1s
    #     'id': tf.constant(0, dtype=tf.int64),
    #     'label': tf.constant("", dtype=tf.string)
    # }

    # Apply padded_batch
    final_loader = processed_dataset.batch(
        batch_size
    )

    # Prefetch to overlap data loading with model execution
    final_loader = final_loader.prefetch(buffer_size=AUTO)

    print("Inference data loader setup complete (using padded_batch for variable length).")
    return final_loader




