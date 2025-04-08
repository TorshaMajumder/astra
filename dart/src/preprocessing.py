import os
import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from dart.bands.bands import ztf_band
from dart.utils.helper import standardize


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
AUTO = tf.data.AUTOTUNE
os.system('clear')
tf.random.set_seed(1024)
np.random.seed(1024)


def photometric_outlier(tt, mask, mag_limit, mag_saturation):
  """
  Introduces photometric outliers to a TensorFlow tensor.

  Args:
      tt: The input TensorFlow tensor.
      mag_limit: The upper limit for magnitude.
      mag_saturation: The lower limit for magnitude.

  Returns:
      The modified TensorFlow tensor with added outliers.
  """
  # Get the indices where the mask is 1
  valid_indices = tf.where(mask)
  random_index = tf.constant(-1, dtype=tf.int64)

  
  num_valid_indices = tf.shape(valid_indices)[0]
  if num_valid_indices != 0:

    random_index = tf.random.shuffle(valid_indices)[0] # Use tf.random.shuffle for TensorFlow 1.x
    random_index = tf.squeeze(random_index)
    mag = tf.gather(tt, random_index)
    mag =  mag_saturation - tf.random.uniform([], 0, 0.5, dtype=tf.float32),  ## further lower the value of mag
    mag = tf.cast(mag, tt.dtype)  # Ensure data types match
    indices = tf.reshape(random_index, [1])
    updates = tf.reshape(mag, [1])
    tt = tf.tensor_scatter_nd_update(tt, tf.expand_dims(indices, axis=-1), updates)

  return tt, random_index

def bin_lc3(sequence, bin_width, drop_data):

  time = sequence[:,0]

  min_time = tf.reduce_min(time)
  max_time = tf.reduce_max(time)

  bins = tf.range(min_time, max_time + bin_width, bin_width, dtype=time.dtype)

  bin_index = tf.searchsorted(bins, time, side="left") #left for lower-bound

  uniq_bin_index, time_index = tf.unique(bin_index)


  num_bins_to_drop = tf.cast(tf.cast(tf.shape(uniq_bin_index)[0], tf.float32) * drop_data, tf.int32)

  num_bins_to_drop = tf.maximum(num_bins_to_drop, 1) # Handle case where num_bins_to_drop is 0


  bin_index_drop = tf.random.shuffle(uniq_bin_index)[:num_bins_to_drop]

  time_index_drop = tf.math.reduce_any(tf.equal(time_index[:, tf.newaxis], bin_index_drop), axis=1)

  mask_serie = tf.where(time_index_drop, tf.zeros_like(time_index_drop, dtype=sequence.dtype), tf.ones_like(time_index_drop, dtype=sequence.dtype))

  new_serie = tf.where(tf.expand_dims(time_index_drop, axis=-1), tf.zeros_like(sequence), sequence)


  return new_serie, mask_serie


def process_serie(current_serie, mask_serie, max_len, num_cols):
    assert current_serie.shape[0] == mask_serie.shape[0]

    serie_len = tf.shape(current_serie)[0]
    pivot = 0

    # Check if the serie is larger than the maximum allowed
    if serie_len > max_len:
        max_val = tf.maximum(serie_len - max_len, 0)

        pivot = tf.random.uniform([],
                                    minval=tf.cast(0, tf.int64),
                                    maxval=tf.cast(max_val, tf.int64),
                                    dtype=tf.int64)
        current_serie = tf.slice(current_serie, [pivot, 0], [max_len, -1])
        mask_serie = tf.slice(mask_serie, [pivot], [max_len])
    else:
        padding_rows = max_len - serie_len
        if padding_rows > 0:
            zero_padding = tf.zeros([padding_rows, num_cols], dtype=current_serie.dtype)
            current_serie = tf.concat([current_serie, zero_padding], axis=0)
            mask_serie = tf.concat([mask_serie, tf.zeros([padding_rows], dtype=mask_serie.dtype)], axis=0)

    return current_serie, mask_serie



def get_window(sequence, mask, last_index, bands_tensor, max_len):
    """
    Extracts sliding windows from sequences in the input dictionary,
    padding sequences shorter than max_len with zeros.

    Args:
      input_dict: A dictionary containing 'input' (a tensor of shape [num_steps, num_features])
                  and 'last_index' (a tensor of the indices of the end of each series).
      max_len: The maximum length of each sequence.

    Returns:
      An updated input_dict with 'new_input' containing the processed sequences.
    """

    # ztf_band = {'g':23.4, 'r': 234.5, 'i': 345.7} # Global variable
    num_cols = tf.shape(sequence)[1]

    band_indices = {band: tf.cond(
        tf.reduce_any(tf.equal(bands_tensor, band)),
        lambda: tf.cast(tf.where(tf.equal(bands_tensor, band))[0][0], tf.int64), # Cast to tf.int64 if condition is True
        lambda: tf.constant(-1, dtype=tf.int64) # Otherwise, return -1 as tf.int64
    ) for band in ztf_band.keys()}


    series = []
    mask_series = []
    idx = tf.cast(0, dtype=tf.int64)

    for fil in ztf_band.keys():
      index_in_bands = band_indices[fil]
      is_in_bands = index_in_bands != -1

      if is_in_bands:
          current_serie = sequence[idx:last_index[index_in_bands] + 1]
          mask_serie = mask[idx:last_index[index_in_bands] + 1]
          current_serie, mask_serie = process_serie(current_serie, mask_serie, max_len, num_cols)
          idx = last_index[index_in_bands] + 1
      else:
          current_serie = tf.zeros((max_len, num_cols), dtype=sequence.dtype)
          mask_serie = tf.zeros(max_len, dtype=mask.dtype)

      series.append(current_serie)
      mask_series.append(mask_serie)

    result_series, result_mask = tf.concat(series, axis=0), tf.concat(mask_series, axis=0)
    return result_series, result_mask

def gaussian_noise(sequence, noise_level=0.1):
    """
    Adds white Gaussian noise to a sequence.

    Args:
        sequence: A TensorFlow tensor representing the sequence.
        noise_level: The standard deviation of the Gaussian noise.

    Returns:
        A TensorFlow tensor representing the sequence with added noise.
    """
    # Get the length of the sequence
    length = tf.shape(sequence)[0]
    # Reshape sequence to (length, 1) to ensure correct broadcasting
    sequence = tf.reshape(sequence, (length, 1))
    # print("length", length, sequence.dtype)

    # Generate Gaussian noise with mean 0 and specified standard deviation
    noise = tf.random.normal(shape=(length, 1), mean=0.0, stddev=noise_level, dtype=sequence.dtype)
    # print("noise", noise)
    # print("seq", sequence)

    # Add the noise to the sequence
    noisy_sequence = sequence + noise
    # print("noisy", noisy_sequence)

    return noisy_sequence




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





def deserialize(sample):
    '''
    "mjd", "mag", "magerr", "band_sorted"
    '''

    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.string),
                        'bands': tf.io.VarLenFeature(dtype=tf.string),
                        'last_index': tf.io.VarLenFeature(dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.int64)}

    num_keys = 4

    sequence_features = dict()

    for i in range(num_keys):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['id']   = tf.cast(context['id'], tf.int64)
    input_dict['last_index'] = tf.sparse.to_dense(context['last_index'])#changed
    input_dict['label']  = tf.cast(context['label'], tf.string)
    input_dict['bands']  = tf.sparse.to_dense(context['bands'])


    casted_inp_parameters = []

    for i in range(num_keys):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float64)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input_id'] = sequence

    return input_dict


def augmentation(data,
                 maxlen,
                 sliding_window,
                 window_size,
                 binning,
                 bin_width,
                 keep_data,
                 add_noise,
                 add_outlier):

  if data is not None:
    input_dict = deserialize(data)
    mag = input_dict['input_id'][:,1]
    magerr = input_dict['input_id'][:,2]
    mjd = input_dict['input_id'][:,0]
    
    new_mag, mean = standardize(mag, magerr)
    

    if add_noise:
      new_mag = gaussian_noise(new_mag)

    new_input = tf.concat([
                            input_dict['input_id'][:, 0:1], #mjd
                            tf.reshape(new_mag, (-1,1)),#the new standardized magnitude
                            input_dict['input_id'][:, 2:]#magerr and band_sorted
                        ], axis=1)

    # # input_dict['input'] = new_input #replaces original with the updated version.


    if binning:
      new_input, mask = bin_lc3(new_input, bin_width, keep_data)

    new_input, mask = get_window(new_input, mask, input_dict['last_index'], input_dict['bands'], maxlen)

    if add_outlier:
      mag_limit = 21
      mag_saturation = 13.5
      new_mag, idx = photometric_outlier(new_input[:, 1], mask, mag_limit, mag_saturation)

    new_input = tf.concat([
                            new_input[:, 0:1], #mjd
                            tf.reshape(new_mag, (-1,1)),#the new standardized magnitude
                            new_input[:, 2:]#magerr and band_sorted
                        ], axis=1)


    return new_input









def prefetch_batches(source,
                        seed=42,
                        batch_size=100,
                        maxlen=200,
                        sliding_window=True,
                        window_size=0.5,
                        binning = True,
                        bin_width = 5,
                        drop_data = 0.6,
                        add_noise = True,
                        add_outlier = True):

    labels = list()
    chunks = list()
    filenames = list()
    #
    #
    #
    for p in os.listdir(source):
        for lbl in os.listdir(source+p):
            for cnk in os.listdir(source+p+"/"+lbl):
                filenames.append(source+p+"/"+lbl+'/'+cnk)

    for f in filenames:
        #
        #
        #
        dataset = tf.data.TFRecordDataset(f)
        #
        #
        #
        #
        #
        #
        dataset = dataset.shuffle(seed).map(lambda data: augmentation(data,
                                                                        maxlen,
                                                                        sliding_window,
                                                                        window_size,
                                                                        binning,
                                                                        bin_width,
                                                                        drop_data,
                                                                        add_noise,
                                                                        add_outlier))

        dataset = dataset.padded_batch(batch_size).cache()
        dataset = dataset.prefetch(buffer_size=AUTO)
    
        #
        #
        #
    return dataset




