import os
import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from dart.utils.helper import standardize

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
AUTO = tf.data.AUTOTUNE
os.system('clear')
tf.random.set_seed(1024)
np.random.seed(1024)

# @tf.function
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


# @tf.function
def bin_lc(sequence, bin_width, drop_data):

  time = sequence[:,0]
  # time = tf.where(tf.math.is_inf(time), tf.where(time < 0, tf.constant(-1e10, dtype=time.dtype), tf.constant(1e10, dtype=time.dtype)), time)
  # time = tf.where(tf.math.is_nan(time), tf.constant(0.0, dtype=time.dtype), time)
  
  # mag = sequence[:,1]
  # magerr = sequence[:,2]
  # band_sorted = sequence[:,3]

  min_time = tf.reduce_min(time)
  max_time = tf.reduce_max(time)
  bins = tf.range(min_time, max_time + bin_width, bin_width, dtype=time.dtype) # Create the bins with tf functions.

  #Use tf.searchsorted instead of tf.raw_ops.Bucketize
  inds = tf.searchsorted(bins, time)

  drop_inds = tf.random.shuffle(tf.range(tf.shape(inds)[0]))[:tf.cast(tf.cast(tf.shape(inds)[0], tf.float32) * drop_data, tf.int32)] # Create indices of which rows will be dropped using tf functions.


  mask = tf.ones(tf.shape(sequence)[0], dtype=tf.bool) # Create a mask that will drop rows.
  updates = tf.zeros_like(drop_inds, dtype=tf.bool)
  mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(drop_inds, axis=1), updates) # Drop data based on calculated mask.
  filtered_seq = tf.boolean_mask(sequence, mask)

  return filtered_seq


# @tf.function
def get_window(sequence, last_index, max_len, binning, bin_width, drop_data):
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

    concatenated_sequences_ta = tf.TensorArray(dtype=sequence.dtype, size=0, dynamic_size=True)
    idx = tf.constant(0, dtype=tf.int64)
    ta_idx = tf.constant(0, dtype=tf.int32)

    # Iterate through the last_index to get the start and end of each series.
    for li in last_index:

      current_serie = sequence[idx:li+1]
      
      if binning:
        current_serie = bin_lc(current_serie, bin_width, drop_data)
      
      serie_len = tf.shape(current_serie)[0]
      # serie_len = tf.cast(li+1-idx, tf.int64)
      pivot = 0

      # Check if the serie is larger than the maximum allowed
      if serie_len > max_len:

        max_val = tf.maximum(serie_len - max_len, 0)
        pivot = tf.random.uniform([],
                                    minval=tf.cast(0,tf.int64),
                                    maxval=tf.cast(max_val,tf.int64),
                                    dtype=tf.int64)


        current_serie = tf.slice(current_serie, [pivot,0], [max_len, -1])

      else:

        padding_rows = max_len - serie_len

        if padding_rows > 0:

            num_cols = tf.shape(current_serie)[1]
            zero_padding = tf.zeros([padding_rows, num_cols], dtype=current_serie.dtype)
            current_serie = tf.concat([current_serie, zero_padding], axis=0)


      idx = li + 1
      concatenated_sequences_ta = concatenated_sequences_ta.write(ta_idx, current_serie)
      ta_idx = ta_idx + 1

    result = concatenated_sequences_ta.concat()

    return result


# @tf.function
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
    input_dict['input'] = sequence

    return input_dict


# @tf.function
def augmentation(data,
                 maxlen,
                 sliding_window,
                 window_size,
                 binning,
                 bin_width,
                 drop_data,
                 white_noise):

  if data is not None:
    input_dict = deserialize(data)
    mag = input_dict['input'][:,1]
    magerr = input_dict['input'][:,2]
    mjd = input_dict['input'][:,0]
    #check for infinite values in mjd column
    # mjd = tf.where(tf.math.is_nan(mjd), tf.zeros_like(mjd), mjd)
    # mjd = tf.where(tf.math.is_inf(mjd), tf.zeros_like(mjd), mjd)
    standardized_mag, mean = standardize(mag, magerr)
    if white_noise:
      standardized_mag = gaussian_noise(standardized_mag)
    # Create a new tensor by concatenating:
    # 1. the first column of original tensor (time or mjd)
    # 2. the newly calculated standardized_mag
    # 3. the remaining columns of the original tensor (magerr and band_sorted)
    new_input = tf.concat([
                            input_dict['input'][:, 0:1], #mjd
                            tf.reshape(standardized_mag, (-1,1)),#the new standardized magnitude
                            input_dict['input'][:, 2:]#magerr and band_sorted
                        ], axis=1)

    # input_dict['input'] = new_input #replaces original with the updated version.


    # if binning:
    #   new_input = bin_lc(new_input, bin_width, drop_data)

    input = get_window(new_input, input_dict['last_index'], maxlen, binning, bin_width, drop_data)


    return input



def prefetch_batches(source,
                     seed=42,
                    batch_size=100,
                    maxlen=200,
                    sliding_window=True,
                    window_size=0.5,
                    binning = True,
                    bin_width = 5,
                    drop_data = 0.5,
                    white_noise = True):

  labels = list()
  chunks = list()
  filenames = list()
  #
  #
  #
  for p in os.listdir(path_to_read):
      for lbl in os.listdir(path_to_read+p):
          for cnk in os.listdir(path_to_read+p+"/"+lbl):
              filenames.append(path_to_read+p+"/"+lbl+'/'+cnk)

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
    dataset = dataset.shuffle(1024).map(lambda data: augmentation(data,
                                                      maxlen,
                                                      sliding_window,
                                                      window_size,
                                                      binning,
                                                      bin_width,
                                                      drop_data,
                                                      white_noise))
    #
    #
    #
    # for element in dataset:
    #   print(element)
    #   break