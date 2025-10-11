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
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

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
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input_id'] = sequence

    return input_dict



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
    # For later runs
    # weights = 1.0 / (tf.square(err) + 1e-6) # Added epsilon for safety
    weights = 1.0 / tf.square(err)
    weighted_sum = tf.reduce_sum(x * weights)
    sum_of_weights = tf.reduce_sum(weights)
    mean = tf.math.divide_no_nan(weighted_sum, sum_of_weights) 
    #
    # Center the data by subtracting the weighted mean
    #
    x_new = x - mean

    return x_new , mean

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



def filter_by_class(source, target_counts, shuffle=True, seed=42):
    """
    Filters a TFRecord dataset to get a specific number of samples for given classes.

    Args:
        filenames (tf.data.Dataset): A dataset of TFRecord filenames.
        target_counts (dict): A dictionary mapping class names (bytes) to the desired 
                              number of samples (int). E.g., {b'A': 1000, b'B': 500}.
        parse_fn (function): The function to parse a single TFRecord proto.

    Returns:
        tf.data.Dataset: A new dataset containing the specified number of samples
                         for each class.
    """
    # 1. Create a single, parsed dataset from all files.
    # Using AUTOTUNE is best practice for performance.
    AUTO = tf.data.AUTOTUNE

    glob_pattern = os.path.join(source, 'partition_*', '*', 'chunk_*.record')
    filenames = tf.data.Dataset.list_files(glob_pattern, shuffle=shuffle, seed=seed)
    base_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    parsed_dataset = base_dataset.map(deserialize, num_parallel_calls=AUTO)
    
    # Cache the parsed dataset for efficiency, as we will iterate over it multiple times.
    parsed_dataset = parsed_dataset.cache()

    # 2. Create a list of limited datasets, one for each target class.
    limited_datasets = []
    print("Filtering for each class...")
    for class_name, count in target_counts.items():
        print(f" - Getting {count} samples for class: {class_name.decode()}")

        # Filter for the specific class
        class_ds = parsed_dataset.filter(lambda x: x['label'] == class_name)
        
        # Take the specified number of samples
        class_ds = class_ds.take(count)
        
        limited_datasets.append(class_ds)

    if not limited_datasets:
        return tf.data.Dataset.from_tensor_slices([]) # Return an empty dataset if dict is empty

    # 3. Concatenate all the small datasets into one.
    # Start with the first dataset in the list.
    final_dataset = limited_datasets[0]
    # Sequentially concatenate the rest.
    for ds_to_add in limited_datasets[1:]:
        final_dataset = final_dataset.concatenate(ds_to_add)

    # 4. Important: Shuffle the final dataset.
    # The dataset is currently ordered (all 'A's, then all 'B's, etc.).
    # Shuffling is crucial for training a model.
    total_samples = sum(target_counts.values())
    final_dataset = final_dataset.shuffle(buffer_size=total_samples, reshuffle_each_iteration=True)

    print(f"\nSuccessfully created a dataset with {total_samples} total samples.")
    
    return final_dataset.prefetch(AUTO)

    


