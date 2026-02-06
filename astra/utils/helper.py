# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import glob
import yaml
import json
import mlflow
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_config(args):
    """
    Loads the YAML configuration file.
    """
    print(f"\nLoading configuration from: {args.config}...")
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"\nError parsing YAML file: {e}")
    #
    # Override config with command-line arguments if they were provided
    # This loop checks if any command-line argument was given a value (is not None)
    # and updates the config dictionary with it.
    # ==============================================
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

@tf.function
def deserialize_for_inference(sample):
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
    input_dict['last_index'] = tf.cast(input_dict['last_index'], tf.int32)
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
    print(f"\nSearching for hyperparameters in event file in: {run_directory}")
    
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

        print("\nHyperparameters loaded successfully from event file.")
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
    parsed_dataset = base_dataset.map(deserialize_for_inference, num_parallel_calls=AUTO)
    
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

    
def process_event_file(run_log_dir, train_loss_tag, val_loss_tag, hparam_tag, learning_rate_tag, use_mlflow=False):
    #
    # --- Load Data using EventAccumulator ---
    #
    run_directory = os.path.dirname(run_log_dir)
    print(f"Loading events from: {run_log_dir}")

    try:
        model_params, training_params, data_params = load_hparams_from_event_file(run_log_dir)
        # Initialize EventAccumulator
        ea = event_accumulator.EventAccumulator(
                                                run_log_dir,
                                                size_guidance={
                                                event_accumulator.SCALARS: 0, # Load all scalars
                                                event_accumulator.TENSORS: 0  # Load all tensors
                                            }
                                        )

        # Load the events
        ea.Reload()
        
        #
        # --- Get TENSOR Tag ---
        #
        available_tags = ea.Tags()
        #
        # ----------- OPTIONAL: Print available tags ----------------------------------------
        #
        # print("Available tag categories:", list(available_tags.keys()))
        if event_accumulator.TENSORS in available_tags:
            print("Available tensor tags:", available_tags[event_accumulator.TENSORS])
        else:
            print("No 'tensors' data found in event file.")
            # If scalar tag is present, print that too:
            if event_accumulator.SCALARS in available_tags:
                print("Available scalar tags:", available_tags[event_accumulator.SCALARS])
            else:
                print("No 'scalars' data found either.")
            exit() # Exit if no tensor data is found 
        # ------------------------------------------------------------------------------------
        # --- Function to extract data from Tensor events ---
        # ------------------------------------------------------------------------------------
        def extract_tensor_data(event_acc, tag_name):
            data = []
            try:
                events = event_acc.Tensors(tag_name)
                for event in events:
                    # Convert the tensor proto to a numpy array
                    value_array = tf.make_ndarray(event.tensor_proto)
                    # Assuming it was logged as a scalar, it should be a 0-D array
                    # Extract the scalar value using .item()
                    scalar_value = value_array.item()
                    data.append((event.step, scalar_value))
            except KeyError:
                print(f"\n\nWarning: Tag '{tag_name}' not found in tensors.")
            except Exception as e:
                print(f"\n\nError processing tag '{tag_name}': {e}")
            return data
        # ------------------------------------------------------------------------------------
        #
        # --- Extract Data using the function ---
        #
        print(f"\nAttempting to read tags from 'tensors' category...")
        train_loss_data = extract_tensor_data(ea, train_loss_tag)
        if train_loss_data:
            print(f"\nFound {len(train_loss_data)} training loss points (tag: {train_loss_tag}) from tensors.")
        # ------------------------------------
        val_loss_data = extract_tensor_data(ea, val_loss_tag)
        if val_loss_data:
            print(f"\n\nFound {len(val_loss_data)} validation loss points (tag: {val_loss_tag}) from tensors.")
        # Check if the tag exists in the 'tensors' category, as text is stored there
        # ------------------------------------
        if hparam_tag not in ea.Tags()['tensors'] and model_params is None:
            print(f"ERROR: Hyperparameter tag '{hparam_tag}' not found in the 'tensors' category of the event file.")
            print("Available tensor tags:", ea.Tags()['tensors'])
            return 
        # ------------------------------------
        learning_rate = extract_tensor_data(ea, learning_rate_tag)
        if learning_rate:
            print(f"\n\nFound {len(learning_rate)} LR points (tag: {val_loss_tag}) from tensors.")
        
        # ------------------------------------
        # --- Store Loss Data ---
        # ------------------------------------
        # Create a Pandas DataFrame inside "run_log_dir" to store the loss data
        #
        if not train_loss_data and not val_loss_data and not learning_rate:
            print("\n\nNo loss and LR data found. Skipping this run.")
            return
        #
        # Get total epochs from both lists
        #
        all_steps = sorted(list(set([e[0] for e in train_loss_data] + [e[0] for e in val_loss_data])))

        train_losses_dict = {e[0]: e[1] for e in train_loss_data}
        val_losses_dict = {e[0]: e[1] for e in val_loss_data}
        LR_dict = {e[0]: e[1] for e in learning_rate}

        df_data = {
            'epoch': [s + 1 for s in all_steps], # Convert step (0-based) to epoch number (1-based)
            'train_loss': [train_losses_dict.get(s, None) for s in all_steps],
            'val_loss': [val_losses_dict.get(s, None) for s in all_steps],
            'LR': [LR_dict.get(s, None) for s in all_steps] if learning_rate else None
        }
        loss_df = pd.DataFrame(df_data)
        # ===============================================
        # Remove MLflow logging before packaging
        # Change the "run_name" to the format - {run_timestamp}_server_name"
        # --- Start MLflow Run ---
        #
        if use_mlflow:
            # --- MLFLOW MODE ---
            run_timestamp = os.path.basename(run_directory)
            with mlflow.start_run(run_name=f"{run_timestamp}_nvidia") as run:
                #
                # Add a tag for easier filtering (optional but good practice)
                mlflow.set_tag("model_type", "AstraNet")
                # ===============================================
                # Change the "run_name" to the format - {run_timestamp}_server_name"
                #
                print(f"\n\nStarted MLflow Run: {run.info.run_id}/ run_name: {run_timestamp}_nvidia\n\n")
                # Log Hyperparameters
                mlflow.log_params({"run_timestamp":run_timestamp,
                                   "model_params":model_params, 
                                   "training_params":training_params, 
                                   "data_params":data_params})
                # Log Metrics
                for step, value in train_loss_data:
                    mlflow.log_metric("loss/epoch_train", value, step=step)
                for step, value in val_loss_data:
                    mlflow.log_metric("loss/epoch_val", value, step=step)
                for step, value in learning_rate:
                    mlflow.log_metric("learning_rate", value, step=step)
                
                print("Successfully logged params and metrics to MLflow.")

        else:
            # --- LOCAL MODE ---
            loss_df.to_csv(os.path.join(os.path.dirname(run_log_dir), 'loss_summary.csv'))
            fig = loss_df.plot(x='epoch', y=['train_loss', 'val_loss'], kind='line', marker='o', markersize=3 ,title='Training and Validation Loss over Epochs', ylabel='Loss', xlabel='Epoch').get_figure()
            fig.savefig(os.path.join(os.path.dirname(run_log_dir), 'loss_plot.png'))
            print(f"\n\n--- DataFrame Summary --- stored in: {os.path.dirname(run_log_dir)} ---")

    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        traceback.print_exc()


def backfill_mlflow_and_plot_loss(run_log_dir=None, train_loss_tag=None, val_loss_tag=None, hparam_tag=None, learning_rate_tag=None, use_mlflow=False):

    try:
        if run_log_dir is None:
            raise FileNotFoundError(f"\nError: Log directory not found: {run_log_dir}")
        
        elif train_loss_tag is None or val_loss_tag is None:
            raise ValueError("\n\nPlease provide 'train_loss_tag', and 'val_loss_tag' parameters.")
    
        if use_mlflow:
            #
            # Remove MLflow logging before packaging
            #
            # Initialize MLflow Tracking
            # Set an URI and Experiment name for MLflow
            #
            mlflow.set_tracking_uri("http://localhost:8000")
            mlflow.set_experiment("ASTRA(Pre-training)")
            print("MLflow mode is ON. Will log to the configured server.")
        else:
            print("MLflow mode is OFF. Will save CSV and plots to local run directories.")

        #
        # The user-defined path is a directory
        #
        if os.path.isdir(run_log_dir):
            print(f"Path '{run_log_dir}' is a directory. Searching for event files recursively...")
            process_event_file(run_log_dir, train_loss_tag, val_loss_tag, hparam_tag, learning_rate_tag, use_mlflow)
        #       
        # For user-defined path is a file
        #
        elif os.path.isfile(run_log_dir):
            print(f"Path '{run_log_dir}' is a file. Attempting to load directly...")
            process_event_file(run_log_dir, train_loss_tag, val_loss_tag, hparam_tag, learning_rate_tag, use_mlflow)
        #  
        # The path doesn't exist
        #
        else:
            print(f"Error: The specified path '{run_log_dir}' does not exist or is not a valid file/directory.")
            return
        # --- End of plot_loss function ---
    
    except Exception as e:
        print(f"\n\n{e}")
        return
    
    

    




if __name__ == "__main__":
    #
    # Example usage of plot_loss function
    #
    # run_log_dir = "/media3/majumder/contrastive_loss_res/"
    run_log_dir = "/home/nvidia/workplace/contrastive_loss_res/run_20251231_140014/"
    # run_log_dir = "media3/majumder/contrastive_loss_res/run_20250822_215059/events.out.tfevents.1755892260.clrlsstsrv02.in2p3.fr.2826236.0.v2"

    backfill_mlflow_and_plot_loss(
        run_log_dir=run_log_dir,
        train_loss_tag='loss/epoch_train',
        val_loss_tag='loss/epoch_val',
        hparam_tag = 'hyperparameters',
        learning_rate_tag='learning_rate',
        use_mlflow=True
    )

    # model_params, training_params, data_params = load_hparams_from_event_file(run_log_dir)
    # print(model_params, training_params, data_params)
    