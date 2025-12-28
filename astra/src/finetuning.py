# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import logging
import numpy as np
import tensorflow as tf
from astra.utils.helper import standardize
from astra.src.preprocessing import deserialize, sliding_window, gaussian_noise
# ===========================================================
AUTO = tf.data.AUTOTUNE
os.system('clear')
logging.getLogger('tensorflow').setLevel(logging.ERROR) 
# ===========================================================

# Ensure your other data functions are defined:
# deserialize, standardize, get_window (if using fixed length)

def finetune_data_loader(source_dir,
                            batch_size=None,
                            label_map=None, 
                            max_len=None, 
                            threshold=18.0, 
                            is_training=True, 
                            buffer_size=10000,
                            apply_white_noise=True
                        ): 
    """
    Creates a pre-processed tf.data.Dataset for fine-tuning 
    NOTE: the pre-processing differs from the original data augmentation

    Parameters:
    ----------------------------------------------------------------------------------------
    source_dir (str): Path to the root of the original dataset (e.g., './dataset').
    batch_size (int): Batch_size for fine-tined model
    label_map (dict): A nested dictionary mapping class names to the desired number of samples.
                            e.g., {'CEP': 20000, 'AGN': 6000}
                        & mapping string labels to integer IDs
    max_len (dict): The maximum length of each band sequence after sliding window
    buffer_size (int): shuffle buffer size
    threshold (float): Brightness threshold for finetuning
    is_training (bool):  Flag to enable shuffling and taking a fraction for training data
    apply_white_noise (bool): Apply white noise for the training set only.

    Returns:
    ----------------------------------------------------------------------------------------
    Tensorflow Dataset: Iterator with pre-processed batches. 
    """
    # --------------------------------------------------------------------------------------
    glob_pattern = os.path.join(source_dir, '*', 'chunk_*.record')
    filenames_dataset = tf.data.Dataset.list_files(glob_pattern, shuffle=is_training)

    
    @tf.function
    def filter_samples(data_record):
        '''
        Helper function to check the threshold and remap labels (if needed)
        '''
        # --------------------------------------------------------------------------------
        # Load the records
        input_dict = deserialize(data_record)
        # --------------------------------------------------------------------------------
        # Remap classes to broader classes (if any)
        #
        label_str = input_dict['label']
        # --------------------------------------------------------------------------------
        if label_str == b'ACEP' or label_str == b'DCEP' or label_str == b'T2CEP':
            mapped_label = tf.constant('CEP', dtype=tf.string)
        elif label_str == b'RRab' or label_str == b'RRc' or label_str == b'RRd': 
            mapped_label = tf.constant('RRLY', dtype=tf.string)
        else:
            mapped_label = label_str 
        # --------------------------------------------------------------------------------
        # Check the brightness threshold
        #
        mag = input_dict['input_id'][:, 1]
        magerr = input_dict['input_id'][:, 2]
        last_index = input_dict['last_index']
        start_index = tf.constant(0, dtype=tf.int32)
        is_candidate = tf.constant(False, dtype=tf.bool)
        #
        # Loop through the bands to calculate weighted mean for each
        #
        for i in tf.range(tf.shape(last_index)[0]):
            end_index = last_index[i] + 1
            mag_band = mag[start_index:end_index]
            magerr_band = magerr[start_index:end_index]
            # Find the weighted mean
            _, weighted_mean = standardize(mag_band, magerr_band)
            #
            # Compare the weighted mean with the threshold
            # Find a bright detection in the band, 
            # if found, no need to check others
            #
            if weighted_mean < threshold:
                is_candidate = tf.constant(True, dtype=tf.bool)
                break 
            #
            start_index = end_index
            
        return input_dict, mapped_label, is_candidate
    #
    # ---------------- Find all candidates and count them --------------------------------
    print("\n--- Starting pre-pass to find and count all candidates ...\n")
    #
    # Create a temporary dataset to find candidates
    initial_dataset = filenames_dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO).map(filter_samples, num_parallel_calls=AUTO)
    # Filter to get only the candidates
    candidate_dataset = initial_dataset.filter(lambda d, l, is_c: is_c)
    # CRITICAL: Cache the full candidate dataset to avoid re-reading from disk
    print("\n--- Caching all candidates for efficient processing. This may take a moment...\n")
    candidate_dataset = candidate_dataset.cache()
    # Count the candidates
    candidate_count = candidate_dataset.reduce(np.int64(0), lambda x, _: x + 1)
    print(f"\n---   Found {candidate_count} total candidate objects brighter than magnitude {threshold}.\n")
    # ------------------------------------------------------------------------------------
    if candidate_count == 0:
        raise ValueError("\nValueError: No candidate objects found. Check your threshold or data.\n")
    # ------------------------------------------------------------------------------------
    # --------------------------- Build the final data pipeline --------------------------
    #
    if is_training:
        #
        print(f"\n--- Building a user-defined training set.")
        subset_datasets = []
        # --------------------------------------------------------------------------------
        # For each class, create a filtered, shuffled, and limited dataset
        #
        for class_name_str, class_info in label_map.items():
            # Get the user-defined target count for this class
            target_count = class_info['count']
            class_name_bytes = tf.constant(class_name_str.encode('utf-8'), dtype=tf.string)
            # Filter the cached dataset for the current class
            class_ds = candidate_dataset.filter(lambda d, l, is_c: l == class_name_bytes)
            # Count how many candidates are available for this class
            available_count = class_ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()
            # Take the smaller of the target count or the available count
            num_to_take = min(target_count, available_count)
            if num_to_take < target_count:
                print(f"\n  WARNING for class '{class_name_str}': Requested {target_count}, but only {available_count} are available. Taking all {num_to_take}.")
            else:
                print(f"\n  Class '{class_name_str}': Found {available_count} candidates, taking the requested {num_to_take}.")
            # Shuffle and take the target number of samples
            class_subset = class_ds.shuffle(buffer_size=available_count).take(num_to_take)
            subset_datasets.append(class_subset)
        
        # ---------------- Combine the subsets into a single dataset ----------------
        # Check if any subsets were actually created
        #
        if not subset_datasets:
            # If no data was found for any class, return an empty dataset
            # We get the structure from the original candidate_dataset 
            final_dataset = tf.data.Dataset.from_generator(lambda: None, output_signature=candidate_dataset.element_spec)
        else:
            # Take the first subset as the starting point
            final_dataset = subset_datasets[0]
            # Loop through the REST of the subsets and concatenate them
            for ds in subset_datasets[1:]:
                final_dataset = final_dataset.concatenate(ds)
        # ---------------------------------------------------------------------------      
    else:
        # For validation, use all candidates found
        final_dataset = candidate_dataset
        print(f"\n--- Using all {candidate_count} candidates for validation.\n")
    #
    # ------------------ Helper function to do final preprocessing ------------------
    # Create the lookup table for the FINAL mapped labels
    final_class_names = [k.encode('utf-8') for k in label_map.keys()]
    final_class_ids = [v['id'] for v in label_map.values()]
    keys_tensor = tf.constant(final_class_names)
    vals_tensor = tf.constant(final_class_ids)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1)
    #
    @tf.function
    def preprocess_and_map_label(input_dict, mapped_label, is_candidate):
        # 
        features = input_dict['input_id']
        mags, magerrs = features[:, 1], features[:, 2]
        # Standardize the magnitude of the light curve
        std_mags, _ = standardize(mags, magerrs)
        # Apply augmentation for the training set
        # Set the noise level
        if is_training and apply_white_noise:
            std_mags = gaussian_noise(std_mags, noise_level=0.02)

        processed_features = tf.concat([
                                        features[:, 0:1], #mjd
                                        tf.reshape(std_mags, (-1,1)),
                                        features[:, 2:] #magerr and band_sorted
                                    ], axis=1)
        #
        # 0: Unmasked values 
        # 1: Masked values 
        #
        initial_mask = tf.zeros(tf.shape(std_mags)[0], dtype=tf.float32)
        #
        # Enforce fixed length
        # 
        final_features, final_mask = sliding_window(
                                                    processed_features,
                                                    initial_mask,
                                                    input_dict['last_index'], 
                                                    input_dict['bands'], 
                                                    max_len
                                                )
        # Prepare model inputs
        model_inputs = {
                        'input': tf.expand_dims(final_features[:, 1], axis=-1),
                        'times': tf.expand_dims(final_features[:, 0], axis=-1),
                        'band_info': tf.expand_dims(final_features[:, 3], axis=-1),
                        'mask': final_mask
                    }
        # Map the remapped string label to a final integer ID
        label_id = table.lookup(mapped_label)
        #
        return model_inputs, label_id
    # -------------------------------------------------------------------------------------
    # Apply the final preprocessing
    #
    final_dataset = final_dataset.map(preprocess_and_map_label, num_parallel_calls=AUTO)
    # Filter out any samples with unknown labels (ID = -1)
    final_dataset = final_dataset.filter(lambda inputs, label: label != -1)
    #
    if is_training:
        # Shuffle the small subset again for good measure
        final_dataset = final_dataset.shuffle(buffer_size=buffer_size)
    #
    # Batch and prefetch
    #
    final_loader = final_dataset.batch(batch_size).prefetch(buffer_size=AUTO)
    #
    return final_loader


def finetune_model(encoder_model, num_classes, unfreeze_layers=None): 
    """
    Takes a pre-trained ASTRA encoder and adds a classification head for fine-tuning.

    Parameters:
    ------------------------------------------------------------------------------------
        encoder_model (tf.keras.Model): The pre-trained ASTRA encoder model.
        num_classes (int): The number of output classes for the classifier.
        unfreeze_layers (int, optional): The number of layers to unfreeze from the end
                                         of the encoder. If None, the entire encoder
                                         remains frozen (linear probing). If 'all',
                                         all layers are unfrozen.

    Returns:
    ------------------------------------------------------------------------------------
        tf.keras.Model: The complete, compiled classification model.
    """
    #
    # -------------------- Control which layers are trainable --------------------------
    #
    if unfreeze_layers is None: # Default: Freeze the entire encoder (linear probing)
        print("\n--- Encoder is FROZEN. Performing linear probing.")
        encoder_model.trainable = False
    elif unfreeze_layers == 'all': # Unfreeze the entire encoder for full fine-tuning
        print("\n--- Encoder is FULLY UNFROZEN for end-to-end fine-tuning.")
        encoder_model.trainable = True
    else:
        # Unfreeze the last N layers of the main Transformer encoder block
        print(f"\n--- Encoder is PARTIALLY UNFROZEN. Unfreezing last {unfreeze_layers} encoder layers.")
        encoder_model.trainable = True # Allow setting trainability per layer
        # Assumes the main encoder is a single layer in the encoder_model
        transformer_encoder_block = None
        for layer in encoder_model.layers:
             if 'encoder' in layer.name: # Find the main Encoder block
                  transformer_encoder_block = layer
                  break
        if transformer_encoder_block:
            # Access the list of layers in the Encoder's __init__
            # This assumes the list is named 'enc_layers'. Change if named it differently.
            encoder_layer_list = transformer_encoder_block.enc_layers
            # Freeze all layers first
            print(f"\n--- Freezing all {len(encoder_layer_list)} layers in the encoder block...")
            for layer in encoder_layer_list:
                layer.trainable = False
            # Then, unfreeze the last N layers
            if unfreeze_layers > len(encoder_layer_list):
                print(f"\nWarning: Trying to unfreeze {unfreeze_layers} but encoder only has {len(encoder_layer_list)} layers. Unfreezing all.")
                unfreeze_layers = len(encoder_layer_list)

            print(f"\n--- Unfreezing the last {unfreeze_layers} layers...")
            for layer in encoder_layer_list[-unfreeze_layers:]:
                layer.trainable = True
                print(f"  -- Unfreezing layer: {layer.name}")
        else:
            print("\nWarning: Could not find main 'encoder' block to partially unfreeze.")
    # ===========================================================================================================================================
    #
    # --------------------------- Build the new classification model --------------------------
    # Get the inputs from the pre-trained encoder model
    inputs = encoder_model.input
    # Get the output of the pre-trained encoder (the embeddings)
    x = encoder_model.output
    # Add a new classification head
    x = tf.keras.layers.Dropout(0.2)(x) # Add dropout for regularization
    outputs = tf.keras.layers.Dense(num_classes, name='classifier_head')(x) # Output logits
    # Create the final model
    finetune_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ASTRA_Classifier")

    return finetune_model
    # =============================================================================================================



