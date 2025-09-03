import tensorflow as tf
import os
import numpy as np
from astra.utils.labels import *
from astra.utils.helper import standardize
from astra.src.preprocessing import deserialize

# Make sure AUTO = tf.data.AUTOTUNE is defined globally
AUTO = tf.data.AUTOTUNE

# Ensure your other data functions are defined:
# deserialize, standardize, get_window (if using fixed length)

def finetune_data_loader(source_dir,
                            batch_size=None,
                            label_map=None, # A dict mapping string labels to integer IDs
                            maxlen=None, # The fixed sequence length the encoder expects
                            threshold=18.0, # Brightness threshold
                            fraction_to_use=0.01, # The fraction of data to use (e.g., 1%)
                            is_training=True, # Flag to enable shuffling and taking a fraction
                            apply_white_noise=True): 
    """
    Creates a tf.data.Dataset for fine-tuning by remapping labels,
    filtering by a brightness threshold, and taking a subset of candidates.
    """
    
    glob_pattern = os.path.join(source_dir, 'partition_*', '*', 'chunk_*.record')
    filenames_dataset = tf.data.Dataset.list_files(glob_pattern, shuffle=is_training)

    # --- Helper function to remap labels and check the threshold ---
    @tf.function
    def _remap_and_filter_samples(data_record):
        input_dict = deserialize(data_record)
        
        # 1. Remap detailed labels to broader ones
        label_str = input_dict['label']
        if label_str == b'ACEP' or label_str == b'DCEP' or label_str == b'T2CEP':
            mapped_label = tf.constant('CEP', dtype=tf.string)
        elif label_str == b'RRab' or label_str == b'RRc' or label_str == b'RRd': # Using your new examples
            mapped_label = tf.constant('RRLY', dtype=tf.string)
        else:
            mapped_label = label_str # Keep other labels as is

        # 2. Check the brightness threshold
        mag = input_dict['input_id'][:, 1]
        magerr = input_dict['input_id'][:, 2]
        last_index = input_dict['last_index']
        
        start_index = tf.constant(0, dtype=tf.int64)
        is_candidate = tf.constant(False, dtype=tf.bool)

        # Loop through the bands to calculate weighted mean for each
        for i in tf.range(tf.shape(last_index)[0]):
            end_index = last_index[i] + 1
            mag_band = mag[start_index:end_index]
            magerr_band = magerr[start_index:end_index]
            
            # Use a dummy tensor for standardized mag since we only need the mean here
            _, weighted_mean = standardize(mag_band, magerr_band)
            
            if weighted_mean < threshold:
                is_candidate = tf.constant(True, dtype=tf.bool)
                break # Found a bright band, no need to check others
            
            start_index = end_index
            
        return input_dict, mapped_label, is_candidate

    # --- Pre-pass to find all candidates and count them ---
    print("Starting pre-pass to find and count all candidate objects...")
    
    # Create a temporary dataset to find candidates
    initial_dataset = filenames_dataset.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=AUTO).map(_remap_and_filter_samples, num_parallel_calls=AUTO)
    
    # Filter to get only the candidates
    candidate_dataset = initial_dataset.filter(lambda d, l, is_c: is_c)

    # Count the candidates. This will iterate through the dataset once.
    candidate_count = candidate_dataset.reduce(np.int64(0), lambda x, _: x + 1)
    print(f"Found {candidate_count} total candidate objects brighter than magnitude {threshold}.")
    
    if candidate_count == 0:
        raise ValueError("No candidate objects found. Check your threshold or data.")

    


def finetune_model(encoder_model,
                            num_classes,
                            unfreeze_layers=None): # Num layers to unfreeze from the end
    """
    Takes a pre-trained encoder and adds a classification head for fine-tuning.

    Args:
        encoder_model (tf.keras.Model): The pre-trained DART encoder model.
        num_classes (int): The number of output classes for the classifier.
        unfreeze_layers (int, optional): The number of layers to unfreeze from the end
                                         of the encoder. If None, the entire encoder
                                         remains frozen (linear probing). If 'all',
                                         all layers are unfrozen.

    Returns:
        tf.keras.Model: The complete, compiled classification model.
    """
    # --- Control which layers are trainable ---
    if unfreeze_layers is None:
        # Default: Freeze the entire encoder (linear probing)
        print("Encoder is FROZEN. Performing linear probing.")
        encoder_model.trainable = False
    elif unfreeze_layers == 'all':
        # Unfreeze the entire encoder for full fine-tuning
        print("Encoder is FULLY UNFROZEN for end-to-end fine-tuning.")
        encoder_model.trainable = True
    else:
        # Unfreeze the last N layers of the main Transformer encoder block
        print(f"Encoder is PARTIALLY UNFROZEN. Unfreezing last {unfreeze_layers} encoder layers.")
        encoder_model.trainable = True # Allow setting trainability per layer
        # Assumes the main encoder is a single layer in the encoder_model
        transformer_encoder_block = None
        for layer in encoder_model.layers:
             if 'encoder' in layer.name: # Find the main Encoder block
                  transformer_encoder_block = layer
                  break
        
        if transformer_encoder_block:
            # Freeze all layers first
            for layer in transformer_encoder_block.layers:
                layer.trainable = False
            # Then, unfreeze the last N
            for layer in transformer_encoder_block.layers[-unfreeze_layers:]:
                layer.trainable = True
                print(f"  - Unfreezing layer: {layer.name}")
        else:
            print("Warning: Could not find main 'encoder' block to partially unfreeze.")

    # --- Build the new model ---
    # Get the inputs from the pre-trained encoder model
    inputs = encoder_model.input

    # Get the output of the pre-trained encoder (the embeddings)
    x = encoder_model.output

    # Add a new classification head
    x = layers.Dropout(0.2)(x) # Add dropout for regularization
    outputs = layers.Dense(num_classes, name='classifier_head')(x) # Output logits

    # Create the final model
    finetune_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ASTRA_Classifier")

    return finetune_model