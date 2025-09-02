import tensorflow as tf
import os
import numpy as np

# Make sure AUTO = tf.data.AUTOTUNE is defined globally
AUTO = tf.data.AUTOTUNE

# Ensure your other data functions are defined:
# deserialize, standardize, get_window (if using fixed length)

def create_finetune_loader(source_dir,
                           batch_size,
                           label_map, # A dict mapping string labels to integer IDs
                           maxlen, # The fixed sequence length the encoder expects
                           fraction_to_use=0.01, # The fraction of data to use (e.g., 1%)
                           is_training=True): # Flag to enable shuffling and taking a fraction
    """
    Creates a tf.data.Dataset for fine-tuning or evaluating a classifier.
    """
    num_classes = len(label_map)
    # Create a lookup table to convert string labels to integer IDs
    keys_tensor = tf.constant(list(label_map.keys()))
    vals_tensor = tf.constant(list(label_map.values()))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1 # Default value for unknown labels
    )

    glob_pattern = os.path.join(source_dir, 'partition_*', '*', 'chunk_*.record')
    filenames_dataset = tf.data.Dataset.list_files(glob_pattern, shuffle=is_training)

    # --- Get the total number of samples to calculate the 1% subset ---
    if is_training and fraction_to_use < 1.0:
        # This is a bit slow but necessary to get an accurate count for .take()
        # For very large datasets, you might pre-calculate this count.
        print("Counting total samples to determine subset size...")
        total_samples = 0
        for fn in filenames_dataset:
            total_samples += sum(1 for _ in tf.data.TFRecordDataset(fn))
        
        subset_size = int(total_samples * fraction_to_use)
        print(f"Using {fraction_to_use*100:.1f}% of data: {subset_size} out of {total_samples} samples.")
        # Shuffle the filenames and then take the subset
        dataset = filenames_dataset.interleave(
            tf.data.TFRecordDataset, cycle_length=AUTO, num_parallel_calls=AUTO
        ).take(subset_size)
    else:
        # For validation or using the full dataset
        dataset = filenames_dataset.interleave(
            tf.data.TFRecordDataset, cycle_length=AUTO, num_parallel_calls=AUTO
        )

    def preprocess_and_map_label(data):
        # (This part is identical to your fixed-length inference preprocessor)
        input_dict = deserialize(data)
        # ... (standardize magnitude, reconstruct features) ...
        features = input_dict['input_features']
        mags, magerrs = features[:, 1], features[:, 2]
        new_mag, _ = standardize(mags, magerrs)
        initial_mask = tf.zeros(tf.shape(new_mag)[0], dtype=tf.float32)
        if apply_white_noise:
            #
            new_mag = gaussian_noise(new_mag, noise_level)
        
        processed_features = tf.stack([features[:, 0], new_mag, features[:, 2], features[:, 3]], axis=1)
        
        num_cols = tf.shape(processed_features)[1]
        
        final_features, final_mask = get_window(processed_features, initial_mask, maxlen, num_cols)

        # Prepare model inputs dictionary
        model_inputs = {
            'input': tf.expand_dims(final_features[:, 1], axis=-1),
            'times': tf.expand_dims(final_features[:, 0], axis=-1),
            'band_info': tf.expand_dims(final_features[:, 3], axis=-1),
            'mask': final_mask
        }

        # Map the string label to an integer ID
        label_id = table.lookup(input_dict['label'])
        
        return model_inputs, label_id

    processed_dataset = dataset.map(preprocess_and_map_label, num_parallel_calls=AUTO)

    # Filter out any samples with unknown labels (ID = -1)
    processed_dataset = processed_dataset.filter(lambda inputs, label: label != -1)

    if is_training:
        # For fine-tuning, shuffle the small subset well
        processed_dataset = processed_dataset.shuffle(buffer_size=1000) # Buffer can be large for small subset

    final_loader = processed_dataset.batch(batch_size).prefetch(buffer_size=AUTO)
    
    return final_loader


def create_finetuning_model(encoder_model,
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