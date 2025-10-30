import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tqdm import tqdm # Or standard tqdm
import pandas as pd
from astra.src.transformer import AstraNet
from astra.src.preprocessing import create_inference_loader
from astra.utils.helper import load_hparams_from_event_file
from astra.src.embedding import AstraEmbedding
from astra.src.encoder import Encoder


run_directory = "/media3/majumder/contrastive_loss_res/run_20250826_222245/" # <--- SET THIS PATH

model_params, training_params, data_params = load_hparams_from_event_file(run_directory)

# Stop if hyperparameters could not be loaded
if model_params is None:
    raise ValueError("Failed to load hyperparameters from event file. Exiting.")

# --- Step 1: Re-define your Model and all its custom sub-layers ---
# You MUST have the exact definitions of all the classes used to build the model.
# Paste the final, correct versions of all these classes here:
# - AstraEmbedding
# - EncoderLayer
# - Encoder
# - ProjectionHead (though we won't use it, it's needed to build the full model first)
# - AstraNet
# - All helper functions like standardize, sliding_window/get_window, deserialize.

# Example placeholder (REPLACE WITH YOUR ACTUAL CLASS DEFINITIONS)
# from your_module import AstraNet, AstraEmbedding, ...
# from your_data_utils import create_inference_loader, ...

# --- Step 2: Define the EXACT Hyperparameters of the Saved Model ---
# It is CRUCIAL that these match the run that generated the .h5 file.
# You can get these from your saved hyperparameter log.
# d_model = 256
# num_layers = 4
# num_heads = 4
# dff = 1024
# projection_dim = 128
# rate = 0.1
# use_band_info = True
# use_embedding_dropout = True
# --- 2. Instantiate the Full Model using Loaded HParams ---
print("\n2. Re-creating the full AstraNet architecture using loaded HParams...")
model = AstraNet(**model_params)
print("   Model instantiated.")


# Build the model with a dummy input to create all variables
# Use the sequence length that the ANCHOR view had during training
# Example: maxlens=(100, 50, 100), ztf_band has 3 bands
# build_seq_len = 100 * 3 = 300
max_len = {'g': 400, 'r': 500, 'i': 100} # <--- IMPORTANT: SET THIS TO YOUR ANCHOR'S MAXLENS
build_seq_len = sum(max_len.values()) # <--- IMPORTANT: SET THIS TO YOUR ANCHOR'S SEQ LENGTH
dummy_input = {
    'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
    'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
    'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
    'mask': tf.zeros((1, build_seq_len), dtype=tf.float32)
}
_ = model(dummy_input, training=False)
print("   Full model built.")

# Path to your saved weights file
weights_path = "/media3/majumder/contrastive_loss_res/run_20250826_222245/finetune_20250907_002342/best_finetuned_model.weights.h5" # <--- SET THIS PATH
print(f"\n2. Loading pre-trained weights from: {weights_path}")
try:
    model.load_weights(weights_path)
    print("   Weights loaded successfully into the model.")
except Exception as e:
    print(f"ERROR: Could not load weights. Ensure architecture matches exactly. Error: {e}")
    # exit() # Stop if weights can't be loaded

# --- Step 4: Isolate the Encoder to Create the Embedding Model ---
# The best embeddings are typically the output of the main encoder, *before* the projection head.
# We will create a new model that stops after the pooling layer.
print("\n3. Creating the encoder-only model for generating embeddings...")

# Define the inputs with variable sequence length
input_layer = {
    'input': tf.keras.Input(shape=(build_seq_len, 1), name='input', dtype=tf.float32),
    'times': tf.keras.Input(shape=(build_seq_len, 1), name='times', dtype=tf.float32),
    'band_info': tf.keras.Input(shape=(build_seq_len, 1), name='band_info', dtype=tf.float32),
    'mask': tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float32)
}

# --- Break down the forward pass step-by-step ---

# 1. Get the embeddings from the embedding layer.
#    The embedding layer takes the full dictionary of inputs.
embeddings = model.embedding_layer(input_layer)

# d_model, base=10000, rate=0.1, use_band_info=True, use_drop=False, mjd=True



# 2. Get the mask tensor from the input dictionary.
# mask_input = tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float64)
mask_input = input_layer['mask']


encoder_output = model.encoder(embeddings, mask=mask_input)
pool_mask = tf.keras.layers.Lambda(
    lambda m: tf.logical_not(tf.cast(m, tf.bool))
)(mask_input)
pooled_output = model.pooling(encoder_output, mask=pool_mask)


# 3. Call the encoder, providing both the embeddings AND the mask.
# encoder_output = model.encoder(embeddings, mask=mask_input)

# 4. Call the pooling layer, also providing the mask for correct averaging.
#    Remember, Keras pooling expects the mask to be True for elements to KEEP.
#    Your mask is 1.0 for elements to IGNORE, so we must invert it.
# pool_mask = tf.keras.layers.Lambda(
#     lambda m: tf.logical_not(tf.cast(m, tf.bool)),
#     name='invert_mask_for_pooling'
# )(mask_input)
# pooled_output = model.pooling(encoder_output, mask=pool_mask)

# --- Create the final model ---
# The inputs are the original dictionary of Input layers.
# The output is the final pooled_output tensor we just defined.
encoder_model = tf.keras.Model(inputs=input_layer, outputs=pooled_output, name="ASTRA_Encoder")
encoder_model.trainable = False # Set to inference mode

print("   Encoder model created successfully.")
encoder_model.summary()

# --- Step 5: Prepare the Inference Data Loader ---
# Use the `create_inference_loader` function we discussed previously.
# This loader should yield batches of dictionaries containing preprocessed data
# AND the metadata ('id', 'label').
print("\n4. Setting up the inference data loader...")

path_to_inference_data = "/media3/majumder/dataset/lyrae_cep/test/" # <--- SET THIS PATH

inference_batch_size = 300 # Can be larger than training batch size

# The maxlen here MUST match the build_seq_len used above
inference_loader = create_inference_loader(
    source=path_to_inference_data,
    batch_size=inference_batch_size,
    maxlen=max_len
)

# --- Step 6: Generate and Collect Embeddings ---
print("\n5. Generating embeddings for the dataset...")

all_embeddings = []
all_labels = []
all_ids = []

# Iterate through the inference loader
for batch_data in tqdm(inference_loader, desc="Generating Embeddings"):
    # The encoder model expects a dictionary of tensors
    # The loader already provides this format
    print(batch_data['input'].shape, batch_data['times'].shape, batch_data['band_info'].shape, batch_data['mask'].shape)
    
    model_inputs = {
        'input': batch_data['input'],
        'times': batch_data['times'],
        'band_info': batch_data['band_info'],
        'mask': batch_data['mask']
    }
    batch_embeddings = encoder_model(model_inputs, training=False)

    # Collect the results
    all_embeddings.append(batch_embeddings.numpy())
    all_labels.append(batch_data['label'].numpy())
    all_ids.append(batch_data['id'].numpy())

# Concatenate results from all batches into single numpy arrays
all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_ids = np.concatenate(all_ids, axis=0)

# Decode labels if they are byte strings
all_labels = np.array([label.decode('utf-8') for label in all_labels])

print(f"\n6. Successfully generated {len(all_embeddings)} embeddings!")
print(f"   Embeddings shape: {all_embeddings.shape}")
print(f"   Labels shape: {all_labels.shape}")
print(f"   IDs shape: {all_ids.shape}")

# --- Optional: Save the results ---
output_dir = os.path.dirname(weights_path) # Save in the same run directory
print(f"\n7. Saving embeddings and metadata to: {output_dir}")
np.save(os.path.join(output_dir, 'embeddings.npy'), all_embeddings)
np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
np.save(os.path.join(output_dir, 'ids.npy'), all_ids)
print("   Files saved: embeddings.npy, labels.npy, ids.npy")