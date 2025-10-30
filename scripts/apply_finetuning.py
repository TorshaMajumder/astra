import os
import datetime # Import datetime
import tensorflow as tf
import numpy as np
from astra.src.finetuning import finetune_data_loader, finetune_model
from astra.src.transformer import AstraNet
from astra.utils.helper import load_hparams_from_event_file

# --- Main Fine-tuning Script ---

# 1. SETUP AND CONFIGURATION
# --- Paths and HParams ---

run_directory = "/media3/majumder/contrastive_loss_res/run_20250826_222245/"
path_to_labeled_data = "/media3/majumder/dataset/lyrae_cep/train/"
# path_to_save_finetuned_model = "/path/to/save/finetuned_models/"

# IMPORTANT: This must match your dataset's classes
LABEL_MAP = {
    'CEP': 0,
    'RRLY': 1,
}
maxlen={'g': 300, 'r': 300, 'i': 300}

build_seq_len = sum(maxlen.values())

if run_directory:
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        finetune_dir = os.path.join(run_directory, f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(finetune_dir, exist_ok=True)
        print(f"\n\n {finetune_dir} is created.\n")
 


# Fine-tuning HParams
FINETUNE_LR = 1e-5 # CRITICAL: Use a very small learning rate
BATCH_SIZE = 32    # Can be smaller for fine-tuning
EPOCHS = 100
PATIENCE = 15
UNFREEZE_LAYERS = 2 # e.g., unfreeze the last 2 EncoderLayers
FRACTION = 0.01   # Use 1% of the data
NUM_CLASSES = len(LABEL_MAP)

# --- 2. LOAD PRE-TRAINED ENCODER ---
# (Use the code from the embedding generation script to load hparams
#  and create the `encoder_model`)
model_params, training_params, data_params = load_hparams_from_event_file(run_directory)
# Stop if hyperparameters could not be loaded
if model_params is None:
    raise ValueError("Failed to load hyperparameters from event file. Exiting.")
print("\n2. Re-creating the full AstraNet architecture using loaded HParams...")
model = AstraNet(**model_params)
print("   Model instantiated.")

# Build the model with a dummy input (use any fixed integer length)
_ = model({
    'input': tf.zeros((1, 300, 1)), 'times': tf.zeros((1, 300, 1)),
    'band_info': tf.zeros((1, 300, 1)), 'mask': tf.zeros((1, 300))
}, training=False)

# Path to your saved weights file
weights_path = "/media3/majumder/contrastive_loss_res/run_20250825_214016/best_contrastive.weights.h5" # <--- SET THIS PATH
print(f"\n2. Loading pre-trained weights from: {weights_path}")
try:
    model.load_weights(weights_path)
    print("   Weights loaded successfully into the model.")
except Exception as e:
    print(f"ERROR: Could not load weights. Ensure architecture matches exactly. Error: {e}")
    # exit() # Stop if weights can't be loaded


# --- Isolate the encoder part into a new, clean Keras model ---
print("Extracting the pre-trained encoder...")

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
# 2. Get the mask tensor from the input dictionary.
mask_input = input_layer['mask']
encoder_output = model.encoder(embeddings, mask=mask_input)
pool_mask = tf.keras.layers.Lambda(lambda m: tf.logical_not(tf.cast(m, tf.bool)))(mask_input)
pooled_output = model.pooling(encoder_output, mask=pool_mask)

# This is the final, standalone, pre-trained encoder model
encoder_model = tf.keras.Model(inputs=input_layer, outputs=pooled_output, name="ASTRA_Encoder")

# --- 3. CREATE THE FINE-TUNING MODEL ---
finetune_model = finetune_model(
    encoder_model=encoder_model,
    num_classes=NUM_CLASSES,
    unfreeze_layers=UNFREEZE_LAYERS
)
finetune_model.summary()

# --- 4. PREPARE DATA LOADERS ---

train_loader = finetune_data_loader(
    source_dir=path_to_labeled_data,
    batch_size=BATCH_SIZE,
    label_map=LABEL_MAP,
    maxlen=maxlen,
    fraction_to_use=FRACTION,
    is_training=True
)
# Create a validation loader using a different (or the same) labeled set
# Use is_training=False to use the whole set and disable shuffling
val_loader = finetune_data_loader(
    source_dir="/media3/majumder/dataset/lyrae_cep/val/", # Use a separate validation set
    batch_size=BATCH_SIZE,
    label_map=LABEL_MAP,
    maxlen=maxlen,
    fraction_to_use=1.0, # Use 100% of validation data
    is_training=False,
    apply_white_noise=False # No augmentation for validation
)

# --- Calculate steps_per_epoch ---
# You need the size of your 1% subset, which the loader prints out.
# Let's assume the loader prints: "Using 1.0% of candidates for training: 125 samples."
num_finetune_samples = 114 # <--- Get this number from the loader's print output
steps_per_epoch = num_finetune_samples // BATCH_SIZE
if num_finetune_samples % BATCH_SIZE != 0:
    steps_per_epoch += 1 # Add one step for the remainder batch
print(f"Calculated steps_per_epoch for fine-tuning: {steps_per_epoch}")

# --- 5. COMPILE THE MODEL AND TRAIN ---
# Use a standard classification loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

finetune_model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)

# Use Callbacks for saving the best model and early stopping
checkpoint_path = os.path.join(finetune_dir, "best_finetuned_model.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy', # Monitor validation accuracy
    mode='max', # Save the model with the highest accuracy
    save_best_only=True,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=PATIENCE,
    mode='max',
    verbose=1,
    restore_best_weights=True # Good practice for early stopping
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=finetune_dir)

print("\n--- Starting Fine-tuning ---")
history = finetune_model.fit(
    train_loader,
    epochs=EPOCHS,
    validation_data=val_loader,
    callbacks=[checkpoint_callback, early_stopping_callback],
    steps_per_epoch=steps_per_epoch # <--- ADD THIS ARGUMENT
)

print("\n--- Fine-tuning complete! ---")
print(f"The best fine-tuned model weights are saved at: {checkpoint_path}")
# The best model weights are saved at `checkpoint_path`