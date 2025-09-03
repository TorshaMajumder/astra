import os
import datetime # Import datetime
import tensorflow as tf
from astra.src.finetuning import finetune_data_loader, finetune_model
from astra.src.transformer import AstroTransformer
from astra.utils.helper import load_hparams_from_event_file

# --- Main Fine-tuning Script ---

# 1. SETUP AND CONFIGURATION
# --- Paths and HParams ---

run_directory = "/media3/majumder/contrastive_loss_res/run_20250826_222245/"
path_to_labeled_data = "/media3/majumder/dataset/lyrae_cep/train/"
# path_to_save_finetuned_model = "/path/to/save/finetuned_models/"

if run_directory:
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        finetune_dir = os.path.join(run_directory, f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(finetune_dir, exist_ok=True)
        print(f"\n\n {finetune_dir} is created.\n")


finetune_data_loader(
    source_dir=path_to_labeled_data
)
exit()
# Fine-tuning HParams
FINETUNE_LR = 1e-5 # CRITICAL: Use a very small learning rate
BATCH_SIZE = 32    # Can be smaller for fine-tuning
EPOCHS = 50
PATIENCE = 10
UNFREEZE_LAYERS = 2 # e.g., unfreeze the last 2 EncoderLayers
FRACTION = 0.01   # Use 1% of the data

# --- Define Label Mapping ---
# IMPORTANT: This must match your dataset's classes
LABEL_MAP = {
    'CEP': 0,
    'RRLY': 1,
}
NUM_CLASSES = len(LABEL_MAP)

# --- 2. LOAD PRE-TRAINED ENCODER ---
# (Use the code from the embedding generation script to load hparams
#  and create the `encoder_model`)
model_params, training_params, data_params = load_hparams_from_event_file(run_directory)
# Stop if hyperparameters could not be loaded
if model_params is None:
    raise ValueError("Failed to load hyperparameters from event file. Exiting.")
print("\n2. Re-creating the full AstroTransformer architecture using loaded HParams...")
model = AstroTransformer(**model_params)
print("   Model instantiated.")



# Path to your saved weights file
weights_path = "/media3/majumder/contrastive_loss_res/run_20250825_214016/best_contrastive.weights.h5" # <--- SET THIS PATH
print(f"\n2. Loading pre-trained weights from: {weights_path}")
try:
    model.load_weights(weights_path)
    print("   Weights loaded successfully into the model.")
except Exception as e:
    print(f"ERROR: Could not load weights. Ensure architecture matches exactly. Error: {e}")
    # exit() # Stop if weights can't be loaded


# We will create a new model that stops after the pooling layer.
print("\n3. Creating the encoder-only model for generating embeddings...")

# Define the inputs with variable sequence length
input_layer = {
    'input': tf.keras.Input(shape=(build_seq_len, 1), name='input', dtype=tf.float64),
    'times': tf.keras.Input(shape=(build_seq_len, 1), name='times', dtype=tf.float64),
    'band_info': tf.keras.Input(shape=(build_seq_len, 1), name='band_info', dtype=tf.float64),
    'mask': tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float64)
}

# --- Break down the forward pass step-by-step ---

# 1. Get the embeddings from the embedding layer.
#    The embedding layer takes the full dictionary of inputs.
embeddings = model.embedding_layer(input_layer)

# d_model, base=10000, rate=0.1, use_band_info=True, use_drop=False, mjd=True



# 2. Get the mask tensor from the input dictionary.
# mask_input = tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float64)
mask_input = input_layer['mask']

encoder_model = model.encoder(embeddings, mask=mask_input)

# --- 3. CREATE THE FINE-TUNING MODEL ---
finetune_model = finetune_model(
    encoder_model=encoder_model,
    num_classes=NUM_CLASSES,
    unfreeze_layers=UNFREEZE_LAYERS
)
finetune_model.summary()

# --- 4. PREPARE DATA LOADERS ---
# Use the fixed sequence length the encoder was built with
INFERENCE_MAXLEN = encoder_model.input_shape[0][1] # Get maxlen from model e.g., (None, 300, 1) -> 300

train_loader = finetune_data_loader(
    source_dir=path_to_labeled_data,
    batch_size=BATCH_SIZE,
    label_map=LABEL_MAP,
    maxlen=INFERENCE_MAXLEN,
    fraction_to_use=FRACTION,
    is_training=True
)
# Create a validation loader using a different (or the same) labeled set
# Use is_training=False to use the whole set and disable shuffling
val_loader = finetune_data_loader(
    source_dir="/path/to/your/labeled_val_data/", # Use a separate validation set
    batch_size=BATCH_SIZE,
    label_map=LABEL_MAP,
    maxlen=INFERENCE_MAXLEN,
    fraction_to_use=1.0, # Use 100% of validation data
    is_training=False
)

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
checkpoint_path = os.path.join(path_to_save_finetuned_model, "best_finetuned_model.weights.h5")
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
    verbose=1
)

print("\n--- Starting Fine-tuning ---")
history = finetune_model.fit(
    train_loader,
    epochs=EPOCHS,
    validation_data=val_loader,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

print("\n--- Fine-tuning complete! ---")
# The best model weights are saved at `checkpoint_path`