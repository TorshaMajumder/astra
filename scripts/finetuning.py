import os
import tensorflow as tf

# --- Main Fine-tuning Script ---

# 1. SETUP AND CONFIGURATION
# --- Paths and HParams ---
run_directory = "/path/to/your/pre-trained/run_YYYYMMDD_HHMMSS/"
path_to_labeled_data = "/path/to/your/labeled_data_tfrecords/"
path_to_save_finetuned_model = "/path/to/save/finetuned_models/"

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
model_params, _, _ = load_hparams_from_event_file(run_directory)
# ... (create the encoder_model and load its weights) ...
# ... (ensure it's built with the correct fixed maxlen)
# For this example, let's assume `encoder_model` is loaded and ready.

# --- 3. CREATE THE FINE-TUNING MODEL ---
finetune_model = create_finetuning_model(
    encoder_model=encoder_model,
    num_classes=NUM_CLASSES,
    unfreeze_layers=UNFREEZE_LAYERS
)
finetune_model.summary()

# --- 4. PREPARE DATA LOADERS ---
# Use the fixed sequence length the encoder was built with
INFERENCE_MAXLEN = encoder_model.input_shape[0][1] # Get maxlen from model e.g., (None, 300, 1) -> 300

train_loader = create_finetune_loader(
    source_dir=path_to_labeled_data,
    batch_size=BATCH_SIZE,
    label_map=LABEL_MAP,
    maxlen=INFERENCE_MAXLEN,
    fraction_to_use=FRACTION,
    is_training=True
)
# Create a validation loader using a different (or the same) labeled set
# Use is_training=False to use the whole set and disable shuffling
val_loader = create_finetune_loader(
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