# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import h5py
import mlflow
import psutil
import logging
import argparse
import datetime 
import numpy as np
import tensorflow as tf
from astra.src.transformer import AstraNet
from astra.src.finetuning import finetune_data_loader, finetune_model
from astra.utils.helper import load_hparams_from_event_file, load_config
# ==========================================================
# CONFIGURE GPU MEMORY ALLOCATION
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print("\nGPUs are available. Setting memory growth to True.\n")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # This will happen if GPUs are already initialized.
        print(f"RuntimeError setting memory growth: {e}")
# ===========================================================
# SUPPRESS TF WARNINGS
logging.getLogger('tensorflow').setLevel(logging.ERROR)  
os.system('clear')
# ===========================================================

def finetune_model(config):
    # ===============================================
    # ------------- Device Strategy Setup -----------
    #
    # Detect available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #
    # Use user-specified GPUs. Otherwise, use all available GPUs.
    #
    if config['num_gpus'] is not None and config['num_gpus'] > 0:
        if config['num_gpus'] > len(gpus):
            print(f"\nWarning: Requested {config['num_gpus']} GPUs, but only {len(gpus)} are available. Using all available.\n")
            gpus_to_use = gpus
        else:
            gpus_to_use = gpus[:config['num_gpus']]
        #
        # Make only the selected GPUs visible to TensorFlow
        #
        tf.config.experimental.set_visible_devices(gpus_to_use, 'GPU')
        print(f"\nUsing {len(gpus_to_use)} specified GPU(s).\n")
    else:
        # If no GPUs are found, run on CPU.
        print("\nNo GPUs found. Running in CPU mode.\n")
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"\nAvailable CPU cores: Physical={physical_cores}, Logical={logical_cores}\n")
        # Set the number of threads for intra-operation parallelism
        num_intra_threads = 20
        tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)
        # Set the number of threads for inter-operation parallelism
        num_inter_threads = 0 # Let TensorFlow decide
        tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)
    # ====================================================================================================
    # ====================================================================================================
    # Load the hyper-parameters of the model from the path
    #
    run_directory = config['path_to_load']
    model_params, _, _ = load_hparams_from_event_file(run_directory)
    num_classes = len(config['label_map'])
    #
    # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
    #
    try: 
        if run_directory:
            finetune_dir = os.path.join(run_directory, f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(finetune_dir, exist_ok=True)
            print(f"\n'{finetune_dir}' is created.\n")
    except Exception as e:
        print(e)
        return
    #
    # Stop if hyperparameters could not be loaded
    #
    try:
        if model_params is None:
            raise ValueError("\n\nFailed to load hyperparameters from the event file.\nExiting...\n")
    except Exception as e:
        print(e)
        return
    #
    # --- Instantiate the Full Model using loaded hyper-params ---
    #
    print("\nRe-creating the full AstraNet architecture using loaded hyper-parameters...")
    model = AstraNet(
                    num_layers=model_params["num_layers"],
                    d_model=model_params["d_model"],
                    base=model_params["base"],
                    num_heads=model_params["num_heads"],
                    dff=model_params["dff"],
                    rate=model_params["rate"],
                    mjd=model_params["mjd"],
                    use_drop=model_params["use_drop"],
                    use_band_info=model_params["use_band_info"],
                    projection_dim=model_params["projection_dim"] 
                )
    print("\n --Model instantiated!")
    #
    # Building model with dummy input to create all variables
    # Using ANCHOR view sequence length, i.e., build_seg_len
    #
    build_seq_len = sum(config['max_len'].values()) 
    dummy_input = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'mask': tf.zeros((1, build_seq_len), dtype=tf.float32)
    }
    #
    # Set training=FALSE for inference
    #
    _ = model(dummy_input, training=False)
    print("\n --Full model built!")
    #
    # Load model's weight
    #
    try:
        path_to_weight = os.path.join(run_directory, 'best_contrastive.weights.h5') 
        print(f"\nSearching pre-trained weights in: {path_to_weight}...")
        model.load_weights(path_to_weight)
        print(f"\nWeights loaded successfully into the model!")
    except Exception as e:
        print(f"\nERROR: Could not load weights. Check the path to model's weight."
                    f"Ensure architecture matches exactly.\n{e}")
        return
    # ====================================================================================================
    # --- Isolate the ASTRA encoder to generate embeddings ---
    # --- Add the GlobalAveragePooling layer after ASTRA encoder ----
    # 
    print("\n --Extracting ASTRA encoder for generating embeddings...")
    #
    # Define the inputs with a fixed sequence length
    # It should match build_seq_len
    #
    input_layer = {
        'input': tf.keras.Input(shape=(build_seq_len, 1), name='input', dtype=tf.float32),
        'times': tf.keras.Input(shape=(build_seq_len, 1), name='times', dtype=tf.float32),
        'band_info': tf.keras.Input(shape=(build_seq_len, 1), name='band_info', dtype=tf.float32),
        'mask': tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float32)
    }
    # ------------------------------------------------------------------------------------------------
    #
    # (STEP:1) Get the embeddings from the embedding layer 
    # The embedding layer takes the full dictionary of inputs
    #
    embeddings = model.embedding_layer(input_layer)
    #
    # Get the mask tensor from the input dictionary (IMPORTANT for encoder and pooling laye)
    # 
    mask_input = input_layer['mask']
    #
    # (STEP:2) Get the embeddings and the attention weights
    #
    encoder_output, all_attention_weights = model.encoder(embeddings, mask=mask_input)
    #
    # (STEP:3) Invert the mask using ASTRA masking logic and get the pooled output
    #
    pool_mask = tf.keras.layers.Lambda(
                                        lambda m: tf.logical_not(tf.cast(m, tf.bool))
                                        )(mask_input)
    pooled_output = model.pooling(encoder_output, mask=pool_mask)
    #
    # (STEP:4) Get the final ASTRA encoder model 
    #
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=[pooled_output, all_attention_weights], name="ASTRA_Encoder")
    #
    # (STEP:5) Create the supervised finetuned ASTRA model 
    #
    finetune_model = finetune_model(encoder_model=encoder_model,
                                    num_classes=num_classes,
                                    unfreeze_layers=config['unfreeze_layers']
                                )
   
    print("\n -- Supervised Finetuned ASTRA model created successfully...!\n")
    finetune_model.summary()
    # =================================================================================================================
    #
    # ------------------ Prepare the Finetuning Data Loader -----------------------------------
    # 
    print("\nSetting up the inference data loader...")
    train_loader = finetune_data_loader(
                                        source_dir=config['path_to_data'],
                                        batch_size=config['batch_size'],
                                        label_map=config['label_map'],
                                        max_len=config['max_len'],
                                        buffer_size=config['buffer_size'],
                                        is_training=True,
                                        apply_white_noise=True
                                    )
    #                               
    # NOTE: validation data should be different from the training data
    # Use is_training=False to use the whole set and disable shuffling
    # Use 100% of validation data and No augmentation for validation
    #
    val_loader = finetune_data_loader(
                                        source_dir=config['path_to_val'], 
                                        batch_size=config['batch_size'],
                                        label_map=config['label_map'],
                                        maxlen=config['max_len'],
                                        is_training=False,
                                        apply_white_noise=False 
                                    )
    # ==================================================================================================================
    #
    # ------------------------------- Calculate steps_per_epoch ----------------------------------------
    # 
    n_samples = 114 
    steps_per_epoch = n_samples // config['batch_size']
    if n_samples % config['batch_size'] != 0:
        steps_per_epoch += 1 
    print(f"\n--- Calculated steps_per_epoch for fine-tuning: {steps_per_epoch}.\n")
    #
    # ------------------------------- Compile the model and train --------------------------------------
    # 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    #
    finetune_model.compile(
                            optimizer=optimizer,
                            loss=loss_fn,
                            metrics=metrics
                        )
    #
    # ---------------- Applying callbacks for saving the best model and early stopping -----------------
    checkpoint_path = os.path.join(finetune_dir, "best_finetuned_model.weights.h5")
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                monitor='val_sparse_categorical_accuracy', 
                                                                mode='max', 
                                                                save_best_only=True,
                                                                verbose=1
                                                            )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                                                                monitor='val_sparse_categorical_accuracy',
                                                                patience=config['patience'],
                                                                mode='max',
                                                                verbose=1,
                                                                restore_best_weights=True 
                                                            )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=finetune_dir)

    print(f"\n{'='*20} Starting Fine-tuning {'='*20}\n")
    
    _ = finetune_model.fit(
                            train_loader,
                            epochs=config['epochs'],
                            validation_data=val_loader,
                            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
                            steps_per_epoch=steps_per_epoch 
                        )

    print("\n--- Fine-tuning complete!")
    print(f"The best fine-tuned model weights are saved at: {checkpoint_path} .")
    
    # ==================================================================================================================


    


    




def main():
    # ==========================================================
    # Set up the Argument Parser
    # ==========================================================
    parser = argparse.ArgumentParser(prog='astra-finetuning',
                                        description="Supervised finetuning of Astra embeddings")
    # ==========================================================
    # Setup all required arguments
    # =========================================================
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    # ==========================================================
    args = parser.parse_args()
    # ==========================================================
    # --- Load Configuration ---
    config = load_config(args)
    finetune_model(config)
    
    
     






# Fine-tuning HParams
FINETUNE_LR = 1e-5 # CRITICAL: Use a very small learning rate
BATCH_SIZE = 32    # Can be smaller for fine-tuning
EPOCHS = 100
PATIENCE = 15
UNFREEZE_LAYERS = 2 # e.g., unfreeze the last 2 EncoderLayers
FRACTION = 0.01   # Use 1% of the data
NUM_CLASSES = len(LABEL_MAP)








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


if __name__ == '__main__':
    main()