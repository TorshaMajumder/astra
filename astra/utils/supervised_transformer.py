# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import h5py
import psutil
import logging
import argparse
import datetime 
import numpy as np
import tensorflow as tf
from astra.src.transformer import AstraNet, AstraNet_Distil
from astra.src.finetuning import finetune_data_loader, finetune_model
from astra.utils.helper import load_hparams_from_event_file, load_config


def supervise_backbone(config):
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
    # model_params, _, _ = load_hparams_from_event_file(run_directory)
    num_classes = len(config['label_map'])
    #
    # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
    #
    try: 
        if run_directory:
            finetune_dir = os.path.join(config['path_to_save'], f"supervised_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(finetune_dir, exist_ok=True)
            print(f"\n'{finetune_dir}' is created.\n")
    except Exception as e:
        print(e)
        return
    # #
    # # Stop if hyperparameters could not be loaded
    # #
    # try:
    #     if model_params is None:
    #         raise ValueError("\n\nFailed to load hyperparameters from the event file.\nExiting...\n")
    # except Exception as e:
    #     print(e)
    #     return
    #
    # --- Instantiate the teacher model using loaded hyper-params ---
    #
    print("\nRe-creating the full AstraNet_Distil architecture using loaded hyper-parameters...")
    astra_backbone = AstraNet(
                    num_layers=config["num_layers"],
                    d_model=config["d_model"],
                    base=config["base"],
                    num_heads=config["num_heads"],
                    dff=config["dff"],
                    rate=config["rate"],
                    mjd=config["mjd"],
                    use_drop=config["use_drop"],
                    use_band_info=config["use_band_info"],
                    time_scaling=config["time_scaling"],
                    projection_dim=None,
                    name="supervised_backbone"  
                )
    print("\n --Model instantiated!")
    #
    # Building model with dummy input to create all variables
    #
    build_seq_len = sum(config['global_view_maxlens'].values()) 
    num_views = 3
    dummy_input = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'mask': tf.zeros((1, build_seq_len, ), dtype=tf.float32)
    }
    #
    # Set training=FALSE for inference
    #
    _ = astra_backbone(dummy_input, training=True) # Builds Global path
    astra_backbone.trainable = True
    print("\n --Full model built!")
    # ====================================================================================================
    # --- Isolate the ASTRA encoder to generate embeddings ---
    # --- Add the GlobalAveragePooling layer after ASTRA encoder ----
    # 
    print("\n --Extracting ASTRA encoder for generating embeddings...")
    #
    # Define the two input dict with a fixed sequence length
    # input layers for multi-view window inputs & single-view inputs for single-view window/sliding window
    #
    input_layer = {
        'input': tf.keras.Input(shape=(num_views, build_seq_len, 1), name='input', dtype=tf.float32),
        'times': tf.keras.Input(shape=(num_views, build_seq_len, 1), name='times', dtype=tf.float32),
        'band_info': tf.keras.Input(shape=(num_views, build_seq_len, 1), name='band_info', dtype=tf.float32),
        'mask': tf.keras.Input(shape=(num_views, build_seq_len, 1), name='mask', dtype=tf.float32) 
    }
    # It should match build_seq_len
    single_view_input = {
        'input': tf.keras.Input(shape=(build_seq_len, 1), name='sv_input'),
        'times': tf.keras.Input(shape=(build_seq_len, 1), name='sv_times'),
        'band_info': tf.keras.Input(shape=(build_seq_len, 1), name='sv_band_info'),
        'mask': tf.keras.Input(shape=(build_seq_len,), name='sv_mask')
    }
    # ------------------------------------------------------------------------------------------------
    #
    # NOTE: The ASTRA encoder takes single view inputs only. 
    # (STEP:1) Get the embeddings from the embedding layer 
    # The embedding layer takes the full dictionary of inputs
    #
    embeddings = astra_backbone.embedding_layer(single_view_input)
    #
    # Get the mask tensor from the input dictionary (IMPORTANT for encoder and pooling laye)
    # 
    mask_input = single_view_input['mask']
    #
    # (STEP:2) Get the embeddings and the attention weights
    #
    encoder_output, all_attention_weights = astra_backbone.encoder(embeddings, mask=mask_input)
    #
    # (STEP:3) Invert the mask using ASTRA masking logic and get the pooled output
    #
    pool_mask = tf.keras.layers.Lambda(
                                        lambda m: tf.logical_not(tf.cast(m, tf.bool))
                                        )(mask_input)
    pooled_output = astra_backbone.pooling(encoder_output, mask=pool_mask)
    #
    # (STEP:4) Get the final ASTRA encoder model 
    #
    single_view_encoder = tf.keras.Model(inputs=single_view_input, outputs=pooled_output, name="ASTRA_Encoder")
    # =========================================================================================================
    # =========================================================================================================
    # --- (STEP:5) Process each view through the ASTRA encoder ---
    #
    view_embeddings = []
    for i in range(num_views):
        # Slice the i-th view from the main inputs
        input_view_slice = tf.keras.layers.Lambda(lambda x: x[:, i], name=f'input_slice_{i}')(input_layer['input'])
        times_view_slice = tf.keras.layers.Lambda(lambda x: x[:, i], name=f'times_slice_{i}')(input_layer['times'])
        band_info_view_slice = tf.keras.layers.Lambda(lambda x: x[:, i], name=f'band_info_slice_{i}')(input_layer['band_info'])
        # Slice AND Reshape the Mask
        mask_view_slice = tf.keras.layers.Lambda(lambda x: x[:, i, :, 0], name=f'mask_slice_{i}')(input_layer['mask'])
        # Create the input dictionary for this single view
        current_view_input_dict = {
            'input': input_view_slice,
            'times': times_view_slice,
            'band_info': band_info_view_slice,
            'mask': mask_view_slice # shape is (Batch, Seq_Len)
        }    
        # Get the embedding for each view
        view_embedding = single_view_encoder(current_view_input_dict)
        view_embeddings.append(view_embedding)
    # --- (STEP:6) Aggregate the embeddings from all views by CONCATENATING ---
    if len(view_embeddings) > 1:
        # Concatenate along the last axis (the feature dimension)
        # Input: A list of 4 tensors, each of shape (batch_size, 512)
        # Output: A single tensor of shape (batch_size, 4 * 512) -> (batch_size, 2048)
        aggregated_embedding = tf.keras.layers.Concatenate(axis=-1, name='aggregate_embeddings')(view_embeddings)
    else:
        aggregated_embedding = view_embeddings[0]
    #
    # (STEP:7) Create the supervised finetuned ASTRA model 
    #
    supervised_model = finetune_model(encoder_model=single_view_encoder,
                                        num_classes=num_classes,
                                        final_inputs=input_layer,         
                                        aggregated_embedding=aggregated_embedding,
                                        unfreeze_layers=config['unfreeze_layers']
                                )
    print("\n -- Supervised Finetuned ASTRA model created successfully...!\n")
    supervised_model.summary()
    # =================================================================================================================
    #
    # ------------------ Prepare the Finetuning Data Loader -----------------------------------
    # 
    print("\nSetting up the inference data loader...")
    train_loader = finetune_data_loader(
                                        source_dir=config['path_to_data'],
                                        batch_size=config['batch_size'],
                                        label_map=config['label_map'],
                                        max_len=config['global_view_maxlens'],
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
                                        max_len=config['global_view_maxlens'],
                                        is_training=False,
                                        apply_white_noise=False 
                                    )
    # ==================================================================================================================
    #
    # ------------------------------- Compile the model and train --------------------------------------
    # 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    #
    supervised_model.compile(
                            optimizer=optimizer,
                            loss=loss_fn,
                            metrics=metrics
                        )
    #
    # ---------------- Applying callbacks for saving the best model and early stopping -----------------
    checkpoint_path = os.path.join(finetune_dir, "best_supervised_model_weights")
    # Change `best_finetuned_student_model_weights` when checking student model.
    
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
    
    _ = supervised_model.fit(
                            train_loader,
                            epochs=config['epochs'],
                            validation_data=val_loader,
                            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback]
                        )

    print("\n--- Supervised training complete!")
    print(f"The best model weights are saved at: {checkpoint_path} .")
    
    # ==================================================================================================================



def main(path_to_config=None):
    # ==========================================================
    # --- Load Configuration ---
    config = load_config(path_to_config)
    # ----------------------------------------------------------
    supervise_backbone(config)
    



if __name__ == '__main__':
    path_to_config = None  # Default to None, which will use the default config path
    main(path_to_config)