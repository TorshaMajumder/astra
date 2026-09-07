# ===================================================================================
# Import all dependencies
# ===================================================================================
import os
import yaml
import h5py
import psutil
import logging
import argparse
import datetime 
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
from astra.src.preprocessing import create_inference_loader
from astra.src.transformer import AstraNet, AstraNet_Distil
from astra.src.finetuning import finetune_data_loader, finetune_model
from astra.utils.helper import load_hparams_from_event_file, load_config
# ====================================================================
# ====================================================================
tf.keras.backend.clear_session()
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


def finetuned_k_distil_embeddings(config):
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
    num_classes = len(config['label_map'])
    model_params, _, _ = load_hparams_from_event_file(run_directory)
    #
    # Stop if hyperparameters could not be loaded
    #
    try:
        if model_params is None:
            raise ValueError("\n\nFailed to load hyperparameters from the event file.\nExiting...\n")
    except Exception as e:
        print(e)
        return
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        #
        # --- Instantiate the Full Model using loaded hyper-params ---
        #
        print("\nRe-creating the full AstraNet architecture using loaded hyper-parameters...")
    
        supervised_backbone = AstraNet(
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
        _ = supervised_backbone(dummy_input, training=False) # Builds Global path
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
    # (STEP:1) Get the embeddings from the embedding layer 
    # The embedding layer takes the full dictionary of inputs
    #
    embeddings = supervised_backbone.embedding_layer(single_view_input)
    #
    # Get the mask tensor from the input dictionary (IMPORTANT for encoder and pooling laye)
    # 
    mask_input = single_view_input['mask']
    #
    # (STEP:2) Get the embeddings and the attention weights
    #
    encoder_output, all_attention_weights = supervised_backbone.encoder(embeddings, mask=mask_input)
    #
    # (STEP:3) Invert the mask using ASTRA masking logic and get the pooled output
    #
    pool_mask = tf.keras.layers.Lambda(
                                        lambda m: tf.logical_not(tf.cast(m, tf.bool))
                                        )(mask_input)
    pooled_output = supervised_backbone.pooling(encoder_output, mask=pool_mask)
    #
    # (STEP:4) Get the final ASTRA encoder model and Set to inference mode
    #
    single_view_encoder = tf.keras.Model(inputs=single_view_input, outputs=pooled_output, name="ASTRA_Encoder")
    # =================================================================================================================
    #
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
    # ----------------------------------------------------------------------------------------------------------
    #
    # --- (STEP:6) Aggregate the embeddings from all views by CONCATENATING ---
    #
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
    print("\n -- Supervised ASTRA model created successfully...!\n")
    #
    # Load model's weight
    #
    try:
        path_to_weight = os.path.join(run_directory, 'best_supervised_model_weights') 
        print(f"\nSearching finetuned weights in: {path_to_weight}...")
        supervised_model.load_weights(path_to_weight)
        print(f"\nWeights loaded successfully into the model!")
    except Exception as e:
        print(f"\nERROR: Could not load weights. Check the path to model's weight."
                    f"Ensure architecture matches exactly.\n{e}")
        return
    # ====================================================================================================
    # (STEP:8) Create the final embedding extractor from the finetuned model 
    # ====================================================================================================
    print("\n --Creating the final embedding extractor from the supervised model...")
    # Using the same 'input_layer' and the 'aggregated_embedding' tensor 
    # we calculated before the head was added, we can create the final embedding model (encoder)
    embedding_model = tf.keras.Model(inputs=input_layer, outputs=aggregated_embedding, name="Supervised_Embedding_Extractor")
    embedding_model.trainable = False 
    #
    print("\n --Final Embedding Extractor created successfully...!\n")
    embedding_model.summary()
    # =====================================================================================================   
    # =====================================================================================================   
    #
    # ------------------ Prepare the Inference Data Loader -----------------------------------
    # 
    print("\nSetting up the inference data loader...")
    inference_loader = create_inference_loader(
                                                source=config['path_to_data'],
                                                batch_size=config['batch_size'],
                                                maxlen=config['global_view_maxlens']
                                            )

    # ------------------------ Generate Finetuned ASTRA Embeddings ------------------------------------
    print("\nGenerating embeddings for the dataset...\n")
    # ------------------------- Get embedding dimension from the model --------------------------------
    # NOTE: the embedding_model outputs the concatenated embeddings from all views unlike 
    # the single_view_encoder model
    #
    num_views = 3  # Fixed number of views (start, mid, end)
    flattened_embedding_dim = embedding_model.output.shape[-1]  # e.g., 512 * 3 = 1536
    #
    os.makedirs(config['path_to_save'], exist_ok=True)
    h5_path = os.path.join(config['path_to_save'], 'embeddings.h5')
    print(f"\nStreaming embeddings directly to HDF5 file: {h5_path} .")
    #
    # ------------------------- Create the HDF5 file and resizable datasets --------------------------
    try:
        with h5py.File(h5_path, 'w') as hf:
            # 
            string_dtype = h5py.string_dtype(encoding='utf-8')
            dset_ids = hf.create_dataset('ids', (0,), maxshape=(None,), dtype='int64')
            dset_labels = hf.create_dataset('labels', (0,), maxshape=(None,), dtype=string_dtype)
            dset_embeddings = hf.create_dataset('embeddings', (0, flattened_embedding_dim), maxshape=(None, flattened_embedding_dim), dtype='float32')
            #
            num_rows_written = 0
            #
            # Iterate through the inference loader
            #
            for batch in tqdm(inference_loader, desc="Generating Supervised Embeddings"):
                #
                model_inputs = {
                                    'input': batch['input'],
                                    'times': batch['times'],
                                    'band_info': batch['band_info'],
                                    'mask': batch['mask']
                                }
                curr_batch_size = tf.shape(batch['input'])[0]
                # the embedding model directly processes the multi-view inputs 
                # and outputs the concatenated embeddings
                final_embeddings_batch = embedding_model(model_inputs, training=False)
                # Resize the datasets on disk to make space for the new batch
                dset_embeddings.resize((num_rows_written + curr_batch_size, flattened_embedding_dim))
                dset_labels.resize((num_rows_written + curr_batch_size,))
                dset_ids.resize((num_rows_written + curr_batch_size,))
                # Write the new data into the newly created space
                dset_embeddings[num_rows_written:] = final_embeddings_batch.numpy()
                labels_as_bytes = batch['label'].numpy().astype(np.bytes_)
                dset_labels[num_rows_written:] = labels_as_bytes
                dset_ids[num_rows_written:] = batch['id'].numpy()
                # Update the row counter
                num_rows_written += curr_batch_size
                

        print(f"\n-- Generation Complete !")
        print(f"\nSuccessfully wrote {num_rows_written} embeddings to {h5_path} .")
    except Exception as e:
        print(f"\nERROR: Could not save the files. Check: {e}\n")
    # -------------------------------------------------------------------------------------------------
    
    



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
    #
    print("\nRe-creating the full AstraNet_backbone architecture using hyper-parameters...")
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



def main(path_to_config=None, mode=None):
    # ==========================================================
    # --- Load Configuration ---
    with open(path_to_config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"\nError parsing YAML file: {e}")
    # ----------------------------------------------------------
    if mode == "training":
        supervise_backbone(config)
    elif mode == "inference":
        finetuned_k_distil_embeddings(config)
    else:
        raise ValueError(f"\nInvalid mode '{mode}'. Choose either 'training' or 'inference'.")
    



if __name__ == '__main__':
    # Change config file path based on modes
    path_to_config = "/kaggle/working/astra/config/supervised_task.yaml"
    mode = "training"  # Change to "training" for supervised training
    main(path_to_config, mode)