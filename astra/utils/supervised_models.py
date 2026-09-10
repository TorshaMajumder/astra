import os
import yaml
import psutil
import datetime
import tensorflow as tf
from astra.src.finetuning import finetune_data_loader

def build_cnn_baseline(seq_len=600, num_classes=12, band_emb_dim=16):
    """
    Builds a 4-Layer 1D-CNN baseline model.
    """
    input_mag = tf.keras.Input(shape=(seq_len, 1), name='input', dtype=tf.float32)
    input_time = tf.keras.Input(shape=(seq_len, 1), name='times', dtype=tf.float32)
    input_band = tf.keras.Input(shape=(seq_len,), name='band_info', dtype=tf.int32)
    input_mask = tf.keras.Input(shape=(seq_len,), name='mask', dtype=tf.float32) 
    
    input_dict = {'input': input_mag, 'times': input_time, 'band_info': input_band, 'mask': input_mask}

    band_embedded = tf.keras.layers.Embedding(input_dim=3, output_dim=band_emb_dim, name="band_emb")(input_band)
    x = tf.keras.layers.Concatenate(axis=-1)([input_mag, input_time, band_embedded])
    
    valid_mask = tf.expand_dims(1.0 - input_mask, axis=-1)
    x = tf.keras.layers.Multiply()([x, valid_mask])

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    embedding = tf.keras.layers.GlobalAveragePooling1D(name="cnn_embedding")(x)
    x_drop = tf.keras.layers.Dropout(0.3)(embedding)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="classifier")(x_drop)

    full_model = tf.keras.Model(inputs=input_dict, outputs=outputs, name="1D_CNN_Classifier")
    extractor_model = tf.keras.Model(inputs=input_dict, outputs=embedding, name="1D_CNN_Extractor")

    return full_model, extractor_model


def build_bilstm_baseline(seq_len=600, num_classes=12, band_emb_dim=16):
    """
    Builds a 2-Layer Bi-LSTM baseline model.
    """
    input_mag = tf.keras.Input(shape=(seq_len, 1), name='input', dtype=tf.float32)
    input_time = tf.keras.Input(shape=(seq_len, 1), name='times', dtype=tf.float32)
    input_band = tf.keras.Input(shape=(seq_len,), name='band_info', dtype=tf.int32)
    input_mask = tf.keras.Input(shape=(seq_len,), name='mask', dtype=tf.float32) 
    
    input_dict = {'input': input_mag, 'times': input_time, 'band_info': input_band, 'mask': input_mask}

    band_embedded = tf.keras.layers.Embedding(input_dim=3, output_dim=band_emb_dim, name="band_emb")(input_band)
    fused_inputs = tf.keras.layers.Concatenate(axis=-1)([input_mag, input_time, band_embedded])
    
    bool_mask = tf.keras.layers.Lambda(lambda m: tf.logical_not(tf.cast(m, tf.bool)))(input_mask)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(fused_inputs, mask=bool_mask)
    
    x = tf.keras.layers.Dropout(0.3)(x)
    
    embedding = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=False), 
        name="lstm_embedding"
    )(x, mask=bool_mask)

    x_drop = tf.keras.layers.Dropout(0.3)(embedding)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="classifier")(x_drop)

    full_model = tf.keras.Model(inputs=input_dict, outputs=outputs, name="BiLSTM_Classifier")
    extractor_model = tf.keras.Model(inputs=input_dict, outputs=embedding, name="BiLSTM_Extractor")

    return full_model, extractor_model


def supervise_backbone(config):
    # ===============================================
    # ------------- Device Strategy Setup -----------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if config.get('num_gpus') is not None and config['num_gpus'] > 0:
        if config['num_gpus'] > len(gpus):
            print(f"\nWarning: Requested {config['num_gpus']} GPUs, but only {len(gpus)} are available.\n")
            gpus_to_use = gpus
        else:
            gpus_to_use = gpus[:config['num_gpus']]
        tf.config.experimental.set_visible_devices(gpus_to_use, 'GPU')
        print(f"\nUsing {len(gpus_to_use)} specified GPU(s).\n")
    else:
        print("\nNo GPUs found. Running in CPU mode.\n")
        tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count(logical=False))
        tf.config.threading.set_inter_op_parallelism_threads(0)

    # ===============================================
    # --- Setup Parameters & Directories ---
    num_classes = len(config['label_map'])
    
    # Calculate build_seq_len (Sum of your global view max lens, or just 600)
    build_seq_len = sum(config['global_view_maxlens'].values()) if isinstance(config['global_view_maxlens'], dict) else config['global_view_maxlens']

    # Select Model Type
    model_type = config.get('model_type', 'cnn') # Defaults to 'cnn', can be 'bilstm'
    
    finetune_dir = os.path.join(config['path_to_save'], f"supervised_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(finetune_dir, exist_ok=True)
    print(f"\n'{finetune_dir}' created for logging.\n")

    # ===============================================
    # --- Build the chosen model ---
    if model_type == 'cnn':
        full_model, extractor = build_cnn_baseline(seq_len=build_seq_len, num_classes=num_classes)
    elif model_type == 'bilstm':
        full_model, extractor = build_bilstm_baseline(seq_len=build_seq_len, num_classes=num_classes)
    else:
        raise ValueError("model_type must be either 'cnn' or 'bilstm'")
        
    print(f"\n -- {model_type.upper()} model built successfully!")

    # Compile the model (Using Sparse Categorical Crossentropy since labels are integers)
    learning_rate = config.get('lr', 1e-3)
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    full_model.summary()

    # ===============================================
    # --- Prepare Data Loaders ---
    print("\nSetting up the data loaders...")
    train_loader = finetune_data_loader(
        source_dir=config['path_to_data'],
        batch_size=config['batch_size'],
        label_map=config['label_map'],
        max_len=config['global_view_maxlens'],
        buffer_size=config.get('buffer_size', 10000),
        is_training=True,
        apply_white_noise=True
    )
    
    val_loader = finetune_data_loader(
        source_dir=config['path_to_val'], 
        batch_size=config['batch_size'],
        label_map=config['label_map'],
        max_len=config['global_view_maxlens'],
        is_training=False,
        apply_white_noise=False 
    )

    # ---------------------------------------------------------------------
    # FIX: The dataloader outputs (batch, 3, seq_len, ...).
    # We only want ONE view for the baselines. We slice out the first view.
    def slice_single_view(features, labels):
        single_view_features = {
            'input': features['input'][:, 0, ...],
            'times': features['times'][:, 0, ...],
            'band_info': features['band_info'][:, 0, ...],
            'mask': features['mask'][:, 0, ...]
        }
        # If the mask has a trailing 1 e.g. (batch, seq_len, 1), squeeze it
        if len(single_view_features['mask'].shape) == 3:
            single_view_features['mask'] = tf.squeeze(single_view_features['mask'], axis=-1)
            
        return single_view_features, labels

    train_loader = train_loader.map(slice_single_view, num_parallel_calls=tf.data.AUTOTUNE)
    val_loader = val_loader.map(slice_single_view, num_parallel_calls=tf.data.AUTOTUNE)
    # ---------------------------------------------------------------------

    # ===============================================
    # --- Callbacks ---
    checkpoint_path = os.path.join(finetune_dir, f"best_supervised_{model_type}_weights.h5")
    
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
        patience=config.get('patience', 15),
        mode='max',
        verbose=1,
        restore_best_weights=True 
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=finetune_dir)

    # ===============================================
    # --- Start Training ---
    print(f"\n{'='*20} Starting Baseline Training {'='*20}\n")
    
    _ = full_model.fit(
        train_loader,
        epochs=config.get('epochs', 50),
        validation_data=val_loader,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback]
    )

    print(f"\n--- Supervised {model_type.upper()} training complete!")
    print(f"The best model weights are saved at: {checkpoint_path}")


def main(path_to_config=None, mode=None):
    with open(path_to_config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"\nError parsing YAML file: {e}")
            
    if mode == "training":
        # Make sure to set model_type in your YAML config or hardcode it here
        if 'model_type' not in config:
            config['model_type'] = 'cnn' # change to 'bilstm' to run the other
            
        supervise_backbone(config)
    else:
        raise ValueError(f"\nInvalid mode '{mode}'. Choose 'training'.")

if __name__ == '__main__':
    path_to_config = "/Users/torshamajumder/git/astra/config/supervised_task.yaml"
    mode = "training" 
    main(path_to_config, mode)