# =========================================================
# Import all dependencies
# =========================================================
import os
import mlflow
import traceback
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import mlflow.tensorflow
from tensorflow.keras import layers
from astra.src.encoder   import Encoder
from astra.src.loss import nt_xent_loss 
from astra.src.header import ProjectionHead
from astra.src.embedding import AstraEmbedding
from astra.src.preprocessing import contrastive_data_loader


@tf.keras.utils.register_keras_serializable()
class AstraNet(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 base=10000.0, time_scaling=100, use_res=True, use_band_info=True, rate=0.1,
                 use_drop=False, mjd=True, projection_dim=None, name="astra_net", **kwargs):
        super(AstraNet, self).__init__(name=name, **kwargs)

        self.dff=dff
        self.mjd=mjd
        self.rate=rate
        self.base=base
        self.use_res=use_res
        self.d_model = d_model
        self.use_drop=use_drop
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.time_scaling=time_scaling
        self.use_band_info=use_band_info
        self.projection_dim=projection_dim
        #
        # Instantiate Embedding Layer
        #
        self.embedding_layer = AstraEmbedding(
                                                d_model=d_model, base=base, rate=rate, 
                                                use_band_info=use_band_info, use_drop=use_drop, mjd=mjd, time_scaling=time_scaling
                                            )
        #
        # Instantiate Encoder Layer
        #
        self.encoder = Encoder(
                                num_layers=num_layers, d_model=d_model, 
                                num_heads=num_heads, dff=dff, rate=rate, use_res=use_res 
                            )
        #
        # Instantiate Pooling Layer
        # Using Global Average Pooling as default
        #
        self.pooling = layers.GlobalAveragePooling1D(name='avg_pooling')
        #
        # Instantiate Projection Head Layer
        #
        self.projection_head_layer = ProjectionHead(
                                                        d_model=d_model, 
                                                        projection_dim=projection_dim
                                                    )



    
    def call(self, x, training=False):
        """
        Parameters:
        -----------------------------------------------------------------------
            x (dict): Input dictionary containing:
                'input': Magnitude tensor (batch, seq_len, 1)
                'times': Time tensor (batch, seq_len, 1)
                'band_info': Band tensor (batch, seq_len, 1) (optional)
                'mask': Mask tensor (batch, seq_len) or (batch, seq_len, 1)
            training (bool): Flag for training mode 

        Returns:
        -----------------------------------------------------------------------
            tf.Tensor: Final output tensor, usually after projection head.
                       Shape: (batch_size, projection_dim) or (batch_size, d_model) if no projection.
        """
        if not isinstance(x, dict) or not all(k in x for k in ['input', 'times', 'mask', 'band_info']):
             raise ValueError("\nInput 'x' must be a dictionary containing 'input', 'times', 'band_info' and 'mask'.\n")
        #
        # Ensure mask has the correct shape (batch, seq_len) for Encoder/Pooling
        # ensure last dim is squeezed
        mask = x['mask']       
        if len(mask.shape) == 3 and tf.shape(mask)[-1] == 1:
             mask = tf.squeeze(mask, axis=-1)
        elif len(mask.shape) != 2:
             raise ValueError(f"\nUnexpected mask shape in AstraNet: {tf.shape(mask)}. Expected (batch, seq_len).\n")
        # Apply Embedding Layer (takes the dictionary 'x')
        # Pass training flag for potential dropout in embedding
        embeddings = self.embedding_layer(x, training=training) # (batch, seq_len, d_model)
        # Apply Encoder (takes embeddings and mask)
        # Pass training flag for dropout/LN in encoder
        enc_output, all_attention_weights = self.encoder(embeddings, mask, training=training) # (batch, seq_len, d_model)
        # Apply Pooling
        # Invert mask for pooling (True where elements should be KEPT)
        pool_mask = tf.logical_not(tf.cast(mask, tf.bool))
        pooled_output = self.pooling(enc_output, mask=pool_mask) # (batch, d_model)
        # Apply Projection Head 
        # Pass training flag if projection head had dropout/BN 
        final_output = self.projection_head_layer(pooled_output, training=training) # (batch, projection_dim)
        #
        return final_output 

    def get_config(self):
        # Start with the base class's config.
        config = super(AstraNet, self).get_config()
        #
        # Update it with the custom arguments from __init__.
        #
        config.update({
                        "dff": self.dff,
                        "mjd": self.mjd,
                        "rate": self.rate,
                        "base": self.base,
                        "use_res": self.use_res,
                        "d_model": self.d_model,
                        "use_drop": self.use_drop,
                        "num_heads": self.num_heads,
                        "num_layers": self.num_layers,
                        "time_scaling": self.time_scaling
                        "use_band_info": self.use_band_info,
                        "projection_dim": self.projection_dim,
                    })
        return config

    @classmethod
    def from_config(cls, config):
        # This creates a new instance of the model from the config dictionary.
        return cls(**config)


# ==========================================================
# CONTRASTIVE TRAINING: DISTRIBUTED TRAINING STEP
# ==========================================================
@tf.function
def distributed_train_step(model, optimizer, strategy, global_batch_size, dist_inputs, temperature):
    """
    Performs one distributed training step.

    Parameters:
    -----------------------------------------------------------------------
        model (tf.keras.Model): The model to be trained.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.
        strategy (tf.distribute.Strategy): The distribution strategy.
        global_batch_size (int): The total batch size across all replicas.
        dist_inputs (tf.Tensor): The distributed input batch.
        temperature (float): Temperature parameter for NT-Xent loss.

    Returns:
    -----------------------------------------------------------------------
        tf.Tensor: The mean loss across all replicas for this training step.
    """
    def train_step(inputs):
        #
        # Unpack the views for the current replica's mini-batch
        #
        *views_batch, = inputs

        with tf.GradientTape() as tape:
            # Get projections from the model
            z_views = [model(view, training=True) for view in views_batch]
            # Calculate the loss for this replica's mini-batch
            loss = nt_xent_loss(*z_views, temperature=temperature)
            # IMPORTANT: Scale the loss by the GLOBAL batch size.
            # This ensures the gradients are correctly averaged, not summed.
            scaled_loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
        
        # Compute gradients
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return tf.reduce_mean(loss)
    
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


# ==========================================================
# CONTRASTIVE TRAINING: DISTRIBUTED VALIDATION STEP
# ==========================================================
@tf.function
def distributed_validation_step(model, strategy, dist_inputs, temperature):
    """
    Performs one distributed validation step.

    Parameters:
    -----------------------------------------------------------------------
        model (tf.keras.Model): The model to be trained.
        strategy (tf.distribute.Strategy): The distribution strategy.
        dist_inputs (tf.Tensor): The distributed input batch.
        temperature (float): Temperature parameter for NT-Xent loss.

    Returns:
    -----------------------------------------------------------------------
        tf.Tensor: The mean loss across all replicas for this validation step.
    """
    def valid_step(inputs):
        #
        # Unpack the views for the current replica's mini-batch
        #
        *views_batch, = inputs
        z_views = [model(view, training=False) for view in views_batch] 
        loss = nt_xent_loss(*z_views, temperature=temperature)
        return tf.reduce_mean(loss)
    
    per_replica_losses = strategy.run(valid_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    
# ==========================================================
# CONTRASTIVE TRAINING FUNCTION (MAIN)
# ==========================================================
def contrastive_train(model,
                            strategy,
                            optimizer,
                            build_seq_len=None,
                            path_to_read="",
                            path_to_val="",
                            path_to_save=None,
                            n_views=3,
                            global_batch_size=50,
                            temperature=0.1, 
                            patience=20,
                            epochs=10, 
                            initial_lr=1e-3, # For Adam optimizer if not using custom schedule
                            use_custom_schedule=True,
                            warmup_steps=4000,
                            apply_white_noise=(False, True, True),
                            noise_levels=(0.0, 0.1, 0.1), 
                            apply_binning=(False, False, True),
                            apply_outlier=(False, False, True),
                            maxlens=(200, 100, 200),
                            bin_widths=(5, 5, 5),
                            drop_rates=(0.0, 0.30, 0.60),
                            buffer_size=100, 
                        ):

    """
    Trains the model using contrastive learning with distributed strategy.
    
    Parameters:
    -----------------------------------------------------------------------
        model (tf.keras.Model): The model to be trained.
        strategy (tf.distribute.Strategy): The distribution strategy.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.
        build_seq_len (int): Sequence length for building the dataset.
                            NOTE: This is the maximum length required for padding.
        path_to_read (str): Path to the training dataset.
        path_to_val (str): Path to the validation dataset.
        path_to_save (str): Path to save the best model weights and logs.
        n_views (int): Number of augmented views per sample.
                        Use n_views=3 (Triplet arch.) or n_views=2 (Siamese arch.).
        global_batch_size (int): Total batch size across all replicas.
        temperature (float): Temperature parameter for NT-Xent loss.
        patience (int): Patience for early stopping.
        epochs (int): Maximum number of training epochs.
        initial_lr (float): Initial learning rate for the optimizer.
        use_custom_schedule (bool): Whether to use a custom learning rate schedule.
        warmup_steps (int): Number of warmup steps for custom learning rate schedule.
        drop_rates (tuple): Drop rates for each view.
        buffer_size (int): Buffer size for shuffling the dataset.

        AUGMENTATION PARAMETERS:
        -----------------------------------------------------------------------
        apply_white_noise (tuple): Flags to apply white noise to each view.
        noise_levels (tuple): Noise levels for each view, if apply_white_noise = TRUE.
        apply_binning (tuple): Flags to apply binning to each view.
        apply_outlier (tuple): Flags to apply outlier injection to each view.
        maxlens (tuple): Maximum lengths for each view.
        bin_widths (tuple): Bin widths for each view.

        Returns:
        -----------------------------------------------------------------------
            epoch_wise_train_loss (list): List of average training losses per epoch.
            epoch_wise_val_loss (list): List of average validation losses per epoch.
        
    """
    # ===================================================================================================================
    # ------------------------------------------- Setup Paths and TensorBoard Writer ------------------------------------
    #
    best_weights_path = None
    summary_writer = None
    #
    if path_to_save:
       #
       # Create TensorBoard writer
       # No logging if path_to_save is not provided
       #
        weights_filename = 'best_contrastive.weights.h5' 
        best_weights_path = os.path.join(path_to_save, weights_filename)
        summary_writer = tf.summary.create_file_writer(path_to_save)
        print(f"\n\nWill save best weights to: {best_weights_path}")

    else:
        summary_writer = None 
    #
    # ------------------------------------------ End Setup Paths and TensorBoard Writer ----------------------------------
    # ====================================================================================================================
    #
    # Create training data loader ONCE before the loop
    #
    print("\n\nSetting up data loader for training...")
    if not path_to_read or not os.path.exists(path_to_read):
         print("\n\nWarning: Training Dataset path not provided or does not exist. Please provide training data.")
         return None
    else:    
        try:
            train_loader = contrastive_data_loader(
                                                    source=path_to_read,
                                                    n_views=n_views, 
                                                    seed=np.random.randint(1024), 
                                                    batch_size=global_batch_size,
                                                    build_seq_len=build_seq_len,
                                                    apply_white_noise=apply_white_noise,
                                                    noise_levels=noise_levels,
                                                    apply_binning=apply_binning,
                                                    apply_outlier=apply_outlier,
                                                    maxlens=maxlens,
                                                    bin_widths=bin_widths,
                                                    drop_rates=drop_rates,
                                                    buffer_size=buffer_size
                                                )
            # ===============================================================================
            # ------------------ Distribute the datasets using the strategy -----------------
            #
            distributed_train_dataset = strategy.experimental_distribute_dataset(train_loader)
            # -------------------------------------------------------------------------------
            print("\n\nData loader ready.")
        except ValueError as e:
            print(f"\n\nError creating data loader: {e}")
            traceback.print_exc()
            return None # Return None if setup fails
    # ---------------------------- End of Training Data Loader ------------------------------
    # =======================================================================================
    #
    # Create validation data loader ONCE before the loop
    #
    print("\n\nSetting up data loader for validation...")
    valid_loader = None
    distributed_val_dataset = None
    if not path_to_val or not os.path.exists(path_to_val):
         print("\n\nWarning: Validation path not provided or does not exist. Skipping validation.")
    else:
        try:
            valid_loader = contrastive_data_loader(
                                                    source=path_to_val,
                                                    n_views=n_views, 
                                                    seed=np.random.randint(1024), 
                                                    batch_size=global_batch_size,
                                                    build_seq_len=build_seq_len,
                                                    apply_white_noise=apply_white_noise,
                                                    noise_levels=noise_levels,
                                                    apply_binning=apply_binning,
                                                    apply_outlier=apply_outlier,
                                                    maxlens=maxlens,
                                                    bin_widths=bin_widths,
                                                    drop_rates=drop_rates,
                                                    buffer_size=max(buffer_size // 4, 10) 
                                                )
            # ===============================================================================
            # ------------------- Distribute the datasets using the strategy ----------------
            #
            distributed_val_dataset = strategy.experimental_distribute_dataset(valid_loader)
            # -------------------------------------------------------------------------------
            print("\n\nValidation loader ready.")
        except ValueError as e:
            print(f"\n\nError creating validation loader: {e}")
            traceback.print_exc()
            valid_loader = None # Proceed without validation if loader fails
    # --------------------------- End of Validation Data Loader -----------------------------
    #
    # ----------------------------------------- END OF DATA LOADERS ---------------------------------------------
    #
    # ===========================================================================================================
    # 
    # ===============================================================================
    # --------------- All Variables ------------------
    es_count = 0
    global_step_train = 0 
    best_val_loss = np.inf # Initialize best loss to infinity
    epoch_wise_val_loss = []
    epoch_wise_train_loss = []
    #
    # =============================== START EPOCHS ===================================
    print(f"\n\nStarting training run for {epochs} epochs...\n\n")
    # ================================================================================
    for epoch in range(epochs):
        # ---------------------------- TRAINING EPOCH --------------------------------
        total_train_loss = 0.0
        num_train_batches = 0
        model.trainable = True # Ensure model is trainable
        #
        # ------------------------------ setup the progress bar ---------------------------------------------
        pbar_train = tqdm(distributed_train_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Train)', leave=False)
        # ---------------------------------------------------------------------------------------------------
        #
        for step, batch in enumerate(pbar_train):
            try:
                # Call the distributed train step function to get the loss for this global batch
                current_train_loss = distributed_train_step(model, optimizer, strategy, global_batch_size, batch, temperature)
                # Calculate total loss and number of batches in this epoch
                total_train_loss += current_train_loss
                num_train_batches += 1

                # Update progress bar with the current step's loss and learning rate
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
                pbar_train.set_postfix({'Train Loss': f'{current_train_loss.numpy():.4f}', 'LR': f'{current_lr:.1E}'})

                # Log step-wise loss to TensorBoard (e.g., every 10 steps)
                if summary_writer and global_step_train % 10 == 0: 
                    with summary_writer.as_default(step=global_step_train):
                        tf.summary.scalar('loss/step_train', current_train_loss, description="Training loss per step")
                        tf.summary.scalar('learning_rate_step', current_lr, description="Learning rate per step")
                #
                global_step_train += 1       
            except Exception as e:
                print(f"\n\nError during train_step (Epoch {epoch+1}, Step {num_train_batches}): {e}")
                continue # Skip this batch
        # ========================== ROBUSTNESS CHECK (TRAINING) ==========================
        if num_train_batches == 0:
            print("\n\n" + "="*80)
            print("FATAL ERROR: The training dataset was empty or smaller than the batch size.")
            print(f"Please check the data at 'path_to_read' and ensure it contains enough samples.")
            print("Aborting training run.")
            print("="*80 + "\n")
            # Stop the training loop entirely
            break 
        # ========================== END OF CHECK =========================================
        # ---------------------------------------------------------------------------------
        # Calculate average training loss for the epoch
        # 
        mean_epoch_train_loss = total_train_loss / num_train_batches
        epoch_wise_train_loss.append(mean_epoch_train_loss.numpy())
        print(f"Epoch {epoch + 1} - Mean Training Loss (Distributed): {mean_epoch_train_loss.numpy():.4f}")
        # ------------------------------ End of Training Epoch ----------------------------
        #
        # ------------------------------- VALIDATION EPOCH --------------------------------
        mean_epoch_val_loss = np.inf # Default to infinity if no validation
        if distributed_val_dataset: # Only run validation if validation data exists
            #
            total_val_loss = 0.0
            num_val_batches = 0
            model.trainable = False # Set model to non-trainable for validation
            # ------------------------------ setup the progress bar ---------------------------------------------
            pbar_val = tqdm(distributed_val_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Val)', leave=False)
            # ---------------------------------------------------------------------------------------------------
            for step, batch in enumerate(pbar_val):
                try:
                    # Call the distributed validation step function to get the loss for this global batch
                    current_val_loss = distributed_validation_step(model, strategy, batch, temperature)
                    # Calculate total loss and number of batches in this epoch
                    total_val_loss += current_val_loss
                    num_val_batches += 1
                    # Update progress bar with the current step's validation loss 
                    pbar_val.set_postfix({'Val Loss': f'{current_val_loss.numpy():.4f}'})
                        
                except Exception as e:
                    print(f"\n\nError during validation_step (Epoch {epoch+1}, Step {num_val_batches}): {e}")
                    continue # Skip batch

            if num_val_batches > 0:
                mean_epoch_val_loss = total_val_loss / num_val_batches
            else:
                # Handle the case where the validation set was empty.
                # Set a default value and print a helpful warning.
                mean_epoch_val_loss = np.inf # Use infinity so it won't be saved as the "best" model
                print("\nWarning: The validation dataset was empty or smaller than the batch size. No validation metrics were calculated for this epoch.")
            #
            epoch_wise_val_loss.append(mean_epoch_val_loss.numpy() if hasattr(mean_epoch_val_loss, 'numpy') else mean_epoch_val_loss)
            print(f"Epoch {epoch + 1} - Mean Validation Loss (Distributed): {mean_epoch_val_loss.numpy():.4f}")
        # ---------------------------------------------------------------------------------
        #------------------------------ End of Validation Epoch ---------------------------
        #------------------------- END OF EPOCH SUMMARY & LOGGING -------------------------
        #  Print Epoch Summary
        # 
        val_loss_for_print = mean_epoch_val_loss.numpy() if hasattr(mean_epoch_val_loss, 'numpy') else mean_epoch_val_loss
        val_loss_str = f"{val_loss_for_print:.5f}" if np.isfinite(val_loss_for_print) else "N/A"
        print(f"\nEpoch {epoch + 1}/{epochs} -> Train Loss: {mean_epoch_train_loss.numpy():.5f} | Val Loss: {val_loss_str}" )      
        #
        current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
        #
        # (IMPORTANT): Remove MLflow logging before packaging
        #
        # ===============================  MLFLOW METRIC LOGGING ===========================
        #
        #
        # 
        mlflow.log_metric("loss/epoch_train", mean_epoch_train_loss, step=epoch)
        mlflow.log_metric("loss/epoch_val", mean_epoch_val_loss, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        # ==================================================================================
        # Log epoch metrics to TensorBoard
        #
        if summary_writer:
            with summary_writer.as_default(step=epoch):
                tf.summary.scalar('loss/epoch_train', mean_epoch_train_loss)
                # Only log the validation loss if it's a finite number
                val_loss_for_log = mean_epoch_val_loss.numpy() if hasattr(mean_epoch_val_loss, 'numpy') else mean_epoch_val_loss
                if np.isfinite(val_loss_for_log):
                    tf.summary.scalar('loss/epoch_val', val_loss_for_log)
                tf.summary.scalar('learning_rate', current_lr)
            summary_writer.flush()
        #
        # ---------------------- Checkpointing and Early Stopping based on Validation Loss -----------------------
        #
        current_metric_for_saving = mean_epoch_val_loss if distributed_val_dataset else mean_epoch_train_loss # Use train loss if no validation
        current_best_metric = current_metric_for_saving.numpy() if hasattr(current_metric_for_saving, 'numpy') else current_metric_for_saving
        #
        # Apply Early Stopping Check ================================================================================
        #
        if current_best_metric < best_val_loss: # Compare with best_val_loss tracker
            status_prefix = f"Val loss" if distributed_val_dataset else f"Train loss"
            print(f"\n  --{status_prefix} improved from {best_val_loss:.5f} to {current_best_metric:.5f}.")
            best_val_loss = current_best_metric # Update best loss tracker
            es_count = 0
            # 
            # Save best weights
            #
            if best_weights_path:
                print(f"\n\nSaving best weights for this run to {best_weights_path}...")
                try:
                    model.save_weights(best_weights_path) # Overwrites previous best FOR THIS RUN
                    print("\n\nWeights saved successfully.\n")
                except Exception as e:
                    print(f"\n\nError saving weights: {e}\n")
        else: # =====================================================================================================
            # Stop early stopping if validation - didn't improve, not available (metric is training), or is inf/nan
            if distributed_val_dataset and np.isfinite(mean_epoch_val_loss):
                es_count += 1
                print(f"\n --Val loss did not improve. Early stopping count: {es_count}/{patience}\n")
            elif not distributed_val_dataset and np.isfinite(mean_epoch_train_loss):
                es_count += 1 # Can still early stop on train loss if no validation
                print(f"\n --Train loss did not improve. Early stopping count: {es_count}/{patience}\n")
            else:
                print("\n --Loss is inf/nan or validation unavailable, skipping early stopping count.\n")
        # ===========================================================================================================
        # Check if early stopping criterion met and STOP training
        #
        if es_count >= patience:
            print(f'\n\n --[INFO] Early Stopping Triggered after {epoch + 1} epochs.\n')
            break
    #
    # ===================================================== END OF EPOCHS =======================================================
    #    
    if summary_writer:
        summary_writer.close()
    print("\n\nTraining finished.")
    #
    # Remove MLflow logging before packaging
    # =========================== MLFLOW MODEL LOGGING ===================================================
    #
    #
    input_example = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
        'mask': tf.zeros((1, build_seq_len), dtype=tf.float32).numpy()
    }
    if best_weights_path:
        print(f"\n\nLogging the complete model to MLflow...\n")
        mlflow.tensorflow.log_model(
            model=model,
            input_example=input_example,
            name="AstraNet(pre-trained)" 
        )
        print("\n\nComplete model logged.")
    # ====================================================================================================
    # Save the weights to the local directory
    #
    if best_weights_path and os.path.exists(best_weights_path):
         print(f"\n\nBest weights saved at: {best_weights_path} (Best Val Loss: {best_val_loss:.5f})")
    else:
         print("\n\nNo weights were saved (either no improvement found or path issue).")

    return epoch_wise_train_loss, epoch_wise_val_loss 