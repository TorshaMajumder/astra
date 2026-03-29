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
from astra.src.encoder import Encoder
from astra.src.embedding import AstraEmbedding
from astra.src.loss import nt_xent_loss, distil_xent_loss 
from astra.src.header import ProjectionHead, DistilProjectionHead
from astra.src.scheduler import get_teacher_temp_schedule, get_momentum_schedule
from astra.src.preprocessing import contrastive_data_loader, astra_distil_data_loader


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
                                                use_band_info=use_band_info, use_drop=use_drop, mjd=mjd, 
                                                time_scaling=time_scaling, name="astra_embedding"
                                            )
        #
        # Instantiate Encoder Layer
        #
        self.encoder = Encoder(
                                num_layers=num_layers, d_model=d_model, 
                                num_heads=num_heads, dff=dff, rate=rate, use_res=use_res, name="astra_encoder" 
                            )
        #
        # Instantiate Pooling Layer
        # Using Global Average Pooling as default
        #
        self.pooling = layers.GlobalAveragePooling1D(name='avg_pooling')
        #
        # Instantiate Projection Head Layer
        #
        if self.projection_dim is not None:
            self.projection_head_layer = ProjectionHead(
                                                            d_model=d_model, 
                                                            projection_dim=projection_dim,
                                                            name="astra_ph_2"
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
            is_local_view (bool): Whether to use linear segment embedding (True) or nonlinear (False).

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
        if self.projection_dim is not None:
            final_output = self.projection_head_layer(pooled_output, training=training) # (batch, projection_dim)
        else:
            final_output = pooled_output
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
                        "time_scaling": self.time_scaling,
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





@tf.keras.utils.register_keras_serializable()
class AstraNet_Distil(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, projection_out=65536, 
                 base=10000.0, time_scaling=100, use_res=True, use_band_info=True, rate=0.1,
                 use_drop=False, mjd=True, name="astra_net_distil", **kwargs):
        super(AstraNet_Distil, self).__init__(name=name, **kwargs)
        
        self.projection_out = projection_out
        self.backbone = AstraNet(
                                    num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, 
                                    projection_dim=None, # Use DistilProjectionHead
                                    base=base, time_scaling=time_scaling, use_res=use_res, 
                                    use_band_info=use_band_info, rate=rate, use_drop=use_drop, mjd=mjd,
                                    name="astra_net"
                                )
        
        self.distil_head = DistilProjectionHead(in_dim=d_model, out_dim=projection_out, name="ph")

    def call(self, inputs, training=False):
        # Get embeddings from backbone
        features = self.backbone(inputs, training=training)
        # Get Distil logits
        logits = self.distil_head(features, training=training)
        return features, logits

    def get_config(self):
        config = super(AstraNet_Distil, self).get_config()
        backbone_config = self.backbone.get_config()
        config.update(backbone_config)
        config.update({
                        
                        "projection_out": self.projection_out
                        
                    })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.function
def distributed_train_step_distil(student, teacher, optimizer, strategy, dist_inputs, center, 
                                student_temp, teacher_temp, momentum_teacher=0.996, momentum_center=0.9, num_global_views=None):
    
    def train_step(inputs):
        # 3 Global Views, 6 Local Views
        global_views = inputs[:num_global_views]
        all_views = inputs # All 9 views
        
        num_replicas = strategy.num_replicas_in_sync
        replica_context = tf.distribute.get_replica_context()
        
        with tf.GradientTape() as tape:
            # TEACHER Forward Pass (ONLY sees Global Views!)
            teacher_logits =[]
            for gv in global_views:
                # Teacher is in training=False (eval mode)
                _, t_logits = teacher(gv, training=False)
                teacher_logits.append(tf.stop_gradient(t_logits))
                
            # STUDENT Forward Pass (Sees ALL Views)
            student_logits =[]
            for i, view in enumerate(all_views):
                # First 3 are global (is_local=False), next 6 are local 
                is_local = (i >= num_global_views)
                _, s_logits = student(view, training=True)
                student_logits.append(s_logits)
                
            # Compute Cross-Entropy Loss
            total_loss = 0.0
            n_loss_terms = 0
            
            # Every Student view predicts every Teacher global view (except itself)
            for t_idx, t_logit in enumerate(teacher_logits):
                for s_idx, s_logit in enumerate(student_logits):
                    if t_idx == s_idx:
                        continue # Do not predict a view from itself
                    
                    loss_term = distil_xent_loss(
                                                    student_logits=s_logit, 
                                                    teacher_logits=t_logit, 
                                                    center=center, 
                                                    student_temp=student_temp, 
                                                    teacher_temp=teacher_temp
                                                )
                    total_loss += loss_term
                    n_loss_terms += 1
             
            avg_loss = total_loss / tf.cast(n_loss_terms, tf.float32)
            
            # Scale loss for MirroredStrategy
            scaled_loss = avg_loss / tf.cast(num_replicas, tf.float32)

        # Compute and Apply Gradients ONLY to the Student
        grads = tape.gradient(scaled_loss, student.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))

        # EMA Update for the Teacher Weights
        # Teacher_Weight = momentum * Teacher_Weight + (1 - momentum) * Student_Weight
        for s_weight, t_weight in zip(student.variables, teacher.variables):
            t_weight.assign(momentum_teacher * t_weight + (1.0 - momentum_teacher) * s_weight)

        # EMA Update for the Center
        # Gather all teacher logits across the 8x A100s to compute a true global mean
        concat_t_logits = tf.concat(teacher_logits, axis=0) # Stack the 3 global views

        if num_replicas > 1:
            global_t_logits = replica_context.all_gather(concat_t_logits, axis=0)
        else:
            global_t_logits = concat_t_logits
            
        # Compute mean across the global batch
        batch_center = tf.reduce_mean(global_t_logits, axis=0, keepdims=True)
        
        # Update the center
        # Capture the assignment operation
        center_update_op = center.assign(momentum_center * center + (1.0 - momentum_center) * batch_center)
        # ---------------------------------------------------------------------
        # Force TF to wait for the assignment, then return the Tensor
        # ---------------------------------------------------------------------
        with tf.control_dependencies([center_update_op]):
            return tf.identity(avg_loss)   

    # Run distributed step
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    global_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return tf.identity(global_loss)


@tf.function
def distributed_val_step_distil(student, teacher, strategy, dist_inputs, center, student_temp, teacher_temp, num_global_views):
    
    def val_step(inputs):
        # 3 Global Views, 6 Local Views
        global_views = inputs[:num_global_views]
        all_views = inputs
        
        # TEACHER Forward Pass
        teacher_logits =[]
        for gv in global_views:
            _, t_logits = teacher(gv, training=False)
            teacher_logits.append(t_logits)
            
        # STUDENT Forward Pass
        student_logits =[]
        for i, view in enumerate(all_views):
            is_local = (i >= num_global_views)
            _, s_logits = student(view, training=False)
            student_logits.append(s_logits)
            
        # Compute Cross-Entropy Loss (No gradients, no updates)
        total_loss = 0.0
        n_loss_terms = 0
        
        for t_idx, t_logit in enumerate(teacher_logits):
            for s_idx, s_logit in enumerate(student_logits):
                if t_idx == s_idx:
                    continue 
                
                loss_term = distil_xent_loss(
                                                student_logits=s_logit, 
                                                teacher_logits=t_logit, 
                                                center=center, 
                                                student_temp=student_temp, 
                                                teacher_temp=teacher_temp
                                            )
                total_loss += loss_term
                n_loss_terms += 1
                
        avg_loss = total_loss / tf.cast(n_loss_terms, tf.float32)
        return avg_loss

    # Run distributed validation step
    per_replica_losses = strategy.run(val_step, args=(dist_inputs,))
    # Use MEAN reduction to get the true validation loss
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)



def k_distil_training(student,
                         teacher,
                         center,
                         strategy,
                         optimizer,
                         # --------------
                         path_to_read="",
                         path_to_val="",
                         path_to_save=None,
                         # --------------
                         global_batch_size=256,
                         patience=20,
                         epochs=100,
                         buffer_size=10000,
                         seed=1024,
                         # --------------
                         student_temp=0.1,
                         start_t_temp=0.04,
                         base_t_temp=0.07,
                         warmup_epochs_temp=30,
                         base_ema_m=0.996,
                         final_ema_m=1.0,
                         use_ema_scheduler=False,
                         momentum_center=0.9,
                         # ---------------
                         num_global_views=3,
                         num_local_views=6,
                         gv_maxlens=None,
                         lv_maxlens_list=None, 
                         apply_noise_list=None, 
                         noise_levels_list=None, 
                         apply_binning_list=None, 
                         apply_outlier_list=None,
                         bin_widths_list=None, 
                         drop_rates_list=None,
                         # ---------------
                         build_seq_len=None
                    ):
    
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
        weights_filename = 'best_distil_teacher_weights' 
        best_weights_path = os.path.join(path_to_save, weights_filename)
        best_student_wt_path = os.path.join(path_to_save, "student", "best_weight")
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
            train_loader = astra_distil_data_loader(
                                                    source=path_to_read, 
                                                    batch_size=global_batch_size, 
                                                    buffer_size=buffer_size,
                                                    seed=seed,
                                                    gv_maxlens=gv_maxlens, 
                                                    lv_maxlens_list=lv_maxlens_list, 
                                                    num_local_views=num_local_views,
                                                    num_global_views=num_global_views,
                                                    apply_noise_list=apply_noise_list, 
                                                    noise_levels_list=noise_levels_list, 
                                                    apply_binning_list=apply_binning_list, 
                                                    apply_outlier_list=apply_outlier_list,
                                                    bin_widths_list=bin_widths_list, 
                                                    drop_rates_list=drop_rates_list, 
                                                    build_seq_len=build_seq_len   
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
            no_aug_bool = [False] * len(apply_noise_list)
            valid_loader = astra_distil_data_loader(
                                                    source=path_to_val, 
                                                    batch_size=global_batch_size, 
                                                    buffer_size=buffer_size,
                                                    seed=seed,
                                                    gv_maxlens=gv_maxlens, 
                                                    lv_maxlens_list=lv_maxlens_list, 
                                                    num_local_views=num_local_views,
                                                    num_global_views=num_global_views,
                                                    apply_noise_list=no_aug_bool, 
                                                    noise_levels_list=noise_levels_list, 
                                                    apply_binning_list=no_aug_bool, 
                                                    apply_outlier_list=no_aug_bool,
                                                    bin_widths_list=bin_widths_list, 
                                                    drop_rates_list=drop_rates_list, 
                                                    build_seq_len=build_seq_len 
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
    # =======================================================================================
    # Setup Metrics & Checkpoints
    #
    # ===========================================================================================================
    #
    # --------------- All Variables ------------------
    es_count = 0
    best_val_loss = np.inf
    epoch_wise_val_loss =[]
    epoch_wise_train_loss =[]
    #
    # ------------- Initialize Metrics and Checkpointing ------------------
    train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
    val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
    # =============================== START EPOCHS ===================================
    print(f"\n\nStarting training run for {epochs} epochs...\n\n")
    # ================================================================================
    for epoch in range(epochs):
        # ---------------------------- TRAINING EPOCH --------------------------------
        #
        # Reset metrics at the start of each epoch
        #
        train_loss_tracker.reset_states()
        val_loss_tracker.reset_states()
        # 1. Evaluate schedules for the current epoch
        current_t_temp = get_teacher_temp_schedule(epoch, warmup_epochs=warmup_epochs_temp, start_temp=start_t_temp, base_temp=base_t_temp)
        
        if use_ema_scheduler:
            current_m_teacher = get_momentum_schedule(epoch, total_epochs=epochs, base_m=base_ema_m, final_m=final_ema_m)
        else:
            current_m_teacher = base_ema_m

        # 2. Cast to tf.constant to prevent retracing
        s_temp = tf.constant(student_temp, dtype=tf.float32) # Stays fixed at 0.1
        t_temp = tf.constant(current_t_temp, dtype=tf.float32)
        m_teacher = tf.constant(current_m_teacher, dtype=tf.float32)
        m_center = tf.constant(momentum_center, dtype=tf.float32) # Stays fixed at 0.9
        
        # ------------------------------ setup the progress bar ---------------------------------------------
        pbar_train = tqdm(distributed_train_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Train)', leave=False)
        # ---------------------------------------------------------------------------------------------------
        # pbar_train = tqdm(distributed_train_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Train)', leave=False)
        
        for step, batch in enumerate(pbar_train):
            try:
                current_train_loss = distributed_train_step_distil(
                                                                    student=student, 
                                                                    teacher=teacher, 
                                                                    optimizer=optimizer, 
                                                                    strategy=strategy, 
                                                                    dist_inputs=batch, 
                                                                    center=center, 
                                                                    student_temp=s_temp, 
                                                                    teacher_temp=t_temp,
                                                                    momentum_teacher=m_teacher,
                                                                    momentum_center=m_center,
                                                                    num_global_views=num_global_views
                                                                )
                if current_train_loss is None:
                    raise ValueError("CRITICAL BUG: distributed_train_step_dino returned None instead of a Tensor!")

                train_loss_tracker.update_state(current_train_loss)
                
                # Fetch learning rate
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
                pbar_train.set_postfix({'Train Loss': f'{current_train_loss.numpy():.4f}', 'LR': f'{current_lr:.1E}'})

                # TensorBoard Step Logging
                if summary_writer and step % 5 == 0: 
                    with summary_writer.as_default(step=optimizer.iterations):
                        tf.summary.scalar('loss/step_train', current_train_loss, description="Training loss per step")
                        tf.summary.scalar('learning_rate_step', current_lr, description="Learning rate per step")
                        
            except Exception as e:
                print(f"\nError during train_step (Epoch {epoch+1}, Step {step}): {e}")
                traceback.print_exc()
                continue
        #
        # ------------------------------- VALIDATION EPOCH --------------------------------
        if distributed_val_dataset:
            # ------------------------------ setup the progress bar ---------------------------------------------
            pbar_val = tqdm(distributed_val_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Val)', leave=False)
            # ---------------------------------------------------------------------------------------------------
            for step, batch in enumerate(pbar_val):
                try:
                    current_val_loss = distributed_val_step_distil(
                                                                    student=student, 
                                                                    teacher=teacher, 
                                                                    strategy=strategy, 
                                                                    dist_inputs=batch, 
                                                                    center=center, 
                                                                    student_temp=s_temp, 
                                                                    teacher_temp=t_temp,
                                                                    num_global_views=num_global_views
                                                                )
                    if current_val_loss is None:
                        raise ValueError("CRITICAL BUG: distributed_val_step_dino returned None instead of a Tensor!")

                    val_loss_tracker.update_state(current_val_loss)
                    pbar_val.set_postfix({'Val Loss': f'{current_val_loss.numpy():.4f}'})
                except Exception as e:
                    print(f"\nError during validation_step (Epoch {epoch+1}, Step {step}): {e}")
                    continue

        # ---------------------------------------------------------------------------------
        #------------------------------ End of Validation Epoch ---------------------------
        #------------------------- END OF EPOCH SUMMARY & LOGGING -------------------------
        #  Print Epoch Summary
        # 
        current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
        epoch_train_loss = train_loss_tracker.result()
        epoch_val_loss = val_loss_tracker.result()
        
        epoch_wise_train_loss.append(float(epoch_train_loss.numpy()))
        epoch_wise_val_loss.append(float(epoch_val_loss.numpy()) if np.isfinite(epoch_val_loss) else epoch_val_loss)
        print(f"\n{20*'='} EPOCH ({epoch + 1}/{epochs}) Summary {20*'='}\n")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}\n")
        # ==================================================================================
        # Log epoch metrics to TensorBoard
        #
        if summary_writer:
            with summary_writer.as_default(step=epoch):
                
                tf.summary.scalar('loss/epoch_train', epoch_train_loss)
                val_loss_for_log = epoch_val_loss.numpy() if hasattr(epoch_val_loss, 'numpy') else epoch_val_loss
                
                if np.isfinite(val_loss_for_log):
                    tf.summary.scalar('loss/epoch_val', val_loss_for_log)
                
                tf.summary.scalar('learning_rate', current_lr)
            summary_writer.flush()
        #
        #
        # (IMPORTANT): Remove MLflow logging before packaging
        #
        # ===============================  MLFLOW METRIC LOGGING ===========================
        mlflow.log_metric("loss/epoch_train", float(epoch_train_loss.numpy()), step=epoch)
        if np.isfinite(epoch_val_loss):
            mlflow.log_metric("loss/epoch_val", float(epoch_val_loss.numpy()), step=epoch)
        mlflow.log_metric("learning_rate", float(current_lr), step=epoch)

        # ==================================================================================
        # ---------------------- Checkpointing and Early Stopping based on Validation Loss -----------------------
        #
        # if epoch_val_loss < best_val_loss:
        #     print(f"\n  -- Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model...\n")
        #     best_val_loss = epoch_val_loss
        #     es_count = 0
            
        if best_weights_path:
            try:
                # Save the TEACHER model as the final output
                teacher.save_weights(best_weights_path, save_format='tf') 
                student.save_weights(best_student_wt_path, save_format='tf')
                print(f"\nTeacher & Student weights (Epoch: {epoch}/{epochs}) saved successfully to {best_weights_path}.\n")
            except Exception as e:
                print(f"\nError saving weights: {e}\n")
        # else:
        #     if distributed_val_dataset and np.isfinite(epoch_val_loss):
        #         es_count += 1
        #         print(f"\n  -- Val loss did not improve. Early stopping count: {es_count}/{patience}\n")
        #     elif not distributed_val_dataset and np.isfinite(epoch_train_loss):
        #         es_count += 1 
        #         print(f"\n  -- Train loss did not improve. Early stopping count: {es_count}/{patience}\n")
                
        # if es_count >= patience:
        #     print(f'\n\n --[INFO] Early Stopping Triggered after {epoch + 1} epochs.\n')
        #     break       
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
    if best_weights_path:
        input_example = {
            'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
            'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
            'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32).numpy(),
            'mask': tf.zeros((1, build_seq_len), dtype=tf.float32).numpy()
        }
        print(f"\nLogging the Teacher model to MLflow...")
        mlflow.tensorflow.log_model(
            model=teacher,
            input_example=input_example,
            name="AstraNet-Distil-Teacher(pre-trained)" 
        )
        print("\n\nTeacher model logged.")
    # ====================================================================================================
    # Save the weights to the local directory
    #
    if best_weights_path and os.path.exists(best_weights_path):
         print(f"\n\n-- Best weights saved at: {best_weights_path} (Best Val Loss: {best_val_loss:.5f})")
    else:
         print("\n\nNo weights were saved (either no improvement found or path issue).")

    return epoch_wise_train_loss, epoch_wise_val_loss