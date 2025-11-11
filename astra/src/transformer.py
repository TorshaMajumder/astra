import os
import mlflow
import logging
import traceback
import datetime 
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
from astra.src.scheduler import CustomSchedule, warmup_schedule


logging.getLogger('tensorflow').setLevel(logging.ERROR)  
os.system('clear')


class AstraNet(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1,
                 base=10000.0, use_res=True, use_band_info=True,
                 use_drop=False, mjd=True, projection_dim=None, name="astra_net", **kwargs):
        super(AstraNet, self).__init__(name=name, **kwargs)


        self.d_model = d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dff=dff
        self.rate=rate
        self.base=base
        self.use_res=use_res
        self.use_band_info=use_band_info
        self.use_drop=use_drop
        self.mjd=mjd
        self.projection_dim=projection_dim
        # 1. Instantiate Embedding Layer
        self.embedding_layer = AstraEmbedding(
                                                    d_model=d_model, base=base, rate=rate, # Pass shared rate
                                                    use_band_info=use_band_info, use_drop=use_drop, mjd=mjd
                                                )

        # 2. Instantiate Encoder Layer
        self.encoder = Encoder(
                                    num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                                    dff=dff, rate=rate, use_res=use_res 
                                )
        # 3. Instantiate Pooling Layer
        # Using Global Average Pooling as default
        self.pooling = layers.GlobalAveragePooling1D(name='avg_pooling')

        # 4. Instantiate Projection Head Layer
        self.projection_head_layer = ProjectionHead(
                                                        d_model=d_model, 
                                                        projection_dim=projection_dim
                                                    )



    
    def call(self, x, training=False):
        """
        Forward pass through the AstraNet.

        Args:
            x (dict): Input dictionary containing:
                'input': Magnitude tensor (batch, seq_len, 1)
                'times': Time tensor (batch, seq_len, 1)
                'band_info': Band tensor (batch, seq_len, 1) (optional)
                'mask': Mask tensor (batch, seq_len) or (batch, seq_len, 1)
            training (bool): Flag for training mode (affects Dropout, BN).

        Returns:
            tf.Tensor: Final output tensor, usually after projection head.
                       Shape: (batch_size, projection_dim) or (batch_size, d_model) if no projection.
        """
        if not isinstance(x, dict) or not all(k in x for k in ['input', 'times', 'mask']):
             raise ValueError("Input 'x' must be a dictionary containing at least 'input', 'times', and 'mask'.")
        
        mask = x['mask']       # (batch, seq_len) - ensure last dim is squeezed
        # Ensure mask has the correct shape (batch, seq_len) for Encoder/Pooling
        if len(mask.shape) == 3 and tf.shape(mask)[-1] == 1:
             mask = tf.squeeze(mask, axis=-1)
        elif len(mask.shape) != 2:
             raise ValueError(f"Unexpected mask shape in AstraNet: {tf.shape(mask)}. Expected (batch, seq_len).")

        # 1. Apply Embedding Layer (takes the dictionary 'x')
        # Pass training flag for potential dropout in embedding
        embeddings = self.embedding_layer(x, training=training) # (batch, seq_len, d_model)

        # 2. Apply Encoder (takes embeddings and mask)
        # Pass training flag for dropout/LN in encoder
        enc_output, all_attention_weights = self.encoder(embeddings, mask, training=training) # (batch, seq_len, d_model)

        # 3. Apply Pooling
        # Invert mask for pooling (True where elements should be KEPT)
        pool_mask = tf.logical_not(tf.cast(mask, tf.bool))
        pooled_output = self.pooling(enc_output, mask=pool_mask) # (batch, d_model)

        # 4. Apply Projection Head 
        # Pass training flag if projection head had dropout/BN (currently doesn't)
        final_output = self.projection_head_layer(pooled_output, training=training) # (batch, projection_dim or d_model)

        return final_output # Return the final output tensor

    # === SOLUTION: Implement get_config ===
    def get_config(self):
        # Start with the base class's config.
        config = super(AstraNet, self).get_config()
        
        # Update it with the custom arguments from __init__.
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
            "base": self.base,
            "use_res": self.use_res,
            "use_band_info": self.use_band_info,
            "use_drop": self.use_drop,
            "mjd": self.mjd,
            "projection_dim": self.projection_dim,
        })
        return config

    # === SOLUTION: Implement from_config ===
    @classmethod
    def from_config(cls, config):
        # This creates a new instance of the model from the config dictionary.
        return cls(**config)


# ==========================================================
# 1. DEFINE THE DISTRIBUTED TRAINING STEP
# ==========================================================
@tf.function
def distributed_train_step(model, optimizer, strategy, global_batch_size, dist_inputs, temperature):
    """
    Performs one distributed training step.
    """
    # This function will be executed on each replica (GPU).
    def step_fn(inputs):
        # Unpack the views for this replica's mini-batch
        *views_batch, = inputs

        with tf.GradientTape() as tape:
            # Get projections from the model
            z_views = [model(view, training=True) for view in views_batch]
            
            # Calculate the loss for this replica's mini-batch
            per_replica_loss = nt_xent_loss(*z_views, temperature=temperature)
            
            # IMPORTANT: Scale the loss by the GLOBAL batch size.
            # This ensures the gradients are correctly averaged, not summed.
            scaled_loss = tf.nn.compute_average_loss(per_replica_loss, global_batch_size=global_batch_size)

        # Calculate gradients and apply them
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        # Optional: Gradient Clipping
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Return the raw loss for this replica. We'll aggregate it later.
        #return per_replica_loss
        # --- THIS IS THE FIX ---
        # Instead of returning the whole vector, return the scalar mean for this replica.
        return tf.reduce_mean(per_replica_loss)
        # --- END OF FIX ---

    # Use strategy.run to execute step_fn on each replica in parallel.
    per_replica_losses = strategy.run(step_fn, args=(dist_inputs,))
    
    # Aggregate the losses from all replicas. Using MEAN is often more intuitive for logging.
    # This gives you the average loss over the entire global batch.
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

# ==========================================================
# 2. DEFINE THE DISTRIBUTED VALIDATION STEP
# ==========================================================
@tf.function
def distributed_validation_step(model, strategy, dist_inputs, temperature):
    """
    Performs one distributed validation step.
    """
    def step_fn(inputs):
        *views_batch, = inputs
        z_views = [model(view, training=False) for view in views_batch] # training=False
        per_replica_loss = nt_xent_loss(*z_views, temperature=temperature)
        # --- THIS IS THE FIX ---
        # Return the scalar mean for this replica.
        return tf.reduce_mean(per_replica_loss)
        # --- END OF FIX ---

    per_replica_losses = strategy.run(step_fn, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

       





def contrastive_train(model,
          strategy,
          optimizer,
          build_seq_len=None,
          path_to_read="",
          path_to_val="",
          path_to_save=None,
          n_views=3,
          batch_size=50,
          temperature=0.1, # Default from SimCLR
          patience=20,
          epochs=10, # Increased default epochs
          initial_lr=1e-3, # For Adam optimizer if not using custom schedule
          use_custom_schedule=True,
          warmup_steps=4000,
          apply_white_noise=(False, True, True),
          noise_levels=(0.0, 0.1, 0.1), # Example noise levels
          apply_binning=(False, False, True),
          apply_outlier=(False, False, True),
          maxlens=(200, 100, 200),
          bin_widths=(5, 5, 5),
          drop_rates=(0.0, 0.30, 0.60),
          buffer_size=100, # Shuffle buffer size
         ):


    # --- Setup Paths and TensorBoard Writer ---
    best_weights_path = None
    summary_writer = None

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
    # --- End Setup Paths and TensorBoard Writer ---
    # ================================================================================
    #
    # Create training data loader ONCE before the loop
    #
    print("\n\nSetting up data loader for training...\n\n")
    if not path_to_read or not os.path.exists(path_to_read):
         print("\n\nWarning: Training Dataset path not provided or does not exist. Please provide training data.")
         return None
    else:    
        try:
            train_loader = contrastive_data_loader(
                                                    source=path_to_read,
                                                    n_views=n_views, 
                                                    seed=np.random.randint(1024), 
                                                    batch_size=batch_size,
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
            # --- Distribute the datasets using the strategy ---
            #
            distributed_train_dataset = strategy.experimental_distribute_dataset(train_loader)
            # -------------------------------------------------------------------------------
            print("\n\nData loader ready.")
        except ValueError as e:
            print(f"\n\nError creating data loader: {e}")
            traceback.print_exc()
            return None # Return None if setup fails
    # --- End Training Data Loader ---
    # ================================================================================
    #
    # Create validation data loader ONCE before the loop
    #
    print("\n\nSetting up data loader for validation...\n\n")
    valid_loader = None
    distributed_val_dataset = None
    if not path_to_val or not os.path.exists(path_to_val):
         print("\n\nWarning: Validation path not provided or does not exist. Skipping validation.")
    else:
        try:
            valid_loader = contrastive_data_loader(
                                                    source=path_to_val,
                                                    n_views=n_views, 
                                                    seed=np.random.randint(1024), # Use different seed maybe?
                                                    batch_size=batch_size,
                                                    build_seq_len=build_seq_len,
                                                    apply_white_noise=apply_white_noise,
                                                    noise_levels=noise_levels,
                                                    apply_binning=apply_binning,
                                                    apply_outlier=apply_outlier,
                                                    maxlens=maxlens,
                                                    bin_widths=bin_widths,
                                                    drop_rates=drop_rates,
                                                    buffer_size=max(buffer_size // 4, 10) # Smaller buffer for validation
                                                )
            # -------------------------------------------------------------------------------
            # --- Distribute the datasets using the strategy ---
            #
            distributed_val_dataset = strategy.experimental_distribute_dataset(valid_loader)
            # -------------------------------------------------------------------------------
            print("\n\nValidation loader ready.")
        except ValueError as e:
            print(f"\n\nError creating validation loader: {e}")
            traceback.print_exc()
            valid_loader = None # Proceed without validation if loader fails
    # --- End Validation Data Loader ---
    # --- End of Data Loaders ---
    #
    # ================================================================================
    # --- Training Loop ---
    # ===============================================================================
    # ----- Variables --------
    es_count = 0
    global_step_val = 0
    global_step_train = 0 # Separate step counters
    best_val_loss = np.inf # Initialize best loss to infinity
    epoch_wise_val_loss = []
    epoch_wise_train_loss = []
    #
    # ================================================================================
    print(f"\n\nStarting training run for {epochs} epochs...\n\n")
    # ================================================================================
    for epoch in range(epochs):
        # ------------------------- TRAINING EPOCH -------------------------
        #step_wise_train_loss = []
        total_train_loss = 0.0
        num_train_batches = 0
        model.trainable = True # Ensure model is trainable
        # -------- setup the progress bar --------------------------------------------------------
        pbar_train = tqdm(distributed_train_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Train)', leave=False)
        # ----------------------------------------------------------------------------------------
        for step, views in enumerate(pbar_train):
            try:
                # Call the distributed train step function to get the loss for this global batch
                current_train_loss = distributed_train_step(model, optimizer, strategy, batch_size, views, temperature)
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

                global_step_train += 1       
            

            except Exception as e:
                print(f"\n\nError during train_step (Epoch {epoch+1}, Step {num_train_batches}): {e}")
                continue # Skip this batch
        # ----------------------------------------------------------------------
        # Calculate average training loss for the epoch
        # 
        mean_epoch_train_loss = total_train_loss / num_train_batches
        epoch_wise_train_loss.append(mean_epoch_train_loss.numpy())
        print(f"Epoch {epoch + 1} - Average Training Loss: {mean_epoch_train_loss.numpy():.4f}")
        # ------------------------------ End Training Epoch -----------------------
        #
        # -------------------------- VALIDATION EPOCH -----------------------------
        #step_wise_val_loss = []
        mean_epoch_val_loss = np.inf # Default to infinity if no validation
        if distributed_val_dataset: # Only run validation if loader exists
            #
            total_val_loss = 0.0
            num_val_batches = 0
            model.trainable = False # Set model to non-trainable for validation
            pbar_val = tqdm(distributed_val_dataset, desc=f'Epoch {epoch + 1}/{epochs} (Val)', leave=False)
        
            for step, views in enumerate(pbar_val):
                try:
                    current_val_loss = distributed_validation_step(model, strategy, views, temperature)
                    total_val_loss += current_val_loss
                    num_val_batches += 1
                    pbar_val.set_postfix({'Val Loss': f'{current_val_loss.numpy():.4f}'})
                    # Log step-wise loss to TensorBoard (optional)
                    if summary_writer and global_step_val % 10 == 0:
                        with summary_writer.as_default(step=global_step_val):
                            tf.summary.scalar('loss/step_val', current_val_loss, description="Validation loss per step")
                    
                    global_step_val += 1
                        
                except Exception as e:
                    print(f"\n\nError during validation_step (Epoch {epoch+1}, Step {num_val_batches}): {e}")
                    continue # Skip batch


            mean_epoch_val_loss = total_val_loss / num_val_batches
            epoch_wise_val_loss.append(mean_epoch_val_loss.numpy())
        #------------------------------ End Validation Epoch -----------------------
        # ------------------------- END OF EPOCH SUMMARY & LOGGING -------------------------
        #  Print Epoch Summary
        # 
        val_loss_str = f"{mean_epoch_val_loss.numpy():.5f}" if distributed_val_dataset else "N/A"
        print(f"\nEpoch {epoch + 1}/{epochs} -> Train Loss: {mean_epoch_train_loss.numpy():.5f} | Val Loss: {val_loss_str}")
        #
        current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
        #
        # Remove MLflow logging before packaging
        #
        # === MLFLOW METRIC LOGGING ===
        #
        #
        # 
        mlflow.log_metric("loss/epoch_train", mean_epoch_train_loss, step=epoch)
        mlflow.log_metric("loss/epoch_val", mean_epoch_val_loss, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        # =============================
        # Log epoch metrics to TensorBoard
        # Log epoch metrics to TensorBoard
        if summary_writer:
            with summary_writer.as_default(step=epoch):
                tf.summary.scalar('loss/epoch_train', mean_epoch_train_loss)
                if valid_loader:
                    tf.summary.scalar('loss/epoch_val', mean_epoch_val_loss)
                tf.summary.scalar('learning_rate', current_lr)
            summary_writer.flush()
        #
        # --- Checkpointing and Early Stopping based on Validation Loss ---
        #
        current_best_metric = mean_epoch_val_loss if distributed_val_dataset else mean_epoch_train_loss # Use train loss if no validation
        #
        # Early Stopping Check
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
        else:
            # Stop early stopping if validation isn't available or loss is inf/nan
            if distributed_val_dataset and np.isfinite(mean_epoch_val_loss):
                es_count += 1
                print(f"\n --Val loss did not improve. Early stopping count: {es_count}/{patience}\n")
            elif not distributed_val_dataset and np.isfinite(mean_epoch_train_loss):
                es_count += 1 # Can still early stop on train loss if no validation
                print(f"\n --Train loss did not improve. Early stopping count: {es_count}/{patience}\n")
            else:
                print("\n --Loss is inf/nan or validation unavailable, skipping early stopping count.\n")

        if es_count >= patience:
            print(f'\n\n --[INFO] Early Stopping Triggered after {epoch + 1} epochs.\n')
            break
    
    # --- End of Training Loop ---    
    if summary_writer:
        summary_writer.close()
    
    print("\n\nTraining finished.")
    #
    # Remove MLflow logging before packaging
    #    # === MLFLOW MODEL LOGGING ===
    #
    #
    if best_weights_path:
        print(f"\n\nLogging the complete model to MLflow...\n\n")
        mlflow.tensorflow.log_model(
            model=model,
            name="AstraNet(pre-trained)",
            registered_model_name="AstraNet(pre-trained)" 
        )
        print("\n\nComplete model logged.")
    # =============================
    # Save the weights to the local directory
    #
    if best_weights_path and os.path.exists(best_weights_path):
         print(f"\n\nBest weights saved at: {best_weights_path} (Best Val Loss: {best_val_loss:.5f})")
    else:
         print("\n\nNo weights were saved (either no improvement found or path issue).")

    return epoch_wise_train_loss, epoch_wise_val_loss # Return both histories

