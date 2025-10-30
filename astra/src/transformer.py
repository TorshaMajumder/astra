import os
import logging
import traceback
import datetime 
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from astra.src.encoder   import Encoder
from astra.src.loss import nt_xent_loss 
from astra.src.embedding import AstraEmbedding
from astra.src.header import ProjectionHead
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
        enc_output = self.encoder(embeddings, mask, training=training) # (batch, seq_len, d_model)

        # 3. Apply Pooling
        # Invert mask for pooling (True where elements should be KEPT)
        pool_mask = tf.logical_not(tf.cast(mask, tf.bool))
        pooled_output = self.pooling(enc_output, mask=pool_mask) # (batch, d_model)

        # 4. Apply Projection Head 
        # Pass training flag if projection head had dropout/BN (currently doesn't)
        final_output = self.projection_head_layer(pooled_output, training=training) # (batch, projection_dim or d_model)

        return final_output # Return the final output tensor




@tf.function
def contrastive_train_step(model, *views_batch, temperature, optimizer):
    """
    Performs a single training step for a variable number of views.

    Args:
        model: The Keras model to train.
        *views_batch: A variable number of view batches (e.g., anchor_batch, positive_batch).
                      Each view is a dictionary of tensors from the data loader.
        temperature: The temperature for the NT-XENT loss.
        optimizer: The optimizer to use.
    
    Returns:
        The calculated loss for the step.
    """
    

    with tf.GradientTape() as tape:
        
        
        z_views = [model(view, training=True) for view in views_batch] # List of (B, D) tensors

        # Calculate NT-XENT loss between anchor and positive views
        loss = nt_xent_loss(*z_views, temperature=temperature)

        # Handle potential NaN/Inf loss
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            tf.print("\n\nWarning: Loss is NaN or Inf. Setting loss to 0.0 for this step.\n\n", loss)
            loss = tf.constant(0.0, dtype=tf.float32) # Use float32

    # Calculate and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Optional: Gradient Clipping
    # grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss
       


@tf.function
def contrastive_validation_step(model, *views_batch, temperature):
   
    
    z_views = [model(view, training=True) for view in views_batch] # List of (B, D) tensors

    # Calculate NT-XENT loss between anchor and positive views
    loss = nt_xent_loss(*z_views, temperature=temperature)

    # Handle potential NaN/Inf loss
    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
        tf.print("\n\nWarning: Loss is NaN or Inf. Setting loss to 0.0 for this step.\n\n", loss)
        loss = tf.constant(0.0, dtype=tf.float32) # Use float32

    return loss


def contrastive_train(model,
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
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_weights_path = None
    summary_writer = None

    if path_to_save:
       

        weights_filename = 'best_contrastive.weights.h5' # Simpler name within run dir
        best_weights_path = os.path.join(path_to_save, weights_filename)

        # Create TensorBoard writer
        summary_writer = tf.summary.create_file_writer(path_to_save)
        print(f"\n\nRun Directory (Weights & TensorBoard Logs): {path_to_save}")
        print(f"\n\nWill save best weights to: {best_weights_path}")

    else:
        summary_writer = None # No logging if path_to_save is not provided

    

    # Optimizer
    if use_custom_schedule:
        custom_lr = CustomSchedule(model.encoder.d_model, warmup_steps=warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate=custom_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Create data loader ONCE before the loop
    print("Setting up data loader for training...")
    try:
        train_loader = contrastive_data_loader(
            source=path_to_read,
            n_views=n_views, 
            seed=np.random.randint(1024), 
            batch_size=batch_size,
            apply_white_noise=apply_white_noise,
            noise_levels=noise_levels,
            apply_binning=apply_binning,
            apply_outlier=apply_outlier,
            maxlens=maxlens,
            bin_widths=bin_widths,
            drop_rates=drop_rates,
            buffer_size=buffer_size
        )
        print("\n\nData loader ready.")
    except ValueError as e:
        print(f"\n\nError creating data loader: {e}")
        traceback.print_exc()
        return None , None # Return None if setup fails

    # Create data loader ONCE before the loop
    print("\n\nSetting up data loader for validation...")
    if not path_to_val or not os.path.exists(path_to_val):
         print("\n\nWarning: Validation path not provided or does not exist. Skipping validation.")
         valid_loader = None
    else:
        try:
            valid_loader = contrastive_data_loader(
                source=path_to_val,
                n_views=n_views, 
                seed=np.random.randint(1024), # Use different seed maybe?
                batch_size=batch_size,
                apply_white_noise=apply_white_noise,
                noise_levels=noise_levels,
                apply_binning=apply_binning,
                apply_outlier=apply_outlier,
                maxlens=maxlens,
                bin_widths=bin_widths,
                drop_rates=drop_rates,
                buffer_size=max(buffer_size // 4, 10) # Smaller buffer for validation
            )
            print("\n\nValidation loader ready.")
        except ValueError as e:
            print(f"\n\nError creating validation loader: {e}")
            traceback.print_exc()
            valid_loader = None, None # Proceed without validation if loader fails

     # --- End Data Loaders ---
    # Training Loop
    best_val_loss = np.inf # Initialize best loss to infinity
    es_count = 0
    epoch_wise_train_loss = []
    epoch_wise_val_loss = []
    print(f"\n\nStarting training run: {run_timestamp} for {epochs} epochs...")
    global_step_train = 0 # Separate step counters
    global_step_val = 0

    
    for epoch in range(epochs):
        step_wise_train_loss = []
        
        model.trainable = True # Ensure model is trainable
        
        pbar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        
        for step, views in enumerate(pbar_train):
            try:
                train_loss = contrastive_train_step(model, *views, temperature=temperature, optimizer=optimizer)
                step_wise_train_loss.append(train_loss.numpy()) # Get numpy value
                
            except Exception as e:
                 print(f"\n\nError during train_step (Epoch {epoch+1}, Step {step}): {e}")
                 # Decide how to handle: continue, break, etc.
                 continue # Skip batch

            # Log step-wise loss to TensorBoard
            if summary_writer and global_step_train % 1 == 0: # Log less frequently
                with summary_writer.as_default(step=global_step_train):
                    tf.summary.scalar('loss/step_train', train_loss.numpy(), description="Training loss per step")
                # summary_writer.flush()
            
            global_step_train += 1

            # Update progress bar description
            if step % 1 == 0:
                # current_lr = optimizer.lr(optimizer.iterations).numpy() if hasattr(optimizer.lr, '__call__') else optimizer.lr.numpy()
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
                pbar_train.set_postfix({'Train Loss': f'{train_loss.numpy():.4f}', 'LR': f'{current_lr:.1E}'})
        # --- End Training Epoch ---
        
        # --- Validation Epoch ---
        step_wise_val_loss = []
        if valid_loader: # Only run validation if loader exists
            model.trainable = False # Set model to non-trainable for validation
            pbar_val = tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{epochs} (Val)', leave=False)
        
            for step, views in enumerate(pbar_val):
                try:
                    val_loss = contrastive_validation_step(model, *views, temperature=temperature)
                    step_wise_val_loss.append(val_loss.numpy()) # Get numpy value
                    
                except Exception as e:
                    print(f"\n\nError during validation_step (Epoch {epoch+1}, Step {step}): {e}")
                    continue # Skip batch

                # Log step-wise loss to TensorBoard (optional)
                if summary_writer and global_step_val % 1 == 0:
                    with summary_writer.as_default(step=global_step_val):
                         tf.summary.scalar('loss/step_val', val_loss.numpy())
                    # summary_writer.flush()
                global_step_val += 1
            

            
        
        # End of Epoch
        # current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
        # ... (calculate mean_epoch_loss) ...
        mean_epoch_train_loss = np.nanmean(step_wise_train_loss) if step_wise_train_loss else np.inf
        epoch_wise_train_loss.append(mean_epoch_train_loss)

        mean_epoch_val_loss = np.nanmean(step_wise_val_loss) if step_wise_val_loss else np.inf
        epoch_wise_val_loss.append(mean_epoch_val_loss) # Append even if inf/nan for length consistency
        
        # current_lr = optimizer.lr(optimizer.iterations).numpy() if hasattr(optimizer.lr, '__call__') else optimizer.lr.numpy()
        current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
        
        # Print Epoch Summary
        val_loss_str = f"{mean_epoch_val_loss:.5f}" if valid_loader else "N/A"
        print(f"\nEpoch {epoch + 1}/{epochs} -> Train Loss: {mean_epoch_train_loss:.5f} | Val Loss: {val_loss_str} | LR: {current_lr:.4E}")
        # Log epoch metrics to TensorBoard
        # Log epoch metrics to TensorBoard
        if summary_writer:
            with summary_writer.as_default(step=epoch):
                tf.summary.scalar('loss/epoch_train', mean_epoch_train_loss)
                if valid_loader:
                     tf.summary.scalar('loss/epoch_val', mean_epoch_val_loss)
                tf.summary.scalar('learning_rate', current_lr)
            summary_writer.flush()
        
       

        
        # --- Checkpointing and Early Stopping based on Validation Loss ---
        current_best_metric = mean_epoch_val_loss if valid_loader else mean_epoch_train_loss # Use train loss if no validation
        # Early Stopping Check
        if current_best_metric < best_val_loss: # Compare with best_val_loss tracker
            status_prefix = f"Val loss" if valid_loader else f"Train loss"
            print(f"  {status_prefix} improved from {best_val_loss:.5f} to {current_best_metric:.5f}.")
            best_val_loss = current_best_metric # Update best loss tracker
            es_count = 0
            # Optional: Save model weights
            # model.save_weights('best_contrastive_model.weights.h5')
            # model.save_weights(os.path.join(path_to_save, 'weights.h5'))
            # Save the weights *only if improvement* occurs TO THE UNIQUE FILENAME
            if best_weights_path:
                print(f"  Saving best weights for this run to {best_weights_path}...")
                try:
                    model.save_weights(best_weights_path) # Overwrites previous best FOR THIS RUN
                    print("  Weights saved successfully.")
                except Exception as e:
                    print(f"  Error saving weights: {e}")
        else:
            # Stop early stopping if validation isn't available or loss is inf/nan
            if valid_loader and np.isfinite(mean_epoch_val_loss):
                es_count += 1
                print(f"  Val loss did not improve. Early stopping count: {es_count}/{patience}")
            elif not valid_loader and np.isfinite(mean_epoch_train_loss):
                es_count += 1 # Can still early stop on train loss if no validation
                print(f"  Train loss did not improve. Early stopping count: {es_count}/{patience}")
            else:
                print("  Loss is inf/nan or validation unavailable, skipping early stopping count.")

        if es_count >= patience:
            print(f'\n\n[INFO] Early Stopping Triggered after {epoch + 1} epochs.')
            break
    
    if summary_writer:
        summary_writer.close()
    
    print("Training finished.")

    if best_weights_path and os.path.exists(best_weights_path):
         print(f"\n\nBest weights saved at: {best_weights_path} (Best Val Loss: {best_val_loss:.5f})")
    elif path_to_save:
         print("\n\nNo weights were saved (either no improvement found or path issue).")

    return epoch_wise_train_loss, epoch_wise_val_loss # Return both histories

