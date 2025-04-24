


import os
import logging
import traceback
import datetime # Import datetime
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from dart.src.encoder   import Encoder
from dart.src.loss import nt_xent_loss_3views 
from tensorflow.keras.optimizers import Adam
from dart.src.preprocessing import contrastive_data_loader
from dart.src.scheduler import CustomSchedule, warmup_schedule


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')


class AstroTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1,
                 base=10000.0, use_res=True, use_band_info=True,
                 use_drop=False, mjd=True, projection_dim=None, **kwargs):
        super(AstroTransformer, self).__init__(**kwargs)



        self.mjd = mjd
        self.base = base
        self.d_model = d_model
        self.use_drop = use_drop
        self.use_band_info = use_band_info

        # Embedding layers
        self.input_dense = layers.Dense(d_model, name="input_embedding") # Embed magnitude feature
        self.pos_encoding_layer = self.build_positional_encoding() # Precompute or build layer

        if self.use_band_info:
            # Embed band info (log frequency)
            self.band_dense = layers.Dense(d_model, name="band_embedding")



        self.dropout = layers.Dropout(rate)
        # self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate, name='encoder')
        self.pooling = layers.GlobalAveragePooling1D(name='avg_pooling')

        # self.embedding = TimeSeriesEmbedding(d_model=d_model, base=base, rate=rate, use_band_info=use_band_info, use_drop=use_drop, mjd=mjd)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate, use_res, name='encoder')

        # Encoder(num_layers=2, d_model=512, num_heads=4, dff=2048, rate=0.1, base=10000.0, use_res=True)
        # self.decoder = ProjectionHead(1, name='ProjectionHead')
        # self.dense = tf.keras.layers.Dense(d_model, activation='relu')
        # self.dense1 = layers.Dense(d_model,activation=None)
        # self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        # Alternative: Max Pooling/Attention-Based Pooling
        # self.decoder = tf.keras.layers.Dense(1)
        # Optional Projection Head (SimCLR style)
        self.projection_head = None
        if projection_dim:
            self.projection_head = tf.keras.Sequential([
                layers.Dense(d_model, activation='relu', name='projection1'), # Project back to d_model
                layers.Dense(projection_dim, name='projection2') # Final projection dim
            ], name='projection_head')


    def build_positional_encoding(self):
        # Using fixed sinusoidal encoding based on indices, assuming MJD values are too large/sparse
        # If MJD-based PE is desired, uncomment the MJD logic below
        # For index-based PE, we need a max sequence length. Let's assume a reasonable upper bound.
        # Or create dynamically? For now, let's stick to the time-based approach.

        def positional_encoding(times):
            """
            Calculates positional encoding. This is implemented as in the original Transformer paper.
            Follow the link: http://nlp.seas.harvard.edu/2018/04/03/attention.html

            Parameters:
            -----------------------------------------------------------------
                times (tf.Tensor): Time values.

            Returns:
            -----------------------------------------------------------------
                tf.Tensor: Positional encoding tensor.
            """


            if self.mjd:
                indices = times
            else:
                #
                # If MJD is False then the timestep will be np.arange(0, times.shape[1]/seq_len)
                #
                indices = tf.tile(tf.expand_dims(tf.range(tf.shape(times)[1], dtype=times.dtype), 0), [tf.shape(times)[0], 1])
                indices = tf.expand_dims(indices, 2)

            angle_rates = tf.exp((2.0*(tf.range(self.d_model, dtype=times.dtype)//2)) * (-tf.math.log(tf.cast(self.base, dtype=times.dtype))/tf.cast(self.d_model, times.dtype)))
            angle_rates = angle_rates[tf.newaxis, tf.newaxis, :]
            angle_rads = indices * angle_rates
            #
            # Use SIN and COSINE function for even and odd indices
            #
            # angle_rads = tf.where(tf.math.floormod(tf.range(self.d_model), 2) == 0,
            #                       tf.sin(angle_rads[:, :, :]),
            #                       tf.cos(angle_rads[:, :, :]))
            # Apply sin to even indices in the array; 2i
            sines = tf.sin(angle_rads[:, :, 0::2])
            # Apply cos to odd indices in the array; 2i+1
            cosines = tf.cos(angle_rads[:, :, 1::2])

            # Interleave sines and cosines
            # Get shape of angle_rads
            pos_encoding = tf.reshape(
                tf.stack([sines, cosines], axis=-1),
                [tf.shape(times)[0], tf.shape(times)[1], self.d_model]
            )

            # Handle odd d_model dimension if necessary (by padding or adjusting range)
            if self.d_model % 2 != 0:
                # Simple approach: repeat last element or handle based on original paper
                # For now, ensure d_model is even or handle this case explicitly
                pass # Assuming d_model is even for simplicity



            return tf.cast(pos_encoding, dtype=times.dtype)

        return positional_encoding


    def call(self, x, training=False):
        # x = self.embedding(x['input'], x['times'], x['band_info'])
        # x is a dictionary: {'input', 'times', 'band_info', 'mask'}
        input_seq = x['input'] # (batch, seq_len, 1)
        times = x['times']     # (batch, seq_len, 1)
        mask = x['mask']       # (batch, seq_len) - ensure last dim is squeezed
        # 1. Input Embedding (Magnitude)
        embeddings = self.input_dense(input_seq) # (batch, seq_len, d_model)
        # embeddings *= tf.math.sqrt(tf.cast(self.d_model, embeddings.dtype)) # Scaling often helps

        # 2. Add Positional Encoding
        pos_encoding = self.pos_encoding_layer(times)
        embeddings += pos_encoding
        # 3. Add Band Information (Segment Embedding)
        if self.use_band_info and x.get('band_info') is not None:
            band_info = x['band_info'] # (batch, seq_len, 1)
            band_embeddings = self.band_dense(band_info)
            embeddings += band_embeddings
        # 4. Apply Dropout
        embeddings = self.dropout(embeddings, training=training)
        enc_output = self.encoder(embeddings, mask, training=training) # (batch, seq_len, d_model)

        # 6. Pooling
        # print(enc_output.shape, mask.shape)
        pooled_output = self.pooling(enc_output,mask=tf.logical_not(tf.cast(mask, tf.bool))) # Pool only non-masked steps

        # 7. Optional Projection Head
        if self.projection_head:
            projected_output = self.projection_head(pooled_output, training=training)
            return projected_output # Return projected output for loss calculation
        else:
            return pooled_output # Return pooled encoder output

@tf.function
def train_step(model, anchor_batch, positive_batch, negative_batch, temperature, optimizer):
    # anchor_batch, positive_batch, negative_batch are dictionaries from the data loader
    batch_size = tf.shape(list(anchor_batch.values())[0])[0] # Get batch size from first tensor

    with tf.GradientTape() as tape:
        # Get embeddings for anchor and positive views (used in loss)
        # Pass training=True
        # z_anchor = model(anchor_batch, training=True)   # (batch_size, proj_dim or d_model)
        # z_positive = model(positive_batch, training=True) # (batch_size, proj_dim or d_model)

        # # Get embedding for negative view (ensures encoder learns from it too, even if not in loss)
        # # We detach gradients for this forward pass IF we don't want the negative example
        # # characteristics to directly influence the gradient via the loss term.
        # # However, since the weights are shared, computing it normally is fine and simpler.
        # _ = model(negative_batch, training=True) # Compute but ignore output for loss

        z_anchor = model(anchor_batch, training=True)   # (B, D)
        z_positive = model(positive_batch, training=True) # (B, D)
        z_negative = model(negative_batch, training=True) # (B, D)

        # Calculate NT-XENT loss between anchor and positive views
        loss = nt_xent_loss_3views(z_anchor, z_positive, z_negative, temperature)

        # Handle potential NaN/Inf loss
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            tf.print("Warning: Loss is NaN or Inf. Setting loss to 0.0 for this step.", loss)
            loss = tf.constant(0.0, dtype=tf.float32) # Use float32

    # Calculate and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Optional: Gradient Clipping
    # grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss
       


@tf.function
def validation_step(model, anchor_batch, positive_batch, negative_batch, temperature):
   
    z_anchor = model(anchor_batch, training=False)   # (B, D)
    z_positive = model(positive_batch, training=False) # (B, D)
    z_negative = model(negative_batch, training=False) # (B, D)

    # Calculate NT-XENT loss between anchor and positive views
    loss = nt_xent_loss_3views(z_anchor, z_positive, z_negative, temperature)

    # Handle potential NaN/Inf loss
    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
        tf.print("Warning: Loss is NaN or Inf. Setting loss to 0.0 for this step.", loss)
        loss = tf.constant(0.0, dtype=tf.float32) # Use float32

    return loss


def train(model,
          path_to_read="",
          path_to_val="",
          path_to_save=None,
          batch_size=50,
          temperature=0.1, # Default from SimCLR
          patience=20,
          epochs=10, # Increased default epochs
          initial_lr=1e-3, # For Adam optimizer if not using custom schedule
          use_custom_schedule=True,
          warmup_steps=4000,
          apply_white_noise=(False, True, True),
          # noise_levels=(0.0, 0.1, 0.1), # Example noise levels
          apply_binning=(False, False, True),
          apply_outlier=(False, False, True),
          maxlens=(200, 100, 200),
          bin_widths=(5, 5, 5),
          drop_rates=(0.0, 0.30, 0.60),
          buffer_size=100, # Shuffle buffer size
         ):

    # --- Setup Paths and TensorBoard Writer ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = None
    best_weights_path = None
    summary_writer = None

    if path_to_save:
        if not os.path.isdir(path_to_save):
             print(f"Warning: Save directory '{path_to_save}' does not exist. Attempting to create it.")
             try:
                 os.makedirs(path_to_save, exist_ok=True)
             except OSError as e:
                 print(f"Error: Could not create save directory '{path_to_save}'. {e}")
                 path_to_save = None # Disable saving
    
    if path_to_save:
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        run_log_dir = os.path.join(path_to_save, f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)

        weights_filename = 'best_contrastive.weights.h5' # Simpler name within run dir
        best_weights_path = os.path.join(run_log_dir, weights_filename)

        # Create TensorBoard writer
        summary_writer = tf.summary.create_file_writer(run_log_dir)
        print(f"Run Directory (Weights & TensorBoard Logs): {run_log_dir}")
        print(f"Will save best weights to: {best_weights_path}")

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
            seed=np.random.randint(1024), # Use different seed maybe?
            batch_size=batch_size,
            apply_white_noise=apply_white_noise,
            # noise_levels=noise_levels,
            apply_binning=apply_binning,
            apply_outlier=apply_outlier,
            maxlens=maxlens,
            bin_widths=bin_widths,
            drop_rates=drop_rates,
            buffer_size=buffer_size
        )
        print("Data loader ready.")
    except ValueError as e:
        print(f"Error creating data loader: {e}")
        traceback.print_exc()
        return None # Return None if setup fails

    # Create data loader ONCE before the loop
    print("Setting up data loader for validation...")
    if not path_to_val or not os.path.exists(path_to_val):
         print("Warning: Validation path not provided or does not exist. Skipping validation.")
         valid_loader = None
    else:
        try:
            valid_loader = contrastive_data_loader(
                source=path_to_val,
                seed=np.random.randint(1024), # Use different seed maybe?
                batch_size=batch_size,
                apply_white_noise=apply_white_noise,
                # noise_levels=noise_levels,
                apply_binning=apply_binning,
                apply_outlier=apply_outlier,
                maxlens=maxlens,
                bin_widths=bin_widths,
                drop_rates=drop_rates,
                buffer_size=max(buffer_size // 4, 10) # Smaller buffer for validation
            )
            print("Validation loader ready.")
        except ValueError as e:
            print(f"Error creating validation loader: {e}")
            traceback.print_exc()
            valid_loader = None # Proceed without validation if loader fails

     # --- End Data Loaders ---
    # Training Loop
    best_val_loss = np.inf # Initialize best loss to infinity
    es_count = 0
    epoch_wise_train_loss = []
    epoch_wise_val_loss = []
    # best_weights_path = None
    print(f"Starting training run: {run_timestamp} for {epochs} epochs...")
    global_step_train = 0 # Separate step counters
    global_step_val = 0

    
    for epoch in range(epochs):
        step_wise_train_loss = []
        
        model.trainable = True # Ensure model is trainable
        
        pbar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        # pbar = tqdm(enumerate(train_loader), total=?) # If you know steps per epoch

        for step, (anchor, positive, negative) in enumerate(pbar_train):
            try:
                train_loss = train_step(model, anchor, positive, negative, temperature, optimizer)
                step_wise_train_loss.append(train_loss.numpy()) # Get numpy value
                # train_loss_step = train_loss.numpy() # Get loss for current step
            except Exception as e:
                 print(f"\nError during train_step (Epoch {epoch+1}, Step {step}): {e}")
                 # Decide how to handle: continue, break, etc.
                 continue # Skip batch

            # Log step-wise loss to TensorBoard
            if summary_writer and global_step_train % 20 == 0: # Log less frequently
                with summary_writer.as_default(step=global_step_train):
                    tf.summary.scalar('train_loss_step', train_loss.numpy(), description="Training loss per step")
            
            global_step_train += 1

            # Update progress bar description
            if step % 20 == 0:
                # current_lr = optimizer.lr(optimizer.iterations).numpy() if hasattr(optimizer.lr, '__call__') else optimizer.lr.numpy()
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy() if hasattr(optimizer.learning_rate, '__call__') else optimizer.learning_rate.numpy()
                pbar_train.set_postfix({'train_loss': f'{train_loss.numpy():.4f}', 'lr': f'{current_lr:.1E}'})
        # --- End Training Epoch ---
        
        # --- Validation Epoch ---
        step_wise_val_loss = []
        if valid_loader: # Only run validation if loader exists
            model.trainable = False # Set model to non-trainable for validation
            pbar_val = tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{epochs} (Val)', leave=False)
        
            for step, (anchor, positive, negative) in enumerate(pbar_val):
                try:
                    val_loss = validation_step(model, anchor, positive, negative, temperature)
                    step_wise_val_loss.append(val_loss.numpy()) # Get numpy value
                    # val_loss_step = val_loss.numpy() # Get loss for current step
                except Exception as e:
                    print(f"\nError during validation_step (Epoch {epoch+1}, Step {step}): {e}")
                    continue # Skip batch

                # Log step-wise loss to TensorBoard (optional)
                if summary_writer and global_step_val % 20 == 0:
                    with summary_writer.as_default(step=global_step_val):
                         tf.summary.scalar('val_loss', val_loss.numpy())
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
            print(f'\n[INFO] Early Stopping Triggered after {epoch + 1} epochs.')
            break
    
    if summary_writer:
        summary_writer.close()
    
    print("Training finished.")

    if best_weights_path and os.path.exists(best_weights_path):
         print(f"Best weights saved at: {best_weights_path} (Best Val Loss: {best_val_loss:.5f})")
    elif path_to_save:
         print("No weights were saved (either no improvement found or path issue).")

    return epoch_wise_train_loss, epoch_wise_val_loss # Return both histories

