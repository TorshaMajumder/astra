
import tensorflow as tf
import os
import logging
from dart.src.transformer import AstroTransformer, train

# path_to_read = "/media3/majumder/dataset/cepheids/val/"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Main Execution ---

    # Redefine parameters if needed (moved some defaults into train function)
    temperature = 0.01 # Often lower temperature works better
    patience = 10 # Adjust as needed
    epochs = 5 # Train longer
    # lr = 1e-4 # Use initial_lr in train function if not using schedule
    batch_size=200 # Adjust based on GPU memory

    # Aug parameters
    apply_white_noise = (False, True, True) # Anchor, Positive, Negative
    # noise_levels = (0.0, 0.1, 0.2) # Noise level for each view
    apply_binning = (False, True, True) # Apply binning? (Masking based on time bins)
    apply_outlier = (False, False, True) # Apply photometric outlier?
    maxlens = (400, 200, 400) # Sequence lengths for Anchor, Positive, Negative
    bin_widths = (5, 5, 5) # Bin width in days for binning augmentation
    drop_rates = (0.0, 0.10, 0.60) # Fraction of bins/data to drop for binning/masking
    path_to_read = "/media3/majumder/dataset/cepheids/train/" # Make sure this path is correct and mounted
    path_to_val = "/media3/majumder/dataset/cepheids/val/" # Make sure this path is correct and mounted
    path_to_save = "/media3/majumder/CL_results/" # Save path for model checkpoints

    # Model Parameters
    num_layers = 4 # Deeper model?
    d_model = 256 # Smaller embed dim?
    num_heads = 4
    dff = 1024 # Feed-forward dim
    rate = 0.1 # Dropout rate
    use_band_info = True
    projection_dim = 128 # Add projection head (recommended for SimCLR)

    # Instantiate Model
    model = AstroTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=rate,
        use_band_info=use_band_info,
        projection_dim=projection_dim # Pass projection dim
    )

    # Dummy call to build the model (optional but good practice)
    # Need example input shapes - derive from maxlens
    dummy_anchor_input = {
        'input': tf.zeros((1, maxlens[0], 1), dtype=tf.float32),
        'times': tf.zeros((1, maxlens[0], 1), dtype=tf.float32),
        'band_info': tf.zeros((1, maxlens[0], 1), dtype=tf.float32),
        'mask': tf.zeros((1, maxlens[0]), dtype=tf.float32) # Mask shape (batch, seq_len)
    }
    _ = model(dummy_anchor_input)
    model.summary()


    # Start Training
    train_loss_history,  val_loss_history = train(
                                                    model,
                                                    path_to_read=path_to_read,
                                                    path_to_val=path_to_val,
                                                    path_to_save=path_to_save,
                                                    batch_size=batch_size,
                                                    temperature=temperature,
                                                    patience=patience,
                                                    epochs=epochs,
                                                    # initial_lr=lr, # Only if use_custom_schedule=False
                                                    use_custom_schedule=True, # Use AdamW with schedule
                                                    warmup_steps=4000, # Standard warmup
                                                    apply_white_noise=apply_white_noise,
                                                    # noise_levels=noise_levels,
                                                    apply_binning=apply_binning,
                                                    apply_outlier=apply_outlier,
                                                    maxlens=maxlens,
                                                    bin_widths=bin_widths,
                                                    drop_rates=drop_rates,
                                                    buffer_size=10000
                                                )

if __name__ == "__main__":
    main()