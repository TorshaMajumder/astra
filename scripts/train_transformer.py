
import tensorflow as tf
import os
import logging
from dart.src.transformer import AstroTransformer, train
from dart.bands.bands import ztf_band

try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Available CPU cores: Physical={physical_cores}, Logical={logical_cores}")
except ImportError:
    physical_cores = os.cpu_count() # Fallback, might be logical cores
    print("Install 'psutil' for accurate core counts.")
    print(f"Available CPU logical cores (estimated by os.cpu_count): {physical_cores}")

# --- Configuration for CPU Parallelism ---

# Set the number of threads for intra-operation parallelism
# This controls parallelism within a single op (e.g., matrix multiplication)
num_intra_threads = 20
tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)

# Start with 0 or a small number like 2. Setting both high can sometimes cause contention.
num_inter_threads = 0 # Let TF decide, or try a small number like 2
tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def main():

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Main Execution ---

    # Redefine parameters if needed (moved some defaults into train function)
    temperature = 0.1 # Often lower temperature works better
    patience = 10 # Adjust as needed
    epochs = 100 # Train longer
    # lr = 1e-4 # Use initial_lr in train function if not using schedule
    batch_size=250 # Adjust based on GPU memory

    # Aug parameters
    apply_white_noise = (False, True, True) # Anchor, Positive, Negative
    # noise_levels = (0.0, 0.1, 0.2) # Noise level for each view
    apply_binning = (False, False, True) # Apply binning? (Masking based on time bins)
    apply_outlier = (False, False, True) # Apply photometric outlier?
    maxlens = (200, 100, 200) # Sequence lengths for Anchor, Positive, Negative
    bin_widths = (5, 5, 5) # Bin width in days for binning augmentation
    drop_rates = (0.0, 0.0, 0.50) # Fraction of bins/data to drop for binning/masking
    noise_levels = (0.0, 0.10, 0.10) # Fraction of bins/data to drop for binning/masking
    path_to_read = "/media3/majumder/dataset/multi-class/train/" # Make sure this path is correct and mounted
    path_to_val = "/media3/majumder/dataset/multi-class/val/" # Make sure this path is correct and mounted
    path_to_save = "/media3/majumder/contrastive_loss_res/" # Save path for model checkpoints

    # Model Parameters
    num_layers = 4 # Deeper model?
    mjd = True # Use MJD for positional encoding
    base = 10000.0
    d_model = 256 # Smaller embed dim?
    num_heads = 4
    dff = 1024 # Feed-forward dim
    rate = 0.1 # Dropout rate
    use_band_info = True
    use_drop = True
    projection_dim = 128 # Add projection head (recommended for SimCLR)

    # Instantiate Model
    model = AstroTransformer(
        num_layers=num_layers,
        d_model=d_model,
        base=base,
        num_heads=num_heads,
        dff=dff,
        rate=rate,
        mjd=mjd,
        use_drop=use_drop,
        use_band_info=use_band_info,
        projection_dim=projection_dim # Pass projection dim
    )

    # Dummy call to build the model (optional but good practice)
    # Need example input shapes - derive from maxlens
    # --- Build the model with a dummy call (still recommended) ---
    print("Building model with dummy input...")
    # Use the expected sequence length AFTER sliding_window for the anchor view
    build_seq_len = maxlens[0] * len(ztf_band) # e.g., 12 or 200 depending on strategy
    dummy_input = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float64),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float64),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float64),
        'mask': tf.zeros((1, build_seq_len), dtype=tf.float64) # Mask shape (batch, seq_len)
    }
    _ = model(dummy_input,  training=False)
    model.summary()

    # weights_path = "/media3/majumder/CL_results/run_20250425_085427/best_contrastive.weights.h5"
    # model.load_weights(weights_path)
    # print("Weights loaded successfully!")


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
                                                    noise_levels=noise_levels,
                                                    apply_binning=apply_binning,
                                                    apply_outlier=apply_outlier,
                                                    maxlens=maxlens,
                                                    bin_widths=bin_widths,
                                                    drop_rates=drop_rates,
                                                    buffer_size=10000
                                                )

if __name__ == "__main__":
    main()