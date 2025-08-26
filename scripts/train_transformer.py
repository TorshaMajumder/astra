
import tensorflow as tf
import os
import json
import datetime # Import datetime
import logging
from astra.src.transformer import AstroTransformer, train
from astra.bands.bands import ztf_band

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
    patience = 100 # Adjust as needed
    epochs = 20 # Train longer
    # lr = 1e-4 # Use initial_lr in train function if not using schedule
    batch_size=300 # Adjust based on GPU memory

    # Aug parameters
    apply_white_noise = (True, False, True) # Anchor, Positive, Negative
    # noise_levels = (0.0, 0.1, 0.2) # Noise level for each view
    apply_binning = (False, False, True) # Apply binning? (Masking based on time bins)
    apply_outlier = (False, False, True) # Apply photometric outlier?
    maxlens = (300, 150, 300) # Sequence lengths for Anchor, Positive, Negative
    bin_widths = (5, 5, 5) # Bin width in days for binning augmentation
    drop_rates = (0.0, 0.0, 0.50) # Fraction of bins/data to drop for binning/masking
    noise_levels = (0.10, 0.0, 0.10) # Fraction of bins/data to drop for binning/masking
    path_to_read = "/media3/majumder/dataset/lyrae_cep/train/" # Make sure this path is correct and mounted
    path_to_val = "/media3/majumder/dataset/lyrae_cep/val/" # Make sure this path is correct and mounted
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
    initial_lr = 1e-4 # Initial learning rate if not using schedule
    use_custom_schedule = True # Use learning rate schedule with AdamW
    warmup_steps = 4000 # Warmup steps for learning rate schedule
    buffer_size = 10000 # Buffer size for shuffling

    # --- Setup Paths and TensorBoard Writer ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = None
    if path_to_save:
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        run_log_dir = os.path.join(path_to_save, f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(run_log_dir)
        print(f"\n\nSaved all the parameters in: {run_log_dir}")
    # --- 1. Collect and Log Hyperparameters ---
    hparams = {
            "run_timestamp": run_timestamp,
            "model_params": {
                "num_layers": num_layers, "d_model": d_model, "num_heads": num_heads,
                "dff": dff, "projection_dim": projection_dim, "rate": rate, "mjd": mjd,
                "use_band_info": use_band_info, "base": base, "use_drop": use_drop
            },
            "training_params": {
             "epochs": epochs, "patience": patience, "initial_lr": initial_lr,
             "use_custom_schedule": use_custom_schedule, "warmup_steps": warmup_steps,
             "temperature": temperature, "batch_size": batch_size,
        },
        "data_params": {
            "buffer_size": buffer_size,
            "apply_white_noise": list(apply_white_noise), # Convert tuples to lists for JSON
            "noise_levels": list(noise_levels),
            "apply_binning": list(apply_binning), "apply_outlier": list(apply_outlier),
            "maxlens": list(maxlens), "bin_widths": list(bin_widths), "drop_rates": list(drop_rates),
            "path_to_read": path_to_read, "path_to_val": path_to_val, "path_to_save": run_log_dir
        }
    }

    if summary_writer: # Log hparams if writer exists
        # Convert the dictionary to a nicely formatted string
        hparams_string = f"<pre>{json.dumps(hparams, indent=1)}</pre>"
        with summary_writer.as_default(step=0):
            # Log as text summary
            tf.summary.text("hyperparameters", hparams_string, description="Hyperparameters for this run")
        summary_writer.flush() # Write immediately
        summary_writer.close() # Close the writer to free resources

    # Instantiate Model
    model = AstroTransformer(
        num_layers=hparams["model_params"]["num_layers"],
        d_model=hparams["model_params"]["d_model"],
        base=hparams["model_params"]["base"],
        num_heads=hparams["model_params"]["num_heads"],
        dff=hparams["model_params"]["dff"],
        rate=hparams["model_params"]["rate"],
        mjd=hparams["model_params"]["mjd"],
        use_drop=hparams["model_params"]["use_drop"],
        use_band_info=hparams["model_params"]["use_band_info"],
        projection_dim=hparams["model_params"]["projection_dim"] # Pass projection dim
    )

    # Dummy call to build the model (optional but good practice)
    # Need example input shapes - derive from maxlens
    # --- Build the model with a dummy call (still recommended) ---
    print("\n\nBuilding model with dummy input...")
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

    # Start Training
    train_loss_history,  val_loss_history = train(
                                                    model,
                                                    path_to_read=hparams["data_params"]["path_to_read"],
                                                    path_to_val=hparams["data_params"]["path_to_val"],
                                                    path_to_save=hparams["data_params"]["path_to_save"],
                                                    batch_size=hparams["training_params"]["batch_size"],
                                                    temperature=hparams["training_params"]["temperature"],
                                                    patience=hparams["training_params"]["patience"],
                                                    epochs=hparams["training_params"]["epochs"],
                                                    # initial_lr=lr, # Only if use_custom_schedule=False
                                                    use_custom_schedule=hparams["training_params"]["use_custom_schedule"], # Use AdamW with schedule
                                                    warmup_steps=hparams["training_params"]["warmup_steps"], # Standard warmup
                                                    apply_white_noise=hparams["data_params"]["apply_white_noise"],
                                                    noise_levels=hparams["data_params"]["noise_levels"],
                                                    apply_binning=hparams["data_params"]["apply_binning"],
                                                    apply_outlier=hparams["data_params"]["apply_outlier"],
                                                    maxlens=hparams["data_params"]["maxlens"],
                                                    bin_widths=hparams["data_params"]["bin_widths"],
                                                    drop_rates=hparams["data_params"]["drop_rates"],
                                                    buffer_size=hparams["data_params"]["buffer_size"],
                                                )

if __name__ == "__main__":
    main()