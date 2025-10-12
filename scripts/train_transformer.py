

import os
import json
import yaml
import pprint
import logging
import datetime 
import argparse
import tensorflow as tf
from astra.src.transformer import AstroTransformer, contrastive_train


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


def clustered_training():
    pass

def contrastive_training(args):
    
    # 2. Load the YAML configuration file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return

    # 3. Override config with command-line arguments if they were provided
    # This loop checks if any command-line argument was given a value (is not None)
    # and updates the config dictionary with it.
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value

    # --- You now have your final configuration in the `config` dictionary ---
    # print("--- Final Configuration ---")
    # pprint.pprint(config)
    # print("-------------------------")


    # --- Setup Paths and TensorBoard Writer ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = None
    if config['path_to_save']:
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        run_log_dir = os.path.join(config['path_to_save'], f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(run_log_dir)
        print(f"\n\nSaved all the parameters in: {run_log_dir}")

    # --- 1. Collect and Log Hyperparameters ---
    hparams = {
            "run_timestamp": run_timestamp,
            "model_params": {
                "num_layers": config['num_layers'], "d_model": config['d_model'], "num_heads": config['num_heads'],
                "dff": config['dff'], "projection_dim": config['projection_dim'], "rate": config['rate'], "mjd": config['mjd'],
                "use_band_info": config['use_band_info'], "base": config['base'], "use_drop": config['use_drop']
            },
            "training_params": {
                "epochs": config['epochs'], "patience": config['patience'], "initial_lr": config['initial_lr'],
                "use_custom_schedule": config['use_custom_schedule'], "warmup_steps": config['warmup_steps'],
                "temperature": config['temperature'], "batch_size": config['batch_size']
            },
            "data_params": {
                "buffer_size": config['buffer_size'],
                "apply_white_noise": config['apply_white_noise'], 
                "noise_levels": config['noise_levels'],
                "apply_binning": config['apply_binning'], 
                "apply_outlier": config['apply_outlier'],
                "maxlens": config['maxlens'], "bin_widths": config['bin_widths'], 
                "drop_rates": config['drop_rates'],
                "path_to_read": config['path_to_read'], "path_to_val": config['path_to_val'], 
                "path_to_save": run_log_dir
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
    build_seq_len = sum(config['maxlens'][0].values()) # e.g., 12 or 200 depending on strategy
    dummy_input = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'mask': tf.zeros((1, build_seq_len), dtype=tf.float32) # Mask shape (batch, seq_len)
    }

    _ = model(dummy_input,  training=False)
    model.summary()

    # Start Training
    train_loss_history,  val_loss_history = contrastive_train(
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



def main():
    # 1. Set up the Argument Parser
    # We only need ONE argument now: the path to the config file.
    parser = argparse.ArgumentParser(prog='astra-transformer',
                                    description="Train ASTRA transformer model")
    
    # The config file argument is required.
    parser.add_argument('--loss', type=str, required=True, help='Provide the loss function as contrastive or clustering.')
    # The config file argument is required.
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    
    # We can also add arguments here that we might want to override frequently.
    # For example, to quickly run a test for fewer epochs.
    parser.add_argument('--epochs', type=int, help='Override the number of epochs from the config file.')
    parser.add_argument('--batch_size', type=int, help='Override the batch size from the config file.')

    args = parser.parse_args()

    if args.loss == "contrastive":
        contrastive_training(args)
    
    elif args.loss == "clustering":
        clustered_training(args)
    
    else:
        pass

    

if __name__ == '__main__':
    main()