import os
import json
import yaml
import psutil
import mlflow
import pprint
import logging
import datetime 
import argparse
import tensorflow as tf
from astra.src.transformer import AstraNet, contrastive_train
from astra.src.scheduler import CustomSchedule, warmup_schedule


# ==========================================================
# CONFIGURE GPU MEMORY ALLOCATION
# MUST be done at the start of the program!
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print("GPUs are available. Setting memory growth to True.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # This will happen if GPUs are already initialized.
        # It's okay if this happens in an interactive session
        # after TF has already been used.
        print(f"RuntimeError setting memory growth: {e}")
# ==========================================================

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def clustered_training():
    pass

def contrastive_training(args):
    # ===============================================
    # --- Device Strategy Setup ---
    #
    # Detect available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    # If user specifies a number of GPUs, select them. Otherwise, use all available.
    if args.num_gpus is not None and args.num_gpus > 0:
        if args.num_gpus > len(gpus):
            print(f"Warning: Requested {args.num_gpus} GPUs, but only {len(gpus)} are available. Using all available.")
            gpus_to_use = gpus
        else:
            gpus_to_use = gpus[:args.num_gpus]
        
        # Make only the selected GPUs visible to TensorFlow
        tf.config.experimental.set_visible_devices(gpus_to_use, 'GPU')
        print(f"Using {len(gpus_to_use)} specified GPU(s).")
    
    # Re-list visible devices after setting them
    visible_gpus = tf.config.get_visible_devices('GPU')
    if visible_gpus:
        # If GPUs are available and visible, use MirroredStrategy for data parallelism.
        # This will handle distributing data and syncing gradients automatically.
        strategy = tf.distribute.MirroredStrategy()
        print(f"Running in GPU mode with MirroredStrategy on {len(visible_gpus)} device(s).")
    else:
        # If no GPUs are found, run on CPU.
        # We get a "default" strategy that does nothing special.
        strategy = tf.distribute.get_strategy()
        print("No GPUs found. Running in CPU mode.")
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"Available CPU cores: Physical={physical_cores}, Logical={logical_cores}")
        # # Set the number of threads for intra-operation parallelism
        # This controls parallelism within a single op (e.g., matrix multiplication)
        num_intra_threads = 10
        tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)

        # Start with 0 or a small number like 2. Setting both high can sometimes cause contention.
        num_inter_threads = 0 # Let TF decide, or try a small number like 2
        tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)


    # Get the global batch size. The strategy will automatically split this
    # across the available replicas (GPUs).
    # For example, with 8 GPUs, a global batch size of 1024 means each GPU
    # gets a per-replica batch of 128.
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size} (Per-replica: {args.batch_size} x {strategy.num_replicas_in_sync} replicas)")
    #
    # Remove MLflow logging before packaging
    #
    # Initialize MLflow Tracking
    # Set an URI and Experiment name for MLflow
    #
    mlflow.set_tracking_uri("http://127.0.0.1:39105")
    mlflow.set_experiment("Pre-training-AstraNet-Experiments-Test")
    # ===============================================
    # Load the YAML configuration file
    #
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return

    # Override config with command-line arguments if they were provided
    # This loop checks if any command-line argument was given a value (is not None)
    # and updates the config dictionary with it.
    # ==============================================
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    # ================ SKIP ==============================
    # --- You now have your final configuration in the `config` dictionary ---
    # print("--- Final Configuration ---")
    # pprint.pprint(config)
    # print("-------------------------")
    # ===============================================
    #
    #
    # --- Setup Paths and TensorBoard Writer ---
    #
    #
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = None
    if config['path_to_save']:
        #
        # Create a subdirectory for this specific run to hold weights AND TensorBoard logs
        #
        run_log_dir = os.path.join(config['path_to_save'], f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(run_log_dir)
        print(f"\n\nSaved all the parameters in: {run_log_dir}")
    # ===============================================
    # ---  Collect and Log Hyperparameters ---
    #
    hparams = {
            "run_timestamp": run_timestamp,
            "model_params": {
                "n_views": config['n_views'], 
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
    # ===============================================
    #
    # Log hyperparameters to TensorBoard
    #
    if summary_writer: # Log hparams if writer exists
        # Convert the dictionary to a nicely formatted string
        hparams_string = f"<pre>{json.dumps(hparams, indent=1)}</pre>"
        with summary_writer.as_default(step=0):
            # Log as text summary
            tf.summary.text("hyperparameters", hparams_string, description="Hyperparameters for this run")
        summary_writer.flush() # Write immediately
        summary_writer.close() # Close the writer to free resources
    # ===============================================
    # Remove MLflow logging before packaging
    # Change the "run_name" to the format - {run_timestamp}_server_name"
    # --- Start MLflow Run ---
    #
    with mlflow.start_run(run_name=f"{run_timestamp}_NVIDIA") as run:
        #
        # Add a tag for easier filtering (optional but good practice)
        mlflow.set_tag("model_type", "AstraNet")
        # ===============================================
        # Change the "run_name" to the format - {run_timestamp}_server_name"
        #
        print(f"\n\nStarted MLflow Run: {run.info.run_id}/ run_name: {run_timestamp}_NVIDIA\n\n")
        # ===============================================
        # Instantiate Model
        # --- Use the strategy scope to create the model and optimizer ---
        with strategy.scope():
            # ===============================================
            # Instantiate Model (SAME AS BEFORE, just indented)
            model = AstraNet(
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

            # Instantiate Optimizer inside the scope
            if config['use_custom_schedule']:
                # Note: You need d_model from hparams here
                d_model = hparams["model_params"]["d_model"] 
                warmup_steps = hparams["training_params"]["warmup_steps"]
                custom_lr = CustomSchedule(d_model, warmup_steps=warmup_steps)
                optimizer = tf.keras.optimizers.Adam(learning_rate=custom_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=config['initial_lr'])
            # --- END OF CHANGE ---
        
            #
            # Dummy call to build the model (optional but good practice)
            # Need example input shapes - derive from maxlens
            # --- Build the model with a dummy call (still recommended) ---
            print("\n\nBuilding model with dummy input...\n\n")
            # Use the sum of the maxlens as the sequence length for the dummy input
            build_seq_len = tf.cast(sum(config['maxlens'][0].values()), tf.int32)  # Final fixed length for sequences
            dummy_input = {
                'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
                'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
                'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
                'mask': tf.zeros((1, build_seq_len), dtype=tf.float32) # Mask shape (batch, seq_len)
            }
            # Perform the dummy call
            _ = model(dummy_input,  training=False)
            # Print the model summary
            model.summary()
            # ===============================================
        #
        #
        # Start Training
        #
        # --- Contrastive Training ---
        train_loss_history,  val_loss_history = contrastive_train(
                                                        model=model,
                                                        strategy=strategy,
                                                        optimizer=optimizer,
                                                        build_seq_len=build_seq_len,
                                                        path_to_read=hparams["data_params"]["path_to_read"],
                                                        path_to_val=hparams["data_params"]["path_to_val"],
                                                        path_to_save=hparams["data_params"]["path_to_save"],
                                                        n_views=hparams["model_params"]["n_views"],
                                                        global_batch_size=global_batch_size,
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
                                                        buffer_size=hparams["data_params"]["buffer_size"]
                                                    )


        # ----- Log all parameters from the dictionary ---
        # This will log everything from your YAML file
        mlflow.log_params(hparams)
        # ===============================================
        # print("\n\nRun logged to MLflow.")
        #
        #
        # ================ END ==========================



def main():
    # 1. Set up the Argument Parser
    # We only need ONE argument now: the path to the config file.
    parser = argparse.ArgumentParser(prog='astra-net',
                                    description="Train AstraNet model")
    
    # The loss argument is required.
    parser.add_argument('--loss', type=str, required=True, help='Provide the loss function as contrastive or clustering.')
    # The config file argument is required.
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    
    # We can also add arguments here that we might want to override frequently.
    # For example, to quickly run a test for fewer epochs.
    parser.add_argument('--epochs', type=int, help='Override the number of epochs from the config file.')
    parser.add_argument('--batch_size', type=int, help='Provide per-GPU batch_size or batch_size for CPU. Overrides the batch size from the config file.')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs to use for training. Default set to 0 for CPU mode.')

    args = parser.parse_args()

    if args.loss == "contrastive":
        contrastive_training(args)
    
    elif args.loss == "clustering":
        clustered_training(args)
    
    else:
        pass 

    

if __name__ == '__main__':
    main()