# =========================================================
# Import all dependencies
# =========================================================
import os
import umap
import yaml
import mlflow
import psutil
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
import tensorflow as tf
import plotly.express as px
from astra.src.transformer import AstraNet
from astra.src.preprocessing import create_inference_loader
from astra.utils.helper import load_hparams_from_event_file
# ==========================================================
# CONFIGURE GPU MEMORY ALLOCATION
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print("\nGPUs are available. Setting memory growth to True.\n")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # This will happen if GPUs are already initialized.
        print(f"RuntimeError setting memory growth: {e}")
# ===========================================================
# SUPPRESS TF WARNINGS
logging.getLogger('tensorflow').setLevel(logging.ERROR)  
os.system('clear')
# ===========================================================

def generate_plot(path_to_save, model_params, mlflow_upload, mlflow_name):
    #
    # ==================================== LOAD ASTRA embeddings, metadata ==================================
    #
    # UMAP Parameters
    #
    MIN_DIST = 0.5   
    N_NEIGHBORS = 15                                      
    METRIC = 'cosine'               
    RANDOM_STATE = 42    
    #
    # Plotting parameters
    #
    POINT_SIZE = 6
    ALPHA = 0.6  
    # 
    # --- Load the Saved Embeddings and Metadata ---
    #
    print(f"\nLoading data from: {path_to_save}...")
    ids_file = os.path.join(path_to_save, 'ids.npy') 
    labels_file = os.path.join(path_to_save, 'labels.npy')
    embeddings_file = os.path.join(path_to_save, 'embeddings.npy')
    #
    # Check if all the files exists
    #
    if not all(os.path.exists(f) for f in [embeddings_file, labels_file, ids_file]):
        print(f"\nERROR: One or more .npy files (embeddings.npy, labels.npy, ids.npy) not found. \
                Please check the 'run_directory' path.")
        return
    # --------------------------------------------------------------------------------------------
    all_ids = np.load(ids_file)
    all_labels = np.load(labels_file)
    all_embeddings = np.load(embeddings_file)
    #
    # Decode labels from byte strings to regular strings
    try:
        labels_decoded = [label.decode('utf-8') for label in all_labels]
    except (UnicodeDecodeError, AttributeError):
        print("\nLabels are not byte strings, using them as is.")
        labels_decoded = all_labels
    print(f"\nLoaded {len(all_embeddings)} embeddings with d_model={all_embeddings.shape[1]}")
    print(f"\nFound {len(np.unique(labels_decoded))} unique labels.")
    #
    # ---------------------- Perform UMAP Dimensionality Reduction ----------------------------------
    #
    print(f"\nPerforming UMAP reduction (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}, metric='{METRIC}')...")
    # Initialize UMAP 2D estimator
    reducer = umap.UMAP(
                        n_neighbors=N_NEIGHBORS,
                        min_dist=MIN_DIST,
                        n_components=2,
                        metric=METRIC,
                        random_state=RANDOM_STATE
                    )
    embedding_2d = reducer.fit_transform(all_embeddings)
    print("\nUMAP reduction completed!")
    #
    # -------- Prepare Data for Plotting with Pandas and Seaborn -------
    # 
    df = pd.DataFrame()
    df['id'] = all_ids
    df['label'] = labels_decoded
    df['umap-x'] = embedding_2d[:, 0]
    df['umap-y'] = embedding_2d[:, 1]
    #
    # --- Create and Save the Plot ---
    #
    print("\nGenerating plot...")
    # -----------------------------------------------------------------------------------------------------
    fig = px.scatter(
                        df,
                        x="umap-x", 
                        y="umap-y",
                        color="label",
                        hover_data=['label', 'id'],         
                        title=f"2D-UMAP Projection of ASTRA Embeddings (d_model={model_params["d_model"]})"
                    )
    fig.update_layout(
                        xaxis_title='UMAP Dimension 1',
                        yaxis_title='UMAP Dimension 2',
                        legend_title_text='Classes',
                        font=dict(size=14),
                        title_font_size=18,
                        legend=dict(
                                        x=1.05, 
                                        y=1,
                                        xanchor='left',
                                        yanchor='top'
                                    )
                    )
    fig.update_traces(
                        marker=dict(
                                        size=POINT_SIZE,  
                                        opacity=ALPHA     
                                    )
                    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, showline=False)
    #
    #
    if mlflow_upload:
        # ==========================================================================================
        # (IMPORTANT): Remove MLflow logging before packaging
        #
        # Initialize MLflow Tracking
        # Set an URI and Experiment name for MLflow
        #
        mlflow.set_tracking_uri("http://127.0.0.1:37533")
        mlflow.set_experiment("Set1")
        # ===============================================
        with mlflow.start_run(run_id=f"{mlflow_name}") as run:
            # Log the Plotly figure to MLflow
            print("\nLogging interactive figure to MLflow...")
            mlflow.log_figure(fig, f"plots/umap_plot_{mlflow_name}.html")
            print("\nInteractive figure logged successfully to MLflow!")
        #
        #
        # ==================================== END OF LOGGING =======================================




def contrastive_embeddings(args):
    # ===============================================
    # ------------- Device Strategy Setup -----------
    #
    # Detect available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #
    # Use user-specified GPUs. Otherwise, use all available GPUs.
    #
    if args.num_gpus is not None and args.num_gpus > 0:
        if args.num_gpus > len(gpus):
            print(f"\nWarning: Requested {args.num_gpus} GPUs, but only {len(gpus)} are available. Using all available.\n")
            gpus_to_use = gpus
        else:
            gpus_to_use = gpus[:args.num_gpus]
        #
        # Make only the selected GPUs visible to TensorFlow
        #
        tf.config.experimental.set_visible_devices(gpus_to_use, 'GPU')
        print(f"\nUsing {len(gpus_to_use)} specified GPU(s).\n")
    else:
        # If no GPUs are found, run on CPU.
        print("\nNo GPUs found. Running in CPU mode.\n")
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"\nAvailable CPU cores: Physical={physical_cores}, Logical={logical_cores}\n")
        # Set the number of threads for intra-operation parallelism
        num_intra_threads = 20
        tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)
        # Set the number of threads for inter-operation parallelism
        num_inter_threads = 0 # Let TensorFlow decide
        tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)
    # ====================================================================================================
    # ===============================================
    # Load the YAML configuration file
    #
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\nError: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"\nError loading YAML file: {e}")
        return
    #
    # Override config with command-line arguments if they were provided
    # This loop checks if any command-line argument was given a value (is not None)
    # and updates the config dictionary with it.
    # ==============================================
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    # ==============================================
    # ====================================================================================================
    # Load the hyper-parameters of the model from the path
    #
    run_directory = config['path_to_load']
    model_params, _, _ = load_hparams_from_event_file(run_directory)
    #
    # Stop if hyperparameters could not be loaded
    #
    try:
        if model_params is None:
            raise ValueError("\n\nFailed to load hyperparameters from the event file.\nExiting...\n")
    except Exception as e:
        print(e)
        return
    #
    # --- Instantiate the Full Model using loaded hyper-params ---
    #
    print("\nRe-creating the full AstraNet architecture using loaded hyper-parameters...")
    model = AstraNet(
                    num_layers=model_params["num_layers"],
                    d_model=model_params["d_model"],
                    base=model_params["base"],
                    num_heads=model_params["num_heads"],
                    dff=model_params["dff"],
                    rate=model_params["rate"],
                    mjd=model_params["mjd"],
                    use_drop=model_params["use_drop"],
                    use_band_info=model_params["use_band_info"],
                    projection_dim=model_params["projection_dim"] 
                )
    print("\n --Model instantiated!")
    #
    # Building model with dummy input to create all variables
    # Using ANCHOR view sequence length, i.e., build_seg_len
    #
    build_seq_len = sum(config['max_len'].values()) 
    dummy_input = {
        'input': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'times': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'band_info': tf.zeros((1, build_seq_len, 1), dtype=tf.float32),
        'mask': tf.zeros((1, build_seq_len), dtype=tf.float32)
    }
    #
    # Set training=FALSE for inference
    #
    _ = model(dummy_input, training=False)
    print("\n --Full model built!")
    #
    # Load model's weight
    #
    try:
        path_to_weight = os.path.join(run_directory, 'best_contrastive.weights.h5') 
        print(f"\nSearching pre-trained weights in: {path_to_weight}...")
        model.load_weights(path_to_weight)
        print(f"\nWeights loaded successfully into the model!")
    except Exception as e:
        print(f"\nERROR: Could not load weights. Check the path to model's weight."
                    f"Ensure architecture matches exactly.\n{e}")
        return
    # ====================================================================================================
    # --- Isolate the ASTRA encoder to generate embeddings ---
    # --- Add the GlobalAveragePooling layer after ASTRA encoder ----
    # 
    print("\n --Extracting ASTRA encoder for generating embeddings...")
    #
    # Define the inputs with a fixed sequence length
    # It should match build_seq_len
    #
    input_layer = {
        'input': tf.keras.Input(shape=(build_seq_len, 1), name='input', dtype=tf.float32),
        'times': tf.keras.Input(shape=(build_seq_len, 1), name='times', dtype=tf.float32),
        'band_info': tf.keras.Input(shape=(build_seq_len, 1), name='band_info', dtype=tf.float32),
        'mask': tf.keras.Input(shape=(build_seq_len,), name='mask', dtype=tf.float32)
    }
    # ------------------------------------------------------------------------------------------------
    #
    # (STEP:1) Get the embeddings from the embedding layer 
    # The embedding layer takes the full dictionary of inputs
    #
    embeddings = model.embedding_layer(input_layer)
    #
    # Get the mask tensor from the input dictionary (IMPORTANT for encoder and pooling laye)
    # 
    mask_input = input_layer['mask']
    #
    # (STEP:2) Get the embeddings and the attention weights
    #
    encoder_output, all_attention_weights = model.encoder(embeddings, mask=mask_input)
    #
    # (STEP:3) Invert the mask using ASTRA masking logic and get the pooled output
    #
    pool_mask = tf.keras.layers.Lambda(
                lambda m: tf.logical_not(tf.cast(m, tf.bool))
                )(mask_input)
    pooled_output = model.pooling(encoder_output, mask=pool_mask)
    #
    # (STEP:4) Get the final ASTRA encoder model and Set to inference mode
    #
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=[pooled_output, all_attention_weights], name="ASTRA_Encoder")
    encoder_model.trainable = False 
    #
    print("\n --ASTRA Encoder created successfully...!\n")
    encoder_model.summary()
    # =================================================================================================================
    #
    # ------------------ Prepare the Inference Data Loader -----------------------------------
    # 
    print("\nSetting up the inference data loader...")
    inference_loader = create_inference_loader(
                                                source=config['path_to_data'],
                                                batch_size=config['batch_size'],
                                                maxlen=config['max_len']
                                            )

    # ------------------------ Generate ASTRA Embeddings ------------------------------------
    print("\nGenerating embeddings for the dataset...\n")
    #
    # Declare variables
    #
    all_ids = []
    all_labels = []
    all_embeddings = []
    #
    # Iterate through the inference loader
    #
    for batch in tqdm(inference_loader, desc="Generating Embeddings"):
        model_inputs = {
            'input': batch['input'],
            'times': batch['times'],
            'band_info': batch['band_info'],
            'mask': batch['mask']
        }
        batch_embeddings, batch_attention_weights = encoder_model(model_inputs, training=False)
        
        all_ids.append(batch['id'].numpy())
        all_labels.append(batch['label'].numpy())
        all_embeddings.append(batch_embeddings.numpy())
    # --------------------------------------------------------------------------
    # Concatenate results from all batches into single numpy arrays
    #
    all_ids = np.concatenate(all_ids, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    # Decode labels if they are byte strings
    all_labels = np.array([label.decode('utf-8') for label in all_labels])
    # ---------------------------------------------------------------------------
    assert len(all_embeddings) == len(all_labels) == len(all_ids) 
    print(f"\n --Successfully generated {len(all_embeddings)} embeddings!")
    # ---------------------------------------------------------------------------
    # ----------------------------- Save the results ----------------------------
    #
    print(f"\nSaving embeddings, metadata, and attention weights to: {run_directory}")
    try:
        np.save(os.path.join(run_directory, 'ids.npy'), all_ids)
        np.save(os.path.join(run_directory, 'labels.npy'), all_labels)
        np.save(os.path.join(run_directory, 'embeddings.npy'), all_embeddings)
        print("\n --Files saved successfully!\n")
        
    except Exception as e:
        print(f"\nERROR: Could not save the files. Check: {e}\n")
    # -------------------------------------------------------------------------------------------------
    generate_plot(config['path_to_save'], model_params, config['mlflow_upload'], config['mlflow_name'])
    # -------------------------------------------------------------------------------------------------
    
def clustered_embeddings():
    """ 
    TODO: Implement clustered embeddings function
    """
    pass

        

def main():
    # ==========================================================
    # Set up the Argument Parser
    # ==========================================================
    parser = argparse.ArgumentParser(prog='astra-embeddings',
                                        description="Generate AstraNet embeddings")
    # ==========================================================
    # Setup all required arguments
    # =========================================================
    parser.add_argument('--loss', type=str, required=True, help='Provide the loss function as contrastive or clustering.' \
                                                                ' NOTE: clustering loss not yet implemented. We currently support contrastive loss only.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    # ==========================================================
    # Optional arguments to override config file parameters
    # ==========================================================
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs to use for generating emebeddings. Default set to 0 for CPU mode.')
    parser.add_argument('--batch_size', type=int, default=500, help='Provide the batch size for the inference data.')
    parser.add_argument('--mlflow_upload', type=bool, default=False, help='Provide TRUE if you want to upload the 2D-UMAP ' \
                                                                            'ASTRA embeddings to MLflow else FALSE.')
    args = parser.parse_args()
    # ==========================================================
    if args.loss == "contrastive":
        contrastive_embeddings(args)
    
    elif args.loss == "clustering":
        clustered_embeddings(args)
    
    else:
        print("\nError: Unsupported loss function specified. Use 'contrastive' or 'clustering'.\n")


if __name__ == '__main__':
    main()