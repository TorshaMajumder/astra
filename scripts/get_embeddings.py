# =========================================================
# Import all dependencies
# =========================================================
import os
import umap
import h5py
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
import plotly.graph_objects as go
from astra.utils.helper import load_config
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

def generate_plot(path_to_save, path_to_class_count, model_params, mlflow_upload, mlflow_name, mlflow_exp):
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
    # --------------- Load the class counts from the CSV and sort them -----------------
    # the larger classes are at the TOP of the CSV so they are plotted FIRST (background)
    #
    counts_df = pd.read_csv(path_to_class_count, header=0, sep='\t', index_col=False) 
    counts_df = counts_df.sort_values('total_records', ascending=False)
    #
    # Create a mapping for easy lookup
    #
    class_to_count = dict(zip(counts_df['class_name'], counts_df['total_records']))
    sorted_class_names = counts_df['class_name'].tolist()
    #
    # 
    # --- Load the Saved Embeddings and Metadata ---
    #
    print(f"\nLoading data from: {path_to_save}...")
    try:
        with h5py.File(path_to_save, 'r') as hf:
            embeddings = hf['embeddings'][:]
            labels_raw = hf['labels'][:]
            ids = hf['ids'][:]
        labels_as_bytes = np.array(labels_raw, dtype=np.bytes_)
        labels = np.char.decode(labels_as_bytes, encoding='utf-8')
        print(f"\nSuccessfully loaded {len(embeddings)} embeddings and {len(ids)} ids...")
    except FileNotFoundError:
        print(f"\nError: HDF5 file not found at path - {path_to_save}")
        return 
    except KeyError as e:
        print(f"\nError: Dataset '{e.args[0]}' not found in the HDF5 file.")
        print("Please ensure the file contains 'embeddings', 'labels', and 'ids' datasets.")
        return
    # ------------------------------------------------------------------------------------------------
    #
    # Decode labels from byte strings to regular strings
    try:
        labels_decoded = [label.decode('utf-8') for label in labels]
    except (UnicodeDecodeError, AttributeError):
        print("\nLabels are not byte strings, using them as is.")
        labels_decoded = labels
    print(f"\nLoaded {len(embeddings)} embeddings with d_model={embeddings.shape[1]}")
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
    embedding_2d = reducer.fit_transform(embeddings)
    print("\nUMAP reduction completed!")
    #
    # -------- Prepare Data for Plotting with Pandas and Plotly -------
    # 
    df = pd.DataFrame()
    df['id'] = ids
    df['label'] = labels_decoded
    df['umap-x'] = embedding_2d[:, 0]
    df['umap-y'] = embedding_2d[:, 1]
    df['total_records'] = df['label'].map(class_to_count)
    df = df.sort_values('total_records', ascending=False)
    # 
    # Define Unique Colors for all the classes
    # 
    palette = px.colors.qualitative.Alphabet[:len(sorted_class_names)]
    color_map = {name: palette[i] for i, name in enumerate(sorted_class_names)}
    # -------------------------------------------------------------------------
    # Create Figure using Scattergl 
    # -------------------------------------------------------------------------
    print("\nGenerating plot...")
    fig = go.Figure()
    for name in sorted_class_names:
        cls_data = df[df['label'] == name]
        count = class_to_count[name]
        # ---------- Applying conditioning for opacity and marker size based on class count -----
        if count > 200000:
            opacity = 0.15
            marker_size = 1.5
        elif count > 50000:
            opacity = 0.3
            marker_size = 2.5
        elif count > 20000:
            opacity = 0.4
            marker_size = 3.5
        elif count > 5000:
            opacity = 0.65
            marker_size = 4.5
        else:
            opacity = 0.80
            marker_size = 6.0
        # ------------------------------------------------------------------------------------------
        # ------------------------------- Create and Save the Plot ---------------------------------
        # ------------------------------------------------------------------------------------------
        fig.add_trace(go.Scattergl(
                                    x=cls_data['umap-x'],
                                    y=cls_data['umap-y'],
                                    mode='markers',
                                    name=f"{name} (n={count})",
                                    marker=dict(
                                                color=color_map[name],
                                                size=marker_size,
                                                opacity=opacity
                                            ),
                                    text=cls_data['id'],
                                    hoverinfo='text+name'
                                ))
    # --------------------------------- Update Layout -------------------------------------------
    # -------------------------------------------------------------------------------------------
    fig.update_layout(
                        title=f'2D-UMAP Projection of ASTRA Embeddings (d_model={model_params["d_model"]})',
                        xaxis_title='UMAP Dimension 1',
                        yaxis_title='UMAP Dimension 2',
                        legend_title_text='Classes with counts',
                        template='plotly_white',
                        font=dict(size=14),
                        title_font_size=18,
                        legend=dict(itemsizing='constant', font=dict(size=10)),
                        width=1200,
                        height=900

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
        mlflow.set_tracking_uri("http://localhost:8000")
        mlflow.set_experiment(f"{mlflow_exp}")
        # ===============================================
        with mlflow.start_run(run_name=f"{mlflow_name}") as run:
            # Log the Plotly figure to MLflow
            output_path = f"umap_plot_{mlflow_name}.html"
            print("\nLogging interactive figure to MLflow...")
            fig.write_html(output_path, include_plotlyjs='cdn')
            mlflow.log_artifact(output_path, artifact_path="plots")
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
    # --- Load Configuration ---
    config = load_config(args)
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
    # ----------------- Get embedding and attention weights dimension from the model -----------------
    embedding_dim = encoder_model.output[0].shape[-1]
    #
    os.makedirs(config['path_to_save'], exist_ok=True)
    h5_path = os.path.join(config['path_to_save'], 'embeddings.h5')
    print(f"\nStreaming embeddings directly to HDF5 file: {h5_path} .")
    # -------------- Create the HDF5 file and resizable datasets ---------------
    try:
        with h5py.File(h5_path, 'w') as hf:
            # 
            string_dtype = h5py.string_dtype(encoding='utf-8')
            dset_ids = hf.create_dataset('ids', (0,), maxshape=(None,), dtype='int64')
            dset_labels = hf.create_dataset('labels', (0,), maxshape=(None,), dtype=string_dtype)
            dset_embeddings = hf.create_dataset('embeddings', (0, embedding_dim), maxshape=(None, embedding_dim), dtype='float32')
        
            num_rows_written = 0
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
                batch_embeddings, _ = encoder_model(model_inputs, training=False)
                batch_size = batch_embeddings.shape[0]
                # Resize the datasets on disk to make space for the new batch
                dset_embeddings.resize((num_rows_written + batch_size, embedding_dim))
                dset_labels.resize((num_rows_written + batch_size,))
                dset_ids.resize((num_rows_written + batch_size,))
                # Write the new data into the newly created space
                dset_embeddings[num_rows_written:] = batch_embeddings.numpy()
                # dset_attention[num_rows_written:] = batch_attention_weights.numpy()
                labels_as_bytes = batch['label'].numpy().astype(np.bytes_)
                dset_labels[num_rows_written:] = labels_as_bytes
                dset_ids[num_rows_written:] = batch['id'].numpy()
                # Update the row counter
                num_rows_written += batch_size

        print(f"\n-- Generation Complete !")
        print(f"\nSuccessfully wrote {num_rows_written} embeddings to {h5_path} .")
    except Exception as e:
        print(f"\nERROR: Could not save the files. Check: {e}\n")
    # -------------------------------------------------------------------------------------------------
    generate_plot(h5_path, config['path_to_class_count'], model_params, config['mlflow_upload'], config['mlflow_name'], config['mlflow_exp'])
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