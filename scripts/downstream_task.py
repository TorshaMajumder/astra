# =========================================================
# Import all dependencies
# =========================================================
import os
import h5py
import mlflow
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import defaultdict
from astra.utils.helper import load_config
from astra.src.classifier import mlp_classifier
# from coniferest.isoforest import IsolationForest
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# ==========================================================
os.system('clear')
# ===========================================================

def sample_embeddings(embeddings, labels, ids, sampling_config, random_state=None):
    """
    Performs custom over/under-sampling on the dataset based on a configuration dictionary.

    Parameters:
    ---------------------------------------------------------------------------------------
        embeddings (np.ndarray): The full embeddings array
        labels (np.ndarray): The full array of string labels
        ids (np.ndarray): The full array of IDs
        sampling_config (dict): A dictionary mapping class names to desired sample counts
        random_state (int, optional): Seed for the random number generator for reproducibility

    Returns:
    ---------------------------------------------------------------------------------------
        tuple: A tuple containing the new (sampled_embeddings, sampled_labels, sampled_ids).
    """
    if random_state is not None:
        np.random.seed(random_state)

    print("\n--- Starting custom sampling based on configuration...")
    final_indices = []
    # -------------------------------- Get class counts --------------------------------
    # Get unique classes and their counts from the original dataset
    #
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    original_counts_dict = dict(zip(unique_classes, class_counts))

    for class_name, desired_count in sampling_config.items():
        # Check if the class exists in the dataset
        if class_name not in original_counts_dict:
            print(f"\n---   Warning: Class '{class_name}' not found in the dataset. Skipping...")
            continue

        actual_count = original_counts_dict[class_name]
        # Get all indices for the current class
        class_indices = np.where(labels == class_name)[0]
        #
        if actual_count > desired_count:
            # --- Undersampling Case ---
            print(f"\n---   Class '{class_name}': Found {actual_count}, Target {desired_count}. Undersampling...")
            sampled_indices = np.random.choice(class_indices, size=desired_count, replace=False)
        
        elif actual_count < desired_count:
            # --- Oversampling Case ---
            print(f"\n---   Class '{class_name}': Found {actual_count}, Target {desired_count}. Oversampling...")
            # 
            all_existing_indices = class_indices
            num_to_oversample = desired_count - actual_count
            oversampled_indices = np.random.choice(class_indices, size=num_to_oversample, replace=True)
            sampled_indices = np.concatenate([all_existing_indices, oversampled_indices])
        
        else: 
            # --- Exact Match Case ---
            print(f"\n---   Class '{class_name}': Found {actual_count}, Target {desired_count}. Taking all samples...")
            sampled_indices = class_indices

        final_indices.extend(sampled_indices)
    # -------------------------------- Sampling summary --------------------------------------------------------
    print(f"\n---   Total samples after sampling: {len(final_indices)}")
    
    # Shuffle the final list of indices to mix the classes together
    np.random.shuffle(final_indices)
    # Use the final embeddings after sampling
    sampled_embeddings = embeddings[final_indices]
    sampled_labels = labels[final_indices]
    sampled_ids = ids[final_indices]

    return sampled_embeddings, sampled_labels, sampled_ids


def save_bootstrap_results(aggregated, output_filepath, model_key, n_iterations):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        f.write(f"--- Bootstrap Results: {model_key.upper()} ---\n")
        f.write(f"Iterations: {n_iterations}\n\n")
        
        for m in ['accuracy', 'micro_f1', 'macro_f1']:
            f.write(f"{m.replace('_', ' ').title()}:\n")
            f.write(f"  Mean: {aggregated[m]['mean']:.4f}\n")
            f.write(f"  Std:  {aggregated[m]['std']:.4f}\n\n")
        
        f.write("Per-Class F1-Scores:\n")
        for m, stats in aggregated.items():
            if m.startswith('f1_cls_'):
                cls_name = m.replace('f1_cls_', '')
                f.write(f"  Class {cls_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
    print(f"Report saved to {output_filepath}")


def run_bootstrap_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config):
    """
    Splits data, then trains and evaluates all classifiers specified in the config
    on ASTRA embeddings using BOOTSTRAPPING.

    Parameters:
    ---------------------------------------------------------------------
        embeddings (np.array): ASTRA embeddings
        labels (np.array): labels (ground truth)
        config (dict): loaded configuration dictionary

    Return:
    ---------------------------------------------------------------------
        Save the confusion matrix in "path_to_save" folder
    """
    #
    # load the Classification parameters 
    #
    class_config = config['classification_params']
    n_iter = class_config.get('n_iterations', 10)
    path_to_save = config.get('path_to_save', './results')
    unique_labels = sorted(np.unique(val_labels))
    # =============================================================================
    # ------------- Preprocessing the embeddings to extract the mean --------------
    train_embeddings = train_embeddings.reshape(-1, 3, 512).mean(axis=1)
    val_embeddings = val_embeddings.reshape(-1, 3, 512).mean(axis=1)
    # =============================================================================
    # 
    # Encode string labels to integers
    # 
    print("\n--- Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    train_label_encoded = label_encoder.fit_transform(train_labels)
    val_label_encoded = label_encoder.transform(val_labels)
    num_classes = len(label_encoder.classes_)
    # =============================================================================
    # Metrics storage
    all_model_results = {}
    # =============================================================================
    
    print(f"---   Found {len(np.unique(train_label_encoded))} classes in training set and {len(np.unique(val_label_encoded))} classes in validation set.")
    # --------------------------------------------------------------------------------------------
    #
    # ------------------- Loop through and evaluate each specified model ----------------------
    #
    for model_key in class_config['models']:
        #
        # Store metrics from each iteration
        #
        metrics = defaultdict(list)
        #
        model_key = model_key.lower()
        model_params = class_config.get(f'{model_key}_params', {})
        print(f"\n{'='*20} Starting Bootstrap Evaluation ({class_config['n_iterations']} iterations) for {model_key.upper()} {'='*20}")
        #
        for i in tqdm(range(n_iter), desc=f"Evaluating {model_key}"):
            # --------------------------------------------------------------------------------------------
            # RESAMPLE WITH REPLACEMENT of the same size as the original dataset
            boot_embeddings, boot_labels = resample(train_embeddings, train_label_encoded, random_state=i)
            # --------------------------------------------------------------------------------------------
            if model_key == 'knn':
                boot_embeddings = normalize(boot_embeddings, norm='l2', axis=1)
                val_embeddings = normalize(val_embeddings, norm='l2', axis=1)
                y_pred = weighted_knn(
                                        boot_embeddings, boot_labels, val_embeddings, 
                                        k=model_params.get('k', 20), 
                                        temperature=model_params.get('temperature', 0.07),
                                        num_classes=num_classes,
                                        batch_size=model_params.get('batches', 1000)
                                    )
            elif model_key == 'lr':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(boot_embeddings)
                X_test_scaled = scaler.transform(val_embeddings)
                clf = LogisticRegression(random_state=i, **model_params)
                clf.fit(X_train_scaled, boot_labels)
                y_pred = clf.predict(X_test_scaled)
            
            elif model_key == 'rf':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(boot_embeddings)
                X_test_scaled = scaler.transform(val_embeddings)
                clf = RandomForestClassifier(random_state=i, **model_params)
                clf.fit(X_train_scaled, boot_labels)
                y_pred = clf.predict(X_test_scaled)
                # --------------------------------------------------------------------------------------------
            # 
            metrics['accuracy'].append(accuracy_score(val_label_encoded, y_pred))
            metrics['macro_f1'].append(f1_score(val_label_encoded, y_pred, average='macro'))
            metrics['micro_f1'].append(f1_score(val_label_encoded, y_pred, average='micro'))
            # 
            report = classification_report(val_label_encoded, y_pred, output_dict=True, zero_division=0)
            for cls_idx in range(num_classes):
                label_str = str(cls_idx)
                if label_str in report:
                    metrics[f'f1_cls_{label_encoder.classes_[cls_idx]}'].append(report[label_str]['f1-score'])
            
            # ------------------------------------------------------------------------
            # Optional: Print progress
            if (i + 1) % 5 == 0:
                print(f"   : Completed iteration {i+1}/{class_config['n_iterations']}")
        # ----------------------------------------------------------------------------  
        # Aggregate Results
        #
        aggregated = {m: {'mean': np.mean(v), 'std': np.std(v)} for m, v in metrics.items()}
        all_model_results[model_key] = aggregated
        
        output_path = os.path.join(path_to_save, f"{model_key}_bootstrap_report.txt")
        save_bootstrap_results(aggregated, output_path, model_key, n_iter)
        
        

def weighted_knn(train_embeddings, train_labels, val_embeddings, k, temperature, num_classes, batch_size=1000):
    
    all_predictions = []
    
    for i in range(0, len(val_embeddings), batch_size):
        val_batch = val_embeddings[i : i + batch_size]
        # Find cosine similarity 
        sim_matrix = np.matmul(val_batch, train_embeddings.T)
        # Get Top K neighbors
        # Partition to get top K indices (unsorted)
        topk_idx_unsorted = np.argpartition(-sim_matrix, kth=k-1, axis=1)[:, :k]
        # Extract sims to sort them properly
        topk_sims_unsorted = np.take_along_axis(sim_matrix, topk_idx_unsorted, axis=1)
        # Sort the top K
        sort_order = np.argsort(-topk_sims_unsorted, axis=1)
        topk_idx = np.take_along_axis(topk_idx_unsorted, sort_order, axis=1)
        topk_sims = np.take_along_axis(topk_sims_unsorted, sort_order, axis=1)
        # Calculate Weights
        weights = np.exp(topk_sims / temperature)
        # Vectorized Weighted Voting
        # Get classes of neighbors: shape (batch_size, k)
        neighbor_classes = train_labels[topk_idx]
        # Create a score buffer: (batch_size, num_classes)
        class_scores = np.zeros((len(val_batch), num_classes))
        rows = np.arange(len(val_batch))[:, np.newaxis]
        np.add.at(class_scores, (rows, neighbor_classes), weights)
        all_predictions.append(np.argmax(class_scores, axis=1))
        
    return np.concatenate(all_predictions)




def run_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config):
    """
    Splits data, then trains and evaluates all classifiers specified in the config
    on ASTRA embeddings.

    Parameters:
    ---------------------------------------------------------------------
        embeddings (np.array): ASTRA embeddings
        labels (np.array): labels (ground truth)
        config (dict): loaded configuration dictionary

    Return:
    ---------------------------------------------------------------------
        Save the confusion matrix in "path_to_save" folder
    """
    #
    # load the Classification parameters and sampling config if exits
    #
    class_config = config.get('classification_params', {}) 
    # 
    # Encode string labels to integers
    # 
    print("\n--- Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    train_label_encoded = label_encoder.fit_transform(train_labels)
    val_label_encoded = label_encoder.transform(val_labels)
    num_classes = len(label_encoder.classes_)
    print(f"\n---   Found {len(np.unique(train_label_encoded))} classes in training set and {len(np.unique(val_label_encoded))} classes in validation set.")
    # --------------------------------------------------------------------------------------------
    print("\n-------------------------- Running Supervised Classification Task ---------------------------")
    #
    # ------------ Standardize the Embeddings ------------
    #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_embeddings)
    X_test_scaled = scaler.transform(val_embeddings)
    #
    # ----------------------- Loop through and evaluate each specified model ------------------------
    #
    for model_key in class_config['models']:
        #
        model_key = model_key.lower()
        model_params = class_config.get(f'{model_key}_params', {})
        print(f"\n{'='*20} Evaluating: {model_key.upper()} {'='*20}")
        #
        # ----------------------------- Instantiate Classifier --------------------------------
        #
        if model_key == 'rf':
            print("\n-- Instantiating Random Forest classifier with params:", model_params)
            classifier = RandomForestClassifier(random_state=class_config['random_state'], **model_params)
            #
            # ------------------ Train Classifier ------------------
            print("\nTraining classifier...")
            classifier.fit(train_embeddings, train_label_encoded)
            #
            print("\nEvaluating on the held-out validation data...")
            y_pred = classifier.predict(val_embeddings)
            # ------------------------------------------------------
        elif model_key == 'knn':
                train_embeddings_ = normalize(train_embeddings, norm='l2', axis=1)
                val_embeddings_ = normalize(val_embeddings, norm='l2', axis=1)
                y_pred = weighted_knn(
                                        train_embeddings_, 
                                        train_label_encoded, 
                                        val_embeddings_, 
                                        k=model_params.get('k', 20), 
                                        temperature=model_params.get('temperature', 0.07),
                                        num_classes=num_classes,
                                        batch_size=model_params.get('batches', 1000)
                                    )
        elif model_key == 'lr':
            print("\n-- Instantiating Logistic Regression classifier with params:", model_params)
            classifier = LogisticRegression(random_state=class_config['random_state'], **model_params)
            # classifier = LogisticRegressionCV(
            #                                     Cs=[0.01, 0.1, 1.0, 10.0, 100.0], 
            #                                     cv=3, 
            #                                     random_state=42, 
            #                                     max_iter=2000, 
            #                                     solver='lbfgs',
                                                
            #                                     n_jobs=20  # Uses all CPU cores to run the tests in parallel
            # )
            #
            # ------------------ Train Classifier ------------------
            print("\nTraining classifier...")
            classifier.fit(X_train_scaled , train_label_encoded)
            #
            print("\nEvaluating on the held-out validation data...")
            y_pred = classifier.predict(X_test_scaled)
            # print(f"Best C value found: {classifier.C_}")
            # return
            # ------------------------------------------------------
        elif model_key == 'mlp':
            print("\n-- Instantiating MLP classifier with params:", model_params)
            history, accuracy, y_pred = mlp_classifier(
                                        X_train_scaled, 
                                        train_label_encoded, 
                                        X_test_scaled, 
                                        val_label_encoded, 
                                        input_dim=X_train_scaled.shape[1], 
                                        num_classes=len(label_encoder.classes_), 
                                        mlp_params=model_params
                                    )
            
        else:
            print(f"\nWarning: Classifier type '{model_key}' not recognized. Skipping...")
            continue
        #
        # ------------------ Evaluate the classifier ------------------
        #
        accuracy = accuracy_score(val_label_encoded, y_pred)
        report = classification_report(val_label_encoded, y_pred, target_names=label_encoder.classes_)
        #
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)
        # ---------- Confusion Matrix ------------
        #
        # Define the order of classes for better visualization
        #
        class_order = ["AGN", "YSO", "SOLAR_LIKE", "S", "CV", "LPV", "DSCT|GDOR|SXPHE", "RR", "CEP", "ECL", "ELL", "RS"]
        #
        # Get the reordered indices based on the original label encoder classes
        #
        original_class_order = list(label_encoder.classes_)
        reorder_indices = [original_class_order.index(co) for co in class_order]
        #
        # Compute and reorder the confusion matrix
        #
        cm = confusion_matrix(val_label_encoded, y_pred)
        cm_perc = confusion_matrix(val_label_encoded, y_pred, normalize='true')
        cm_reordered = cm[np.ix_(reorder_indices, reorder_indices)]
        cm_perc_reordered = cm_perc[np.ix_(reorder_indices, reorder_indices)]
        # ---------------- Create custom labels (e.g., "50 \n (85.2%)") -------------------
        labels = [f"{count}\n({perc:.1%})" for count, perc in zip(cm_reordered.flatten(), cm_perc_reordered.flatten())]
        labels = np.asarray(labels).reshape(cm_reordered.shape)
        # -------------------------------- Plots --------------------------
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_perc_reordered, annot=labels, fmt="", cmap='YlGnBu', vmin=0.0, vmax=1.0,
                    annot_kws={"size": 8}, cbar_kws={'label': 'Purity Scale', 'ticks': [0.0, 0.25, 0.5, 0.75, 1.0],'format': '%.1f'},
                    xticklabels=class_order, yticklabels=class_order)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        ax.set_ylabel('True Labels', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_key.upper()} Classifier', fontsize=14, pad=20)
        # ------------------- Save the figure ----------------------
        output_filename = os.path.join(config["path_to_save"], f'confusion_matrix_{model_key}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {output_filename} .")
        plt.close() 
        #
        #
        if config["mlflow_upload"]:
            # ==========================================================================================
            # (IMPORTANT): Remove MLflow logging before packaging
            #
            # Initialize MLflow Tracking
            # Set an URI and Experiment name for MLflow
            #
            mlflow.set_tracking_uri("http://localhost:8000")
            mlflow.set_experiment(f'{config["mlflow_exp"]}')
            print(f"\n{'='*20} Logging to MLflow {'='*20}")
            # ===============================================
            with mlflow.start_run(run_name=f"{config['mlflow_name']}") as run:
                # Log the confusion matrix, classification report, and accuracy score to MLflow
                mlflow.log_metric(f"{model_key}.accuracy", accuracy)
                print("\nLogged accuracy score...")
                # Log CLASSIFICATION REPORT as a text artifact
                mlflow.log_text(report, f"reports/classification_report_{model_key}.txt")
                print(f"\nLogged classification_report...")
                # Log the confusion matrix plot as an ARTIFACT (figure)
                # mlflow.log_figure(fig, f"plots/confusion_matrix_{model_key}.png")
                print("\nLogged confusion matrix...")
                print("\nAll METRICS logged successfully!\n")  
            #
            #
            # ==================================== END OF LOGGING =======================================


def run_anomaly_detection_task(embeddings, labels, ids, config):
    """
    Evaluates an Isolation Forest model on the ASTRA embeddings.

    Parameters:
    ---------------------------------------------------------------------
        embeddings (np.array): ASTRA embeddings
        labels (np.array): labels (ground truth)
        ids (np.array): object ids of the emebeddings
        config (dict): loaded configuration dictionary

    Return:
    ---------------------------------------------------------------------
        Save the anomalous object ids and labels in "path_to_save" folder
    """
    #
    # load the AD parameters 
    #
    ad_config = config['anomaly_detection_params']
    print(f"\n{'-'*20} Running Anomaly Detection Task {'-'*20}")
    #
    # ---------------------- Loop through and evaluate each specified model ------------------------
    #
    for model_key in ad_config['models']:
        #
        model_key = model_key.lower()
        model_params = ad_config.get(f'{model_key}_params', {})
        #
        # --------------------------- Instantiate Classifier --------------------------
        #
        if model_key == 'if':
            print("\n-- Instantiating Isolation Forest with params:", model_params)
            model = IsolationForest(**model_params)
            #
            # Get anomaly score and save the top "inspect_samples" candidates
            #
            scores = model.fit(embeddings).score_samples(embeddings)
            index = scores.argsort()[:ad_config.get('inspect_samples')]
        else:
            print(f"\nWarning: Model type '{model_key}' not recognized. Skipping...")
            continue
        # -----------------------------------------------------------------------------------------------
        # Save the object ids and labels as a text file
        #
        object_ids, object_labels = ids[index], labels[index]
        np.savetxt(os.path.join(config["path_to_save"], f'discovered_anomalies_ids_{model_key}.txt'), 
                    object_ids, 
                    fmt='%d', 
                    )
        np.savetxt(os.path.join(config["path_to_save"], f'discovered_anomalies_labels_{model_key}.txt'), 
                    object_labels, 
                    fmt='%s', 
                    )
        print(f'\nSuccessfully saved {len(object_ids)} anomaly indices to: {config["path_to_save"]} .')
        #
        #------------------------------------------------------------------------------------------------
        if config["mlflow_upload"]:
            # ==========================================================================================
            # (IMPORTANT): Remove MLflow logging before packaging
            #
            # Initialize MLflow Tracking
            # Set an URI and Experiment name for MLflow
            #
            mlflow.set_tracking_uri("http://localhost:8000")
            mlflow.set_experiment(f'{config["mlflow_exp"]}')
            print(f"\n{'='*20} Logging to MLflow {'='*20}")
            # ===============================================
            with mlflow.start_run(run_name=f"{config['mlflow_name']}") as run:
                #
                # Save the object ids and labels as txt file in artifacts/
                #
                mlflow.log_artifact(local_path=os.path.join(config["path_to_save"], f'discovered_anomalies_ids_{model_key}.txt'), artifact_path="reports")
                mlflow.log_artifact(local_path=os.path.join(config["path_to_save"], f'discovered_anomalies_labels_{model_key}.txt'), artifact_path="reports")
                print("\nAll anomaly objects are logged successfully!\n")  
            #
            #
            # ==================================== END OF LOGGING =======================================


def main():
    # ==========================================================
    # Set up the Argument Parser
    # ==========================================================
    parser = argparse.ArgumentParser(prog='astra-downstream',
                                        description="Downstream tasks for Astra embeddings")
    # ==========================================================
    # Setup all required arguments
    # =========================================================
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    # ==========================================================
    # Optional arguments to override config file parameters
    # ==========================================================
    # parser.add_argument('--mlflow_upload', type=bool, default=True, help='Provide TRUE if you want to upload the METRICS of ' \
    #                                                                         'the ASTRA downstream task to MLflow else FALSE.')
    args = parser.parse_args()
    # ==========================================================
    #
    # ----------------------------------- Load Configuration ----------------------------------------
    #
    config = load_config(args)
    # Load paths
    path_to_data = config.get('path_to_data', {})
    # -------------------------------------- Load Training Data -------------------------------------
    #
    try:
        if path_to_data.get('train'):
            print(f"\nLoading data from: {path_to_data['train']}...")
            with h5py.File(path_to_data['train'], 'r') as hf:
                train_embeddings = hf['embeddings'][:]
                labels_raw = hf['labels'][:]
                train_ids = hf['ids'][:]
            labels_as_bytes = np.array(labels_raw, dtype=np.bytes_)
            train_labels = np.char.decode(labels_as_bytes, encoding='utf-8')
            print(f"\nSuccessfully loaded {len(train_embeddings)} embeddings and {len(train_ids)} ids...")
    
    except FileNotFoundError:
        print(f"\nError: HDF5 file not found at path - {path_to_data['train']}")
        return 
    except KeyError as e:
        print(f"\nError: Dataset '{e.args[0]}' not found in the HDF5 file.")
        print("Please ensure the file contains 'embeddings', 'labels', and 'ids' datasets.")
        return
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ----------------------------------- Load Validation Data ---------------------------------------
    try:
        if path_to_data.get('val'):
            print(f"\nLoading data from: {path_to_data['val']}...")
            with h5py.File(path_to_data['val'], 'r') as hf:
                val_embeddings = hf['embeddings'][:]
                labels_raw = hf['labels'][:]
                val_ids = hf['ids'][:]
            labels_as_bytes = np.array(labels_raw, dtype=np.bytes_)
            val_labels = np.char.decode(labels_as_bytes, encoding='utf-8')
            print(f"\nSuccessfully loaded {len(val_embeddings)} embeddings and {len(val_ids)} ids...")
    except FileNotFoundError:
        print(f"\nError: HDF5 file not found at path - {path_to_data['val']}")
        return 
    except KeyError as e:
        print(f"\nError: Dataset '{e.args[0]}' not found in the HDF5 file.")
        print("Please ensure the file contains 'embeddings', 'labels', and 'ids' datasets.")
        return
    # ------------------------------------------------------------------------------------------------
    # --- Execute the Correct Task Based on Config ---
    task_type = config.get('task', '').lower()
    
    if task_type == 'classification':
        
        if config["classification_params"].get('bootstrap'):
            run_bootstrap_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config)
        else:
            run_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config)
            # knn_evaluation(train_embeddings, train_labels, val_embeddings, val_labels, k=20, temperature=0.07)
    
    elif task_type == 'anomaly_detection':
        if path_to_data.get('test'):
            # ------------------------------------------------------------------------------------------------
            #
            # LOAD TEST DATA ONLY FOR ANOMALY DETECTION TASK
            #
            # ----------------------------------- Load Validation Data ---------------------------------------
            try:
                print(f"\nLoading data from: {path_to_data['test']}...")
                with h5py.File(path_to_data['test'], 'r') as hf:
                    test_embeddings = hf['embeddings'][:]
                    labels_raw = hf['labels'][:]
                    test_ids = hf['ids'][:]
                labels_as_bytes = np.array(labels_raw, dtype=np.bytes_)
                test_labels = np.char.decode(labels_as_bytes, encoding='utf-8')
                print(f"\nSuccessfully loaded {len(test_embeddings)} embeddings and {len(test_ids)} ids...")
            except FileNotFoundError:
                print(f"\nError: HDF5 file not found at path - {path_to_data['test']}")
                return 
            except KeyError as e:
                print(f"\nError: Dataset '{e.args[0]}' not found in the HDF5 file.")
                print("Please ensure the file contains 'embeddings', 'labels', and 'ids' datasets.")
                return
            # ------------------------------------------------------------------------------------------------


        else:
            print("\nWARNING: 'test' data path is missing for anomaly detection task. So, concatenating the 'train' and 'val' data for anomaly detection...")
            # Concatenate the 'train' and 'val' Embeddings
            test_embeddings = np.concatenate((train_embeddings, val_embeddings), axis=0)
            # Concatenate the Labels 
            test_labels = np.concatenate((train_labels, val_labels), axis=0)
            # Concatenate the IDs 
            test_ids = np.concatenate((train_ids, val_ids), axis=0)
        
        run_anomaly_detection_task(test_embeddings, test_labels, test_ids, config)
    
    else:
        raise ValueError(f"\nTask type '{config.get('task')}' in config file is not supported. "
                            "Choose 'classification' or 'anomaly_detection'.")
    
    print("\nDownstream tasks finished!\n")


if __name__ == '__main__':
    main()