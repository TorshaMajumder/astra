# =========================================================
# Import all dependencies
# =========================================================
import os
import h5py
import mlflow
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import defaultdict
from astra.utils.helper import load_config
from astra.src.classifier import mlp_classifier
from coniferest.isoforest import IsolationForest
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def save_bootstrap_results(results, output_filepath, classifier_type, n_iterations):
    """
    Formats and saves the aggregated bootstrap results to a text file.

    Parameters:
    -------------------------------------------------------------------------------------
        results (dict): The dictionary of aggregated results from the bootstrap function
        output_filepath (str): The full path where the text file will be saved
        classifier_type (str): The type of classifier used (e.g., 'lr', 'rf')
        n_iterations (int): The number of bootstrap iterations performed
    """
    #
    #  Ensure the directory exists before trying to write the file
    #
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    #
    with open(output_filepath, 'w') as f:
        f.write("--- Bootstrap Evaluation Results ---\n\n")
        f.write(f"Classifier Type: {classifier_type.upper()}\n")
        f.write(f"Number of Iterations: {n_iterations}\n")
        f.write("="*40 + "\n\n")
        # Write Overall Accuracy
        f.write("Overall Accuracy:\n")
        f.write(f"  - Mean: {results['accuracy_mean']:.4f}\n")
        f.write(f"  - Standard Deviation: {results['accuracy_std']:.4f}\n\n")
        # Write Per-Class F1-Scores
        f.write("Per-Class F1-Scores:\n")
        # Sort the labels for consistent output order
        sorted_labels = sorted(results['f1_scores'].keys())
        for label in sorted_labels:
            metrics = results['f1_scores'][label]
            f.write(f"  - Class '{label}':\n")
            f.write(f"    - Mean: {metrics['mean']:.4f}\n")
            f.write(f"    - Standard Deviation: {metrics['std']:.4f}\n")
        
        f.write("\n--- End of Report ---\n")
    
    print(f"\nBootstrap results successfully saved to: {output_filepath}\n")


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
    unique_labels = sorted(np.unique(val_labels))
    # =============================================================================
    # 
    # Encode string labels to integers
    # 
    print("\n--- Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    train_label_encoded = label_encoder.fit_transform(train_labels)
    val_label_encoded = label_encoder.transform(val_labels)
    
    print(f"---   Found {len(np.unique(train_label_encoded))} classes in training set and {len(np.unique(val_label_encoded))} \
          classes in validation set.")
    # --------------------------------------------------------------------------------------------
    #
    # ------------------- Loop through and evaluate each specified model ----------------------
    #
    for model_key in class_config['models']:
        #
        # Store metrics from each iteration
        #
        accuracies = []
        f1_scores_per_class = defaultdict(list)
        #
        model_key = model_key.lower()
        model_params = class_config.get(f'{model_key}_params', {})
        print(f"\n{'='*20} Starting Bootstrap Evaluation ({class_config['n_iterations']} iterations) for {model_key.upper()} {'='*20}")
        #
        # ----------------------------- Instantiate Classifier --------------------------------
        #
        if model_key == 'rf':
            print("\n-- Instantiating Random Forest classifier with params:", model_params)
            classifier = RandomForestClassifier(random_state=class_config['random_state'], **model_params)
        elif model_key == 'lr':
            print("\n-- Instantiating Logistic Regression classifier with params:", model_params)
            classifier = LogisticRegression(random_state=class_config['random_state'], **model_params)
        else:
            print(f"\nWarning: Classifier type '{model_key}' not recognized. Skipping...")
            continue
        #
        for i in range(class_config['n_iterations']):
            #
            # RESAMPLE WITH REPLACEMENT of the same size as the original dataset
            boot_embeddings, boot_labels = resample(train_embeddings, train_label_encoded, random_state=i)
            if model_key == 'rf':
                #
                # ------------------ Train and Evaluate ------------------
                classifier.fit(boot_embeddings, boot_labels)
                y_pred = classifier.predict(val_embeddings)
                #
            elif model_key == 'lr':
                # -------------- Standardize the Embeddings --------------
                #
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(boot_embeddings)
                X_test_scaled = scaler.transform(val_embeddings)
            #
                # ------------------ Train and Evaluate ------------------
                classifier.fit(X_train_scaled, boot_labels)
                #
                y_pred = classifier.predict(X_test_scaled)
                # ------------------------------------------------------
            # Save metrices
            accuracies.append(accuracy_score(val_label_encoded, y_pred))
            # Use output_dict=True to get a structured report
            report = classification_report(val_label_encoded, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
            for label in unique_labels:
                # Safely access the f1-score for each class
                f1_scores_per_class[str(label)].append(report[str(label)]['f1-score'])
            # ------------------------------------------------------------------------
            # Optional: Print progress
            if (i + 1) % 10 == 0:
                print(f"   : Completed iteration {i+1}/{class_config['n_iterations']}")
        # ----------------------------------------------------------------------------  
        # Aggregate Results
        #
        results = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'f1_scores': {}
                }
        for label in unique_labels:
            scores = f1_scores_per_class[str(label)]
            results['f1_scores'][str(label)] = {
                                                'mean': np.mean(scores),
                                                'std': np.std(scores)
                                            }
        output_path = os.path.join(config['path_to_save'], f"{model_key}_bootstrap_report.txt")                          
        save_bootstrap_results(results, output_path, model_key, class_config['n_iterations'])  
        print(f"\n-- Results:\n\n",results)
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
                mlflow.log_artifact(local_path=output_path, artifact_path="reports")
                print("\nBootstrapped metrices are logged successfully!\n")  
            #
            #
            # ==================================== END OF LOGGING =======================================



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
        elif model_key == 'lr':
            print("\n-- Instantiating Logistic Regression classifier with params:", model_params)
            classifier = LogisticRegression(random_state=class_config['random_state'], **model_params)
            # classifier = LogisticRegressionCV(
            #                                     Cs=[0.01, 0.1, 1.0, 10.0, 100.0], 
            #                                     cv=3, 
            #                                     random_state=42, 
            #                                     max_iter=2000, 
            #                                     solver='lbfgs',
                                                
            #                                     n_jobs=-1  # Uses all CPU cores to run the tests in parallel
            # )
            #
            # ------------------ Train Classifier ------------------
            print("\nTraining classifier...")
            classifier.fit(X_train_scaled , train_label_encoded)
            #
            print("\nEvaluating on the held-out validation data...")
            y_pred = classifier.predict(X_test_scaled)
            # print(f"Best C value found: {classifier.C_}")
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
    # # Calculate the variance of each feature column
    # variances = np.var(train_embeddings, axis=0)
    # feature_std = np.std(train_embeddings, axis=0)
    # print(f"Mean Variance across features: {np.mean(variances):.6f}")
    # print(f"Max Variance: {np.max(variances):.6f}")
    # print(f"Min Variance: {np.min(variances):.6f}")
    # print(f"Mean Std Dev across dimensions: {np.mean(feature_std):.6f}")
    # print(f"Number of dead dimensions (std < 1e-5): {np.sum(feature_std < 1e-5)}")

    # ------------------------------------------------------------------------------------------------
    # --- Execute the Correct Task Based on Config ---
    task_type = config.get('task', '').lower()
    
    if task_type == 'classification':
        
        if config["classification_params"].get('bootstrap'):
            run_bootstrap_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config)
        else:
            run_classification_task(train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids, config)
    
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