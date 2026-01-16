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
# from coniferest.isoforest import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# ==========================================================
os.system('clear')
# ===========================================================

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


def run_bootstrap_classification_task(embeddings, labels, config):
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
    unique_labels = sorted(np.unique(labels))
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
            boot_embeddings, boot_labels = resample(embeddings, labels, random_state=i)
            # ------------------------ Split Data into Training and Testing Sets -------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                                                                boot_embeddings, boot_labels, test_size=class_config['test_size'], 
                                                                random_state=class_config['random_state'], stratify=boot_labels
                                                            )
            # -------------- Standardize the Embeddings --------------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            #
            # ------------------ Train and Evaluate ------------------
            classifier.fit(X_train_scaled, y_train)
            #
            y_pred = classifier.predict(X_test_scaled)
            # Save metrices
            accuracies.append(accuracy_score(y_test, y_pred))
            # Use output_dict=True to get a structured report
            report = classification_report(y_test, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
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



def run_classification_task(embeddings, labels, config):
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
    # load the Classification parameters 
    #
    class_config = config['classification_params']
    # 
    # Encode string labels to integers
    # 
    print("\n--- Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    print(f"---   Found {len(np.unique(label_encoded))} classes!")
    # --------------------------------------------------------------------------------------------
    print("\n-------------------------- Running Supervised Classification Task ---------------------------")
    #
    # ------------------------ Split Data into Training and Testing Sets -------------------------
    #
    X_train, X_test, y_train, y_test = train_test_split(
                                                        embeddings, 
                                                        label_encoded, 
                                                        test_size=class_config['test_size'], 
                                                        random_state=class_config['random_state'], 
                                                        stratify=labels  
                                                    )
    print(f"\nSplitting data into {len(X_train)} training and {len(X_test)} testing samples.")
    #
    # ------------ Standardize the Embeddings ------------
    #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
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
            classifier.fit(X_train_scaled, y_train)
            #
            print("\nEvaluating on the held-out test split...")
            y_pred = classifier.predict(X_test_scaled)
            # ------------------------------------------------------
        elif model_key == 'lr':
            print("\n-- Instantiating Logistic Regression classifier with params:", model_params)
            classifier = LogisticRegression(random_state=class_config['random_state'], **model_params)
            #
            # ------------------ Train Classifier ------------------
            print("\nTraining classifier...")
            classifier.fit(X_train_scaled, y_train)
            #
            print("\nEvaluating on the held-out test split...")
            y_pred = classifier.predict(X_test_scaled)
            # ------------------------------------------------------
        elif model_key == 'mlp':
            print("\n-- Instantiating MLP classifier with params:", model_params)
            history, accuracy, y_pred = mlp_classifier(
                                        X_train_scaled, 
                                        y_train, 
                                        X_test_scaled, 
                                        y_test, 
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
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
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
        cm = confusion_matrix(y_test, y_pred)
        cm_perc = confusion_matrix(y_test, y_pred, normalize='true')
        cm_reordered = cm[np.ix_(reorder_indices, reorder_indices)]
        cm_perc_reordered = cm_perc[np.ix_(reorder_indices, reorder_indices)]
        # ---------------- Create custom labels (e.g., "50 \n (85.2%)") -------------------
        labels = [f"{count}\n({perc:.1%})" for count, perc in zip(cm_reordered.flatten(), cm_perc_reordered.flatten())]
        labels = np.asarray(labels).reshape(cm_reordered.shape)
        # ---------------- Plots -----------------
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_perc_reordered, annot=labels, fmt="", cmap='YlGnBu', 
                    annot_kws={"size": 8}, cbar_kws={'label': 'Purity Scale'},
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
            mlflow.set_tracking_uri("http://localhost:5000")
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
                mlflow.log_figure(fig, f"plots/confusion_matrix_{model_key}.png")
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
    # --- Load Configuration ---
    config = load_config(args)
    # --- Load Data ---
    print(f"\nLoading data from: {config['path_to_data']}...")
    try:
        with h5py.File(config['path_to_data'], 'r') as hf:
            embeddings = hf['embeddings'][:]
            labels_raw = hf['labels'][:]
            ids = hf['ids'][:]
        labels_as_bytes = np.array(labels_raw, dtype=np.bytes_)
        labels = np.char.decode(labels_as_bytes, encoding='utf-8')
        print(f"\nSuccessfully loaded {len(embeddings)} embeddings and {len(ids)} ids...")
    except FileNotFoundError:
        print(f"\nError: HDF5 file not found at path - {config['path_to_data']}")
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
            run_bootstrap_classification_task(embeddings, labels, config)
        else:
            run_classification_task(embeddings, labels, config)
    elif task_type == 'anomaly_detection':
        run_anomaly_detection_task(embeddings, labels, ids, config)
    else:
        raise ValueError(f"\nTask type '{config.get('task')}' in config file is not supported. "
                            "Choose 'classification' or 'anomaly_detection'.")
    
    print("\nDownstream tasks finished!\n")


if __name__ == '__main__':
    main()