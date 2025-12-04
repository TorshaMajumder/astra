# =========================================================
# Import all dependencies
# =========================================================
import os
import mlflow
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from astra.utils.helper import load_config
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# ==========================================================
os.system('clear')
# ===========================================================

def run_classification_task(embeddings, labels, config):
    """
    Splits data, then trains and evaluates all classifiers specified in the config.
    
    Args:
        embeddings (np.array): The full set of embeddings.
        labels (np.array): The corresponding full set of labels.
        config (dict): The loaded configuration dictionary.
        output_dir (str): Directory to save plots and results.
    """
    class_config = config['classification_params']
    print("\n--- Running Supervised Classification Task ---")
    
    # --- Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
                                                        embeddings, 
                                                        labels, 
                                                        test_size=class_config['test_size'], 
                                                        random_state=class_config['random_state'], 
                                                        stratify=labels  
                                                    )
    print(f"\nSplitting data into {len(X_train)} training and {len(X_test)} testing samples.")
    
    # --- Standardize the Embeddings ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Loop Through and Evaluate Each Specified Model ---
    for model_key in class_config['models']:
        model_key = model_key.lower()
        model_params = class_config.get(f'{model_key}_params', {})
        
        print(f"\n{'='*20} Evaluating: {model_key.upper()} {'='*20}")

        # --- Instantiate Classifier ---
        if model_key == 'rf':
            print("\n --Instantiating Random Forest classifier with params:", model_params)
            classifier = RandomForestClassifier(random_state=class_config['random_state'], **model_params)
        elif model_key == 'lr':
            print("\n --Instantiating Logistic Regression classifier with params:", model_params)
            classifier = LogisticRegression(random_state=class_config['random_state'], **model_params)
        else:
            print(f"\nWarning: Classifier type '{model_key}' not recognized. Skipping...")
            continue

        # --- Train and Evaluate ---
        print("\nTraining classifier...")
        classifier.fit(X_train_scaled, y_train)
        
        print("\nEvaluating on the held-out test split...")
        y_pred = classifier.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n")
        print(report)
        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classifier.classes_, yticklabels=classifier.classes_)
        
        ax.set_ylabel('True Labels', fontsize=12)
        ax.set_xlabel('Predicted Labels', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_key.upper()} Classifier', fontsize=14)
        
        # ----- Save the figure -----
        output_filename = os.path.join(config["path_to_save"], f'confusion_matrix_{model_key}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {output_filename}")
        plt.close() # Close the plot to free up memory
        #
        #
        if config["mlflow_upload"]:
            # ==========================================================================================
            # (IMPORTANT): Remove MLflow logging before packaging
            #
            # Initialize MLflow Tracking
            # Set an URI and Experiment name for MLflow
            #
            mlflow.set_tracking_uri("http://127.0.0.1:37533")
            mlflow.set_experiment("Set3")
            print(f"\n{'='*20} Logging to MLflow {'='*20}")
            # ===============================================
            with mlflow.start_run(run_id=f"{config["mlflow_name"]}") as run:
                # Log the confusion matrix, classification report, and accuracy score to MLflow
                mlflow.log_metric(f"{model_key}.accuracy", accuracy)
                print("\nLogged accuracy score...")
                # Log CLASSIFICATION REPORT as a text artifact
                mlflow.log_text(report, f"reports/classification_report_{model_key}.txt")
                print(f"\nLogged classification_report...")
                # Log the confusion matrix plot as an ARTIFACT (figure)
                mlflow.log_figure(fig, f"plots/confusion_matrix_{model_key}.png")
                print("\nLogged confusion matrix...")
                print("\nAll METRICS logged successfully!")  
            #
            #
            # ==================================== END OF LOGGING =======================================


def run_anomaly_detection_task(embeddings, labels, config):
    """
    TODO: Placeholder for the anomaly detection task.
    """
    pass


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
    parser.add_argument('--mlflow_upload', type=bool, default=True, help='Provide TRUE if you want to upload the METRICS of ' \
                                                                            'the ASTRA downstream task to MLflow else FALSE.')
    args = parser.parse_args()
    # ==========================================================
    # --- Load Configuration ---
    config = load_config(args)
    # --- Load Data ---
    print("\nLoading data...")
    try:
        embeddings = np.load(config['path_to_data']['embeddings'])
        labels = np.load(config['path_to_data']['labels'])
        print(f"\nSuccessfully loaded {len(embeddings)} embeddings and {len(labels)} labels...")
    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
        return
    # --- Execute the Correct Task Based on Config ---
    task_type = config.get('task', '').lower()
    
    if task_type == 'classification':
        run_classification_task(embeddings, labels, config)
    elif task_type == 'anomaly_detection':
        run_anomaly_detection_task(embeddings, labels, config)
    else:
        raise ValueError(f"\nTask type '{config.get('task')}' in config file is not supported. "
                            "Choose 'classification' or 'anomaly_detection'.")
    
    print("\nDownstream tasks finished.\n")


if __name__ == '__main__':
    main()