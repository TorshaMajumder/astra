import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Path to the directory where you saved your .npy files for the COMPLETE TEST SET
PRETRAIN_RUN_DIR = "/media3/majumder/contrastive_loss_res/run_20250826_222245/"
FINETUNE_RUN_DIR = "/media3/majumder/contrastive_loss_res/run_20250826_222245/finetune_20250907_002342/"
path_to_save = "/media3/majumder/contrastive_loss_res/run_20250826_222245/"
# --- 2. Helper Function to Run and Evaluate the Classifier ---

def run_linear_probe_with_split(path_to_save, embeddings, labels, model_name, test_size=0.2):
    """
    Splits embeddings into train/test, trains a Logistic Regression
    classifier, and evaluates it.
    
    Args:
        embeddings (np.array): The full set of embeddings for the test data.
        labels (np.array): The corresponding full set of labels.
        model_name (str): Name of the model for printing results.
        test_size (float): The fraction of data to use for the test split.
    """
    print(f"\n--- Running Linear Probe for: {model_name} Encoder ---")
    
    # --- Split Data into Training and Testing Sets ---
    # stratify=labels ensures that the train and test sets have the same
    # proportion of class labels as the input dataset. This is crucial for
    # imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples.")
    
    # --- Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Train the Selected Classifier ---
    if classifier_type.lower() == 'rf':
        print("Training Random Forest classifier...")
        # n_estimators=100 is a good default. n_jobs=-1 uses all available CPU cores.
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif classifier_type.lower() == 'lr':
        print("Training Logistic Regression classifier...")
        classifier = LogisticRegression(random_state=42, C=0.1, max_iter=1000)
    else:
        raise ValueError("classifier_type must be 'rf' or 'lr'")

    classifier.fit(X_train_scaled, y_train)
    
    # --- Evaluate on the Test Split ---
    print("Evaluating on the held-out test split...")
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # --- Detailed Report ---
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # --- Confusion Matrix ---
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
    plt.title(f'Confusion Matrix - {model_name} Encoder')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    # plt.show()
    # Save the figure to the same run directory
    output_filename = os.path.join(path_to_save, f'{model_name}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')


# --- 3. Main Script Logic ---

# --- Load Embeddings and Labels for the ENTIRE TEST SET ---
print("Loading data for PRE-TRAINED model (entire test set)...")
# Make sure these files correspond to the full test set
pt_test_embeddings = np.load(os.path.join(PRETRAIN_RUN_DIR, 'embeddings.npy'))
pt_test_labels = np.load(os.path.join(PRETRAIN_RUN_DIR, 'labels.npy'))

print("\nLoading data for FINE-TUNED model (entire test set)...")
ft_test_embeddings = np.load(os.path.join(FINETUNE_RUN_DIR, 'embeddings.npy'))
ft_test_labels = np.load(os.path.join(PRETRAIN_RUN_DIR, 'labels.npy'))
# The labels are the same, so we can use pt_test_labels


# --- 1. Define your Label Mapping and Create the Reverse Map ---
ztf_labels = {
    'CEP': ['ACEP', 'DCEP', 'T2CEP'],
    'RRLY': ['RRab', 'RRc', 'RRd']
}

# This is the crucial lookup dictionary
reverse_map = {value: key for key, value_list in ztf_labels.items() for value in value_list}
# reverse_map will be: {'ACEP': 'CEP', 'DCEP': 'CEP', ..., 'RRab': 'RRLY', ...}


# --- 2. Load your Data (Simulated for this example) ---
# This line simulates what you are doing with np.load()
# It includes labels from both categories and some that might not be in the map.
# pt_test_labels = np.array(['DCEP', 'RRc', 'T2CEP', 'RRab', 'ACEP', 'OtherLabel'])

print("--- Original Labels ---")
print(pt_test_labels)
print("\n")


# --- 3. Perform the Mapping using Pandas ---

# Convert the NumPy array to a Pandas Series
labels_series = pd.Series(pt_test_labels)

# Use the .map() method. It will look up each value from the series
# in the keys of your reverse_map and replace it with the corresponding value.
mapped_series = labels_series.map(reverse_map)

# If a label from your array was NOT in the reverse_map (e.g., 'OtherLabel'),
# .map() will put a NaN (Not a Number) in its place. We should fill these
# back in with their original values.
mapped_series = mapped_series.fillna(labels_series)

# Convert the final mapped series back to a NumPy array
final_mapped_labels = mapped_series.to_numpy()


# --- 4. Show the Result ---
print("--- Mapped Labels (using Pandas) ---")
print(final_mapped_labels)

# --- Run the Linear Probing ---
# Run for the Pre-trained model
run_linear_probe_with_split(
    path_to_save = path_to_save,
    embeddings=pt_test_embeddings,
    labels=final_mapped_labels,
    model_name="Pre-trained"
)

# Run for the Fine-tuned model
run_linear_probe_with_split(
    path_to_save = path_to_save,
    embeddings=ft_test_embeddings,
    labels=final_mapped_labels, # Use the same labels
    model_name="Fine-tuned"
)