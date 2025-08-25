import numpy as np
import os
import pandas as pd
import umap # From the umap-learn library
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration: Set Your Paths and Parameters ---

# Path to the specific run directory containing the saved .npy files
run_directory = "/media3/majumder/contrastive_loss_res/run_20250824_064043/" # <--- SET THIS PATH

# UMAP parameters (you can tune these to change the visualization)
N_NEIGHBORS = 15      # Controls how UMAP balances local vs. global structure.
                      # Lower values focus more on local structure.
MIN_DIST = 0.5      # Controls how tightly UMAP is allowed to pack points together.
                      # Lower values create more compact clusters.
METRIC = 'cosine'     # Distance metric. 'cosine' is often excellent for high-dimensional
                      # embeddings from deep learning models. 'euclidean' is another option.
RANDOM_STATE = 42     # Set for reproducible UMAP results.

# Plotting parameters
POINT_SIZE = 50
ALPHA = 0.4       # Point transparency, useful for dense plots.

# --- 2. Load the Saved Embeddings and Metadata ---

print(f"Loading data from: {run_directory}")

embeddings_file = os.path.join(run_directory, 'embeddings.npy')
labels_file = os.path.join(run_directory, 'labels.npy')
ids_file = os.path.join(run_directory, 'ids.npy') # Optional, not used in plot but good to load

# Check if files exist
if not all(os.path.exists(f) for f in [embeddings_file, labels_file, ids_file]):
    print("ERROR: One or more .npy files (embeddings.npy, labels.npy, ids.npy) not found.")
    print("Please check the 'run_directory' path.")
    exit()

all_embeddings = np.load(embeddings_file)
all_labels = np.load(labels_file)
all_ids = np.load(ids_file)

# Decode labels from byte strings (e.g., b'Transient') to regular strings
try:
    labels_decoded = [label.decode('utf-8') for label in all_labels]
except (UnicodeDecodeError, AttributeError):
    print("Labels are not byte strings, using them as is.")
    labels_decoded = all_labels

def map_label(label_str):
    """Maps a specific label string to a broader category."""
    if label_str in ['ACEP', 'DCEP', 'T2CEP']:
        return 'CEP' # Cepheid Variables
    elif label_str in ['RRab', 'RRc', 'RRd']:
        return 'RRLY' # RR Lyrae Variables
    # Add other mappings here if you have more classes
    # elif label_str in ['ANOTHER_CLASS_1', 'ANOTHER_CLASS_2']:
    #     return 'BROADER_CATEGORY_2'
    else:
        # If the label doesn't match any mapping, return it as is
        return label_str

# Apply the mapping to all your decoded labels
labels_mapped = [map_label(lbl) for lbl in labels_decoded]

print(f"Loaded {len(all_embeddings)} embeddings with shape {all_embeddings.shape}")
print(f"Found {len(np.unique(labels_mapped))} unique labels from {len(np.unique(labels_decoded))}.")


# --- 3. Perform UMAP Dimensionality Reduction ---

print(f"\nPerforming UMAP reduction (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}, metric='{METRIC}')...")
print("This may take a few moments for large datasets...")

# Initialize UMAP. n_components=2 means we want a 2D plot.
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    n_components=2,
    metric=METRIC,
    random_state=RANDOM_STATE
)

# Fit the model and transform the data
embedding_2d = reducer.fit_transform(all_embeddings)

print("UMAP reduction complete.")


# --- 4. Prepare Data for Plotting with Pandas and Seaborn ---
# Using a DataFrame makes plotting with colored labels very easy.

df = pd.DataFrame()
df['label'] = labels_mapped
df['umap-one'] = embedding_2d[:, 0]
df['umap-two'] = embedding_2d[:, 1]


# --- 5. Create and Save the Plot ---

print("\nGenerating plot...")

# Set plot style
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# Create the scatter plot
plt.figure(figsize=(16, 12))
scatter_plot = sns.scatterplot(
    x="umap-one", y="umap-two",
    hue="label",         # This tells Seaborn to color points by their label
    palette=sns.color_palette("hsv", len(np.unique(labels_mapped))), # Use a nice color palette
    data=df,
    legend="full",
    alpha=ALPHA,
    s=POINT_SIZE
)

# Customize the plot
plt.title(f'UMAP Projection of ASTRA Embeddings (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)

# Move legend to the side if it's too big
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Remove plot ticks for a cleaner look
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Save the figure to the same run directory
output_filename = os.path.join(run_directory, f'umap_plot_n{N_NEIGHBORS}_d{MIN_DIST}.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"\nPlot saved successfully to: {output_filename}")

# Display the plot if running in a notebook environment
plt.show()