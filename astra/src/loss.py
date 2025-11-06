
import numpy as np
import tensorflow as tf



LARGE_NUM = 1e9 # Constant for masking out similarities


def nt_xent_loss(*views, temperature):
    """
    Calculates the generalized NT-XENT loss for a variable number of views.
    
    This function works for both Siamese (2 views) and Triplet (3+ views) models.
    It assumes that for any given sample, its augmentations at the same index
    across all different views form a positive pair.

    Args:
        *views: A variable number of 2D tensors (z_view_1, z_view_2, ...), 
                each of shape (batch_size, projection_dim).
        temperature: A scalar float for the temperature scaling.

    Returns:
        A scalar tensor representing the mean NT-XENT loss.
    """
    # --- 1. Dynamic Input Handling and Normalization ---
    
    # Check if at least two views have been provided.
    if len(views) < 2:
        raise ValueError("nt_xent_loss requires at least two views (e.g., anchor and positive).")

    # Normalize all provided view embeddings in a loop.
    normalized_views = [tf.math.l2_normalize(view, axis=1) for view in views]
    
    # Get batch size from the first view and dynamically determine the number of views.
    batch_size = tf.shape(normalized_views[0])[0]
    n_views = len(normalized_views)
    total_size = n_views * batch_size

    # Concatenate all views into a single large tensor.
    z = tf.concat(normalized_views, axis=0) # Shape: (N*B, D)

    # --- 2. Similarity and Mask Calculation (No change in logic) ---
    
    # Calculate cosine similarity matrix.
    sim_matrix = tf.matmul(z, z, transpose_b=True) # Shape: (N*B, N*B)

    # Mask out diagonal (self-similarity).
    diag_mask = tf.logical_not(tf.eye(total_size, dtype=tf.bool))

    # --- 3. Dynamic Positive Pair Identification (The Core Generalization) ---

    # This section replaces the hard-coded pairs (pairs12, pairs13, etc.).
    # It programmatically generates all cross-view positive pairs.
    indices = tf.range(batch_size)
    all_positive_pairs = []

    # Loop through all unique combinations of views (i, j) where i < j.
    for i in range(n_views):
        for j in range(i + 1, n_views):
            # For each original sample, its augmentations in view i and view j are a positive pair.
            # Example for n=3, i=0, j=1: (anc_k, pos_k)
            # The indices are offset by `i * batch_size` and `j * batch_size`.
            pairs_ij = tf.stack([indices + i * batch_size, indices + j * batch_size], axis=1)
            
            # We must also include the symmetric pair (j, i)
            # Example: (pos_k, anc_k)
            pairs_ji = tf.stack([indices + j * batch_size, indices + i * batch_size], axis=1)
            
            all_positive_pairs.extend([pairs_ij, pairs_ji])

    # Concatenate all found positive pairs into a single tensor of indices.
    all_positive_pairs_indices = tf.concat(all_positive_pairs, axis=0)

    # --- 4. Loss Calculation (No change in logic) ---

    # The rest of the function remains the same, as it operates on the dynamically
    # created `sim_matrix` and `all_positive_pairs_indices`.

    # Scaled similarities.
    logits = sim_matrix / temperature

    # Mask diagonal for denominator calculation.
    logits_masked_diag = tf.where(diag_mask, logits, -LARGE_NUM)

    # Calculate log denominator (logsumexp over non-diagonal elements row-wise).
    log_den = tf.reduce_logsumexp(logits_masked_diag, axis=1) # Shape: (N*B,)

    # Get the numerator terms (similarities of the positive pairs).
    l_pos = tf.gather_nd(logits, all_positive_pairs_indices) # Shape: (num_pairs,)

    # Get the corresponding log denominator for each positive pair's row.
    log_den_for_pairs = tf.gather(log_den, all_positive_pairs_indices[:, 0])

    # Calculate the loss for each positive pair direction.
    pair_losses = -(l_pos - log_den_for_pairs)

    # Total loss is the average over all positive pairs.
    # total_loss = tf.reduce_mean(pair_losses)

    #return total_loss
    return pair_losses



