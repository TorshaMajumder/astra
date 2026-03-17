# =========================================================
# Import all dependencies
# =========================================================
import numpy as np
import tensorflow as tf

# ============ Global Variables ===========================
# Constant for masking out similarities
LARGE_NUM = 1e9 
# =========================================================

def nt_xent_loss(*views, temperature):
    """
    Calculates the generalized NT-XENT loss for a variable number of views.
    
    This function works for both Siamese (2 views) and Triplet (3+ views) models.
    It assumes that for any given sample, its augmentations at the same index
    across all different views form a positive pair/group.

    Parameters:
    --------------------------------------------------------------------------------
        *views: A variable number of 2D tensors (z_view_1, z_view_2, ...), 
                each of shape (batch_size, projection_dim)
        temperature: A scalar float for the temperature scaling

    Returns:
    --------------------------------------------------------------------------------
        A scalar tensor representing the mean NT-XENT loss
    """
    #
    # Check if at least two views have been provided
    #
    if len(views) < 2:
        raise ValueError("ValueError: nt_xent_loss requires at least two views (e.g., anchor and positive).")
    #
    # Normalize all provided view embeddings in a loop
    #
    normalized_views = [tf.math.l2_normalize(view, axis=1) for view in views]
    #
    # Get batch size from the first view and dynamically determine the number of views
    #
    batch_size = tf.shape(normalized_views[0])[0]
    n_views = len(normalized_views)
    total_size = n_views * batch_size
    #
    # Concatenate all views into a single large tensor | Shape: (N*B, D)
    #
    z = tf.concat(normalized_views, axis=0) 
    #
    # Calculate cosine similarity matrix | Shape: (N*B, N*B)
    #
    sim_matrix = tf.matmul(z, z, transpose_b=True) 
    #
    # Mask out diagonal (self-similarity)
    #
    diag_mask = tf.logical_not(tf.eye(total_size, dtype=tf.bool))
    #
    # -------------- Determine the Positive Pairs ----------------
    # generate all cross-view positive pairs
    #
    indices = tf.range(batch_size)
    all_positive_pairs = []
    #
    # Loop through all unique combinations of views (i, j) where i < j
    #
    for i in range(n_views):
        for j in range(i + 1, n_views):
            # For each original sample, its augmentations in view i and view j are a positive pair
            # Example for n=3, i=0, j=1: (anc_k, pos_k)
            # The indices are offset by `i * batch_size` and `j * batch_size`.
            pairs_ij = tf.stack([indices + i * batch_size, indices + j * batch_size], axis=1)
            # include the symmetric pair (j, i)
            # Example: (pos_k, anc_k)
            pairs_ji = tf.stack([indices + j * batch_size, indices + i * batch_size], axis=1)
            # Get all the positive pairs
            all_positive_pairs.extend([pairs_ij, pairs_ji])
    # ---------------------------------------------------------------------------------------------
    # Concatenate all found positive pairs into a single tensor of indices.
    all_positive_pairs_indices = tf.concat(all_positive_pairs, axis=0)
    #
    # Apply teperature scaling to the similarity matrix
    #
    logits = sim_matrix / temperature
    #
    # Mask diagonal for denominator calculation
    logits_masked_diag = tf.where(diag_mask, logits, -LARGE_NUM)
    #
    # Calculate log denominator (logsumexp over non-diagonal elements row-wise) | Shape: (N*B,)
    log_den = tf.reduce_logsumexp(logits_masked_diag, axis=1) 
    #
    # Get the numerator terms (similarities of the positive pairs) | Shape: (num_pairs,)
    l_pos = tf.gather_nd(logits, all_positive_pairs_indices) 
    #
    # Get the corresponding log denominator for each positive pair's row
    log_den_for_pairs = tf.gather(log_den, all_positive_pairs_indices[:, 0])
    #
    # Calculate the loss for each positive pair direction
    pair_losses = -(l_pos - log_den_for_pairs)
    #
    # NOTE: return pair_losses for distributed training else the total_loss
    # -------------------------------------------------------------------------------------
    # Total loss is the average over all positive pairs.
    # total_loss = tf.reduce_mean(pair_losses)
    # return total_loss
    # -------------------------------------------------------------------------------------
    return pair_losses



@tf.function
def sinkhorn_knopp(logits, epsilon=0.05, iterations=3):
    """
    Computes the Sinkhorn-Knopp optimal transport assignment.
    logits: Output of the SwAV prototype layer (batch_size, num_prototypes)
    epsilon: Temperature parameter (usually 0.05 in SwAV)
    iterations: Number of Sinkhorn updates (usually 3)
    """
    # 1. To prevent numerical overflow, subtract the max logit
    # logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

    # 2. Exponentiate and globally normalize
    Q = tf.exp(logits / epsilon)
    Q /= tf.reduce_sum(Q) 

    # Extract dimensions as floats
    K = tf.cast(tf.shape(Q)[1], tf.float32)
    B = tf.cast(tf.shape(Q)[0], tf.float32)
    
    # 3. Sinkhorn iterations
    for _ in range(iterations):
        # Normalize columns (Prototypes) to 1/K
        Q /= tf.reduce_sum(Q, axis=0, keepdims=True) 
        Q /= K

        # Normalize rows (Samples) to 1/B
        Q /= tf.reduce_sum(Q, axis=1, keepdims=True) 
        Q /= B

    # 4. Scale back so the sum of rows equals 1 (Making them valid probability distributions)
    Q *= B
    
    return Q


@tf.function
def swapped_xent_loss(target_codes, predicted_logits, temperature=0.1):
    """
    target_codes: Q matrix from Sinkhorn (computed from Global Views)
    predicted_logits: Logits from Local (or other Global) Views
    temperature: Temperature for softmax (usually 0.1)
    """
    # Log-softmax of the predictions
    log_p = tf.nn.log_softmax(predicted_logits / temperature, axis=1)
    
    # Cross-entropy between Sinkhorn codes and log_p
    # Note: tf.reduce_sum is used here instead of standard categorical_crossentropy
    # because target_codes are soft, dense matrices, not one-hot labels.
    loss = -tf.reduce_mean(tf.reduce_sum(target_codes * log_p, axis=1))
    return loss


@tf.function
def distil_xent_loss(student_logits, teacher_logits, center, student_temp=0.1, teacher_temp=0.04):
    """
    student_logits: (Batch, K)
    teacher_logits: (Batch, K)
    center: (1, K) - The moving average center of the teacher
    """
    # Teacher probabilities (Centered and sharpened)
    teacher_logits = tf.stop_gradient(teacher_logits)
    teacher_probs = tf.nn.softmax((teacher_logits - center) / teacher_temp, axis=1)
    
    # Student log probabilities
    student_logprobs = tf.nn.log_softmax(student_logits / student_temp, axis=1)
    
    # Cross Entropy
    loss = -tf.reduce_sum(teacher_probs * student_logprobs, axis=1)
    return tf.reduce_mean(loss)