
import numpy as np
import tensorflow as tf
# from tensorflow.keras.losses import BinaryCrossentropy


LARGE_NUM = 1e9 # Constant for masking out similarities
def nt_xent_loss_3views(z_anchor, z_positive, z_negative, temperature):
    """
    Calculates the NT-XENT loss for three batches of augmented views.
    Assumes z_anchor, z_positive, z_negative are features from the projection head.
    """
    # Normalize embeddings (important for cosine similarity calculation via matmul)
    z_anchor = tf.math.l2_normalize(z_anchor, axis=1)
    z_positive = tf.math.l2_normalize(z_positive, axis=1)
    z_negative = tf.math.l2_normalize(z_negative, axis=1)

    # Get batch size and total size
    batch_size = tf.shape(z_anchor)[0]
    total_size = 3 * batch_size

    # Concatenate all views
    z = tf.concat([z_anchor, z_positive, z_negative], axis=0) # Shape: (3B, D)

    # Calculate cosine similarity matrix
    sim_matrix = tf.matmul(z, z, transpose_b=True) # Shape: (3B, 3B)

    # print(sim_matrix)

    # Mask out diagonal (self-similarity) - crucial for NT-XENT
    diag_mask = tf.logical_not(tf.eye(total_size, dtype=tf.bool))
    # print(diag_mask)
    # sim_matrix = tf.where(diag_mask, sim_matrix, -LARGE_NUM) # Apply mask early? Or during logsumexp

    # --- Identify indices for all positive pairs ---
    indices = tf.range(batch_size)
    # Pairs involving anchor (view 1)
    pairs12 = tf.stack([indices, indices + batch_size], axis=1) # (anc_i, pos_i)
    pairs21 = tf.stack([indices + batch_size, indices], axis=1) # (pos_i, anc_i)
    pairs13 = tf.stack([indices, indices + 2 * batch_size], axis=1) # (anc_i, neg_i)
    pairs31 = tf.stack([indices + 2 * batch_size, indices], axis=1) # (neg_i, anc_i)
    # Pairs involving positive (view 2) and negative (view 3)
    pairs23 = tf.stack([indices + batch_size, indices + 2 * batch_size], axis=1) # (pos_i, neg_i)
    pairs32 = tf.stack([indices + 2 * batch_size, indices + batch_size], axis=1) # (neg_i, pos_i)

    # Combine all positive pair indices
    # Shape: (6 * B, 2)
    all_positive_pairs_indices = tf.concat([pairs12, pairs21, pairs13, pairs31, pairs23, pairs32], axis=0)
    # print(all_positive_pairs_indices)
    # print(all_positive_pairs_indices[:, 0])

    # --- Calculate Loss ---
    # Scaled similarities
    logits = sim_matrix / temperature
    # print(logits)

    # Mask diagonal for denominator calculation
    logits_masked_diag = tf.where(diag_mask, logits, -LARGE_NUM)
    # print(logits_masked_diag)

    # Calculate log denominator (logsumexp over non-diagonal elements row-wise)
    log_den = tf.reduce_logsumexp(logits_masked_diag, axis=1) # Shape: (3B,)
    # print(log_den)

    # Get the numerator terms (similarities of the positive pairs)
    # Use the original logits (without diagonal masking) for the numerator term
    l_pos = tf.gather_nd(logits, all_positive_pairs_indices) # Shape: (6B,)
    # print("1",l_pos)
    # print( tf.expand_dims(l_pos, axis=-1))
    # l_pos = tf.expand_dims(l_pos, axis=-1)
    # log_pos = tf.reduce_logsumexp(l_pos, axis=1) # Shape: (6B,)
    # print("2",log_pos)

    # Get the corresponding log denominator for each positive pair's row
    # The row index of a pair (r, c) is r, which is all_positive_pairs_indices[:, 0]
    log_den_for_pairs = tf.gather(log_den, all_positive_pairs_indices[:, 0]) # Shape: (6B,)
    # print(log_den_for_pairs)

    # Calculate the loss for each positive pair direction
    # loss = - (numerator - log_denominator)
    pair_losses = -(l_pos - log_den_for_pairs)
    # print(pair_losses)

    # Total loss is the average over all 6*B positive pairs
    total_loss = tf.reduce_mean(pair_losses)

    return total_loss


# def cosine_simmilarity(a, b):
#     #
#     normalize_a = tf.math.l2_normalize(a,1)
#     normalize_b = tf.math.l2_normalize(b,1)
#     distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
#     #
#     return distance

# def nt_bxent_loss(x, pos_indices, temperature):

#     try:
#         assert len(x.shape) == 2
#         # Add indexes of the principal diagonal elements to pos_indices
#         # pos_indices = tf.concat([pos_indices, tf.broadcast_to(tf.reshape(tf.range(x.shape[0]), (x.shape[0],1)), (x.shape[0],2))], axis=0)
#         # print(pos_indices, pos_indices.shape)
        
#         # Ground truth labels
#         # Assign "1" to all positive pairs and "0" to all negative pairs
#         # target = np.zeros((x.shape[0], x.shape[0]))
#         target_np = np.zeros((x.shape[0], x.shape[0]))
#         # print(target_np.shape)
#         # Set the values in the NumPy array
        
#         target_np[pos_indices[:,0].numpy(), pos_indices[:,1].numpy()] = 1
#         # print(target_np[12,12])
#         # exit()
#         # Create a new TensorFlow tensor from the modified NumPy array
#         target = tf.constant(target_np, dtype=tf.float32)
#         # Find cosine similarity 
#         sim_matrix = cosine_simmilarity(x, x)
#         # Set logit of diagonal element to "inf" signifying complete
#         # correlation. sigmoid(inf) = 1.0 so this will work out nicely
#         # when computing the Binary Cross Entropy Loss.
#         # Get indices of True elements
#         sim_matrix_copy = tf.identity(sim_matrix)
#         diag_indices = tf.where(tf.eye(x.shape[0]).numpy().astype(bool))  
#         sim_matrix_updated = tf.tensor_scatter_nd_update(sim_matrix_copy, diag_indices, tf.cast(tf.fill(diag_indices.shape[0], float("inf")), tf.float32))  
#         sim_matrix_sigmoid = tf.nn.sigmoid(sim_matrix_updated/temperature)
#         # Reshape the matrices and calculate the BCE
#         y_true, y_pred = tf.reshape(target, (-1, 1)), tf.reshape(sim_matrix_sigmoid, (-1, 1))
#         bce = tf.keras.losses.BinaryCrossentropy(reduction="none", from_logits=False)
#         loss = bce(y_true, y_pred)
#         # print(loss)
#         # Reshape the matrices back to its initial shape
#         loss = tf.reshape(loss, (x.shape[0], x.shape[0]))
#         y_true, y_pred = tf.reshape(y_true, (x.shape[0],x.shape[0])), tf.reshape(y_pred, (x.shape[0],x.shape[0]))
#         # Mask the positive pairs to calculate the individual loss for positive and negative pairs 
#         y_true_pos = y_true.numpy().astype(bool)
#         y_true_neg = ~y_true_pos
#         #
#         loss_pos = tf.zeros((x.shape[0],x.shape[0]))  
#         loss_neg = tf.zeros((x.shape[0],x.shape[0]))
#         #
#         loss_pos = tf.tensor_scatter_nd_update(loss_pos, tf.where(y_true_pos), tf.boolean_mask(loss, y_true_pos))
#         loss_neg = tf.tensor_scatter_nd_update(loss_neg, tf.where(y_true_neg), tf.boolean_mask(loss, y_true_neg))
#         # Calculate the loss mean per row
#         # Calculate the number of positive and negative pairs per row
#         loss_pos = tf.reduce_sum(loss_pos, axis=1)
#         loss_neg = tf.reduce_sum(loss_neg, axis=1)
#         #
#         num_pos = tf.reduce_sum(y_true, axis=1)
#         num_neg = y_true.shape[0] - num_pos
#         # calculate the total loss
#         total_loss = tf.reduce_mean((tf.cast(loss_pos, tf.float32) / tf.cast(num_pos, tf.float32)) + (tf.cast(loss_neg, tf.float32) / tf.cast(num_neg, tf.float32)))
#         #
#         return total_loss
    
#     except Exception as e:
#         print(e, target_np.shape, pos_indices[:,0].numpy(), pos_indices[:,1].numpy())
