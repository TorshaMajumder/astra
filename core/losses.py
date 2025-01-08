import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import (sigmoid_cross_entropy_with_logits,
                           softmax_cross_entropy_with_logits)
import numpy as np


tf.config.run_functions_eagerly(True)

def custom_rmse(y_true, y_pred, sample_weight=None, mask=None):
    inp_shp = tf.shape(y_true)
    residuals = tf.square(y_true - y_pred)

    if sample_weight is not None:
        residuals = tf.multiply(residuals, sample_weight)

    if mask is not None:
        residuals = tf.multiply(residuals, mask)
        
    residuals  = tf.reduce_sum(residuals, 1)
    
    mse_mean = tf.math.divide_no_nan(residuals,
                         tf.reduce_sum(mask, 1))
        
    mse_mean = tf.reduce_mean(mse_mean)
    return tf.math.sqrt(mse_mean)

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    num_classes = tf.shape(y_pred)[-1]
    if len(tf.shape(y_pred)) > 2:
        num_steps = tf.shape(y_pred)[1]
        y_one = tf.one_hot(y_true, num_classes)
        y_one = tf.expand_dims(y_one, 1)
        y_one =tf.tile(y_one, [1,num_steps,1])
    else:
        num_steps = tf.shape(y_pred)[-1]
        y_one = tf.one_hot(y_true, num_classes)

    losses = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    if len(tf.shape(y_pred)) > 2:
        losses = tf.transpose(losses)
        losses = tf.reduce_sum(losses, 1)

    return tf.reduce_mean(losses)

def cosine_simmilarity(a, b):
    #
    normalize_a = tf.math.l2_normalize(a,1)
    normalize_b = tf.math.l2_normalize(b,1)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    #
    return distance

def nt_bxent_loss(x, pos_indices, temperature):

    try:
        assert len(x.shape) == 2
        # Add indexes of the principal diagonal elements to pos_indices
        # pos_indices = tf.concat([pos_indices, tf.broadcast_to(tf.reshape(tf.range(x.shape[0]), (x.shape[0],1)), (x.shape[0],2))], axis=0)
        # print(pos_indices, pos_indices.shape)
        
        # Ground truth labels
        # Assign "1" to all positive pairs and "0" to all negative pairs
        # target = np.zeros((x.shape[0], x.shape[0]))
        target_np = np.zeros((x.shape[0], x.shape[0]))
        # print(target_np.shape)
        # Set the values in the NumPy array
        
        target_np[pos_indices[:,0].numpy(), pos_indices[:,1].numpy()] = 1
        # print(target_np[12,12])
        # exit()
        # Create a new TensorFlow tensor from the modified NumPy array
        target = tf.constant(target_np, dtype=tf.float32)
        # Find cosine similarity 
        sim_matrix = cosine_simmilarity(x, x)
        # Set logit of diagonal element to "inf" signifying complete
        # correlation. sigmoid(inf) = 1.0 so this will work out nicely
        # when computing the Binary Cross Entropy Loss.
        # Get indices of True elements
        sim_matrix_copy = tf.identity(sim_matrix)
        diag_indices = tf.where(tf.eye(x.shape[0]).numpy().astype(bool))  
        sim_matrix_updated = tf.tensor_scatter_nd_update(sim_matrix_copy, diag_indices, tf.cast(tf.fill(diag_indices.shape[0], float("inf")), tf.float32))  
        sim_matrix_sigmoid = tf.nn.sigmoid(sim_matrix_updated/temperature)
        # Reshape the matrices and calculate the BCE
        y_true, y_pred = tf.reshape(target, (-1, 1)), tf.reshape(sim_matrix_sigmoid, (-1, 1))
        bce = tf.keras.losses.BinaryCrossentropy(reduction="none", from_logits=False)
        loss = bce(y_true, y_pred)
        # print(loss)
        # Reshape the matrices back to its initial shape
        loss = tf.reshape(loss, (x.shape[0], x.shape[0]))
        y_true, y_pred = tf.reshape(y_true, (x.shape[0],x.shape[0])), tf.reshape(y_pred, (x.shape[0],x.shape[0]))
        # Mask the positive pairs to calculate the individual loss for positive and negative pairs 
        y_true_pos = y_true.numpy().astype(bool)
        y_true_neg = ~y_true_pos
        #
        loss_pos = tf.zeros((x.shape[0],x.shape[0]))  
        loss_neg = tf.zeros((x.shape[0],x.shape[0]))
        #
        loss_pos = tf.tensor_scatter_nd_update(loss_pos, tf.where(y_true_pos), tf.boolean_mask(loss, y_true_pos))
        loss_neg = tf.tensor_scatter_nd_update(loss_neg, tf.where(y_true_neg), tf.boolean_mask(loss, y_true_neg))
        # Calculate the loss mean per row
        # Calculate the number of positive and negative pairs per row
        loss_pos = tf.reduce_sum(loss_pos, axis=1)
        loss_neg = tf.reduce_sum(loss_neg, axis=1)
        #
        num_pos = tf.reduce_sum(y_true, axis=1)
        num_neg = y_true.shape[0] - num_pos
        # calculate the total loss
        total_loss = tf.reduce_mean((tf.cast(loss_pos, tf.float32) / tf.cast(num_pos, tf.float32)) + (tf.cast(loss_neg, tf.float32) / tf.cast(num_neg, tf.float32)))
        #
        return total_loss
    
    except Exception as e:
        print(e, target_np.shape, pos_indices[:,0].numpy(), pos_indices[:,1].numpy())

    
