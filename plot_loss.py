import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.core.util import event_pb2

if __name__ == '__main__':

    print("\nStarting!")
    path_to_train="main-code/runs/ztf/logs/train/"
    path_to_valid="main-code/runs/ztf/logs/valid/"
    path_to_image="main-code/runs/ztf/logs/loss.png"
    path_to_train=f"{path_to_train}{os.listdir(path_to_train)[0]}"
    path_to_valid=f"{path_to_valid}{os.listdir(path_to_valid)[0]}"

    train_loss, val_loss = list(), list()
    
    for event in tf.compat.v1.train.summary_iterator(path_to_train):
    # serialized_examples = tf.data.TFRecordDataset(path_to_train)
    # for serialized_example in serialized_examples:
    #     event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            if value.tag == 'xbent':
                t = tf.make_ndarray(value.tensor)
                train_loss.append(t)

    for event in tf.compat.v1.train.summary_iterator(path_to_valid):
        for value in event.summary.value:
            if value.tag == 'xbent':
                t = tf.make_ndarray(value.tensor)
                val_loss.append(t)

    # plot loss
    # print(val_loss, train_loss)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_to_image)
    print("\nDone!")
                
                

        

    