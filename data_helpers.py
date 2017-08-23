import numpy as np
import re
import itertools
from collections import Counter
import pdb

num_classes = 12

def load_data_and_labels():
    data_file = 'xtrain_obfuscated.txt'
    label = 'ytrain.txt'
    with open(data_file, 'r') as f:
        x_text = np.array(f.read().splitlines())
    with open(label, 'r') as f:
        temp = f.read().splitlines()
    # one hot for y
    y = np.zeros((len(temp), num_classes))
    for idx, val in enumerate(temp):
        y[idx][int(val)] = 1
    return [x_text, y]


#TODO: Need to change
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
