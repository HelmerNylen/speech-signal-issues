import numpy as np
import os

# Combine features sequences into one big matrix and a list of sequence lengths
def concat_samples(data):
	return np.concatenate(data), [arr.shape[0] for arr in data]

# Split the sequences into a list of individual sequence matrices
def split_samples(data, lengths=None):
	if lengths == None:
		data, lengths = data
	return np.split(data, np.cumsum(lengths[:-1]))

def root_folder():
	return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))