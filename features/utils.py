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

def check_kaldi_root():
	"""kaldi_io uses an environment variable to find the Kaldi library's root folder.
	If it is invalid or not set, the value of <ROOT_FOLDER>/features/kaldi is set as the Kaldi root."""
	
	if 'KALDI_ROOT' in os.environ and os.path.exists(os.environ['KALDI_ROOT']):
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(root_folder(), "features", "kaldi")
		if os.path.exists(path):
			os.environ['KALDI_ROOT'] = path
			return os.environ['KALDI_ROOT']
		else:
			raise ValueError(f"Invalid environment $KALDI_ROOT and suggested path {path} does not exist.")
