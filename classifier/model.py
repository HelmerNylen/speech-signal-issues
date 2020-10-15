import pickle

class Model():
	"""Base class for models"""
	def __init__(self, config: dict, noise_types: list=None):
		raise NotImplementedError()

	# Models can be either single-class (one model is trained per label)
	# or multi-class (one model is trained on all labels)
	# This affects the expected data structures for train_data and test_data
	MULTICLASS = False
	def get_noise_types(self):
		raise NotImplementedError()
	
	def train(self, train_data, index, labels, config: dict=None):
		raise NotImplementedError()
	def score(self, test_data, index):
		raise NotImplementedError()

	@staticmethod
	def from_file(filename: str):
		with open(filename, 'rb') as f:
			return pickle.load(f)
	def save_to_file(self, filename: str):
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def concatenated(data):
		from features.utils import concat_samples
		return concat_samples(data)
	@staticmethod
	def split(data, lengths=None):
		from features.utils import split_samples
		return split_samples(data, lengths)
	@staticmethod
	def is_concatenated(data):
		# TODO: the whole concatenating data, splitting it, and checking it like this is pretty ugly.
		# Would be neater to just always provide one type, and let the model do the necessary conversion if needed.
		return isinstance(data, tuple)