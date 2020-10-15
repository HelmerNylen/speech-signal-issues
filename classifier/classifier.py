import os
import sys
import pickle
import numpy as np
from contextlib import redirect_stdout
from time import time
from .model import Model
from .confusion_table import ConfusionTable


class Classifier:
	"""Serves as an interface for classification using a certain type of model and correspondig configuration"""

	def __init__(self, noise_types: list, model_type: Model, config: dict, silent: bool):
		self.model_type = model_type
		self.config = config[model_type.__name__]
		self.noise_types = tuple(noise_types)

		if silent:
			# Overwrite any verbosity flags
			for category in self.config:
				if "verbose" in self.config[category]:
					self.config[category]["verbose"] = False

		if self.model_type.MULTICLASS:
			# Create a single model for classification, with knowledge of all noise types
			self.model = model_type(noise_types=noise_types, config=self.config)
		else:
			# Create multiple models for classification, each having knowledge of its own noise type
			self.models = tuple(model_type(config=self.config) for noise_type in noise_types)
	
	def train(self, features, index, labels, models_folder=None, silent=False):
		"""Train the classifier's model for one epoch.
		
		If models_folder is specified and the model type is not multi-class,
		an intermediate version is saved during training to prevent loss of progress in case of an error
		
		If silent is set, stdout is redirected to stderr and some informational output is removed"""

		if self.model_type.MULTICLASS:
			with redirect_stdout(sys.stderr if silent else sys.stdout):
				self.model.train(features, index, labels, config=self.config)
		else:
			for noise_index, model in enumerate(self.models):
				if not silent:
					print(f"Training {self.noise_types[noise_index]} {model.__class__.__name__} model")

				with redirect_stdout(sys.stderr if silent else sys.stdout):
					model.train(features, index, labels[:, noise_index], config=self.config)
				
				if models_folder is not None:
					if not silent:
						print("Saving intermediate classifier")
					self.save_to_file(os.path.join(models_folder, f"intermediate_{self.model_type.__name__}.classifier"))

			if not silent:
				print("Removing intermediate file")
			os.remove(os.path.join(models_folder, f"intermediate_{self.model_type.__name__}.classifier"))
	
	def label(self, features, index, return_scores=False, silent=False):
		"""Label a set of feature sequences. Use test() to test performance on a dataset."""

		kwargs = self.config.get("score", dict())
		kwargs = dict() if kwargs is None else kwargs

		with redirect_stdout(sys.stderr if silent else sys.stdout):
			if self.model_type.MULTICLASS:
				scores = self.model.score(features, index, **kwargs)
				predicted_class = np.argmax(scores, axis=1)
				noise_types = np.array(self.model.get_noise_types())

			else:
				scores = []
				for model in self.models:
					scores.append(model.score(features, index, **kwargs))
				scores = np.column_stack(scores)
				predicted_class = np.argmax(scores, axis=1)
				noise_types = np.array(self.noise_types)
		
		if return_scores:
			return predicted_class, noise_types, scores
		else:
			return predicted_class, noise_types
			
	def test(self, features, index, labels, silent=False) -> ConfusionTable:
		"""Test the classifier's performance on a dataset.
		
		Returns a confusion table."""

		res = ConfusionTable(self.noise_types)
		start = time()

		# Classify each sample
		predicted_class, noise_types = self.label(features, index, silent=silent) #pylint: disable=unbalanced-tuple-unpacking

		# TODO: noise_types is not really needed for classification (though some auxiliary scripts use them)
		# so this assert could be cleaned up and removed
		if not sum(a == b for a, b in zip(noise_types, self.noise_types)) == len(self.noise_types):
			raise ValueError("Different ordering of noise types in multiclass model and classifier")

		for i, true_type in enumerate(self.noise_types):
			res[true_type, ConfusionTable.TOTAL] = labels[:, i].sum()
			for j, confused_type in enumerate(self.noise_types):
				res[true_type, confused_type] = (labels[:, i] & (predicted_class == j)).sum()
		
		res.time = time() - start
			
		return res


	def save_to_file(self, filename):
		"""Save a classifier to a file (usually with the .classifier extension).
		
		Note that the config is saved as well."""

		with open(filename, 'wb') as f:
			pickle.dump(self, f)
	
	@staticmethod
	def from_file(filename) -> 'Classifier':
		"""Read a classifier from file"""

		with open(filename, 'rb') as f:
			classifier = pickle.load(f)
			if isinstance(classifier, Classifier):
				return classifier
			else:
				raise ValueError(f"File {filename} does not contain a {Classifier} but a {type(classifier)}")

	@staticmethod
	def from_bytes(b: bytes):
		"""Read classifier(s) from a byte sequence.
		
		This is a generator yielding each classifier detected in the byte sequence."""
		c = pickle.loads(b)
		if isinstance(c, Classifier):
			yield c
		else:
			# Assume iterable
			for classifier in c:
				if not isinstance(classifier, Classifier):
					raise ValueError("Non-classifier in input")
				else:
					yield classifier

	@staticmethod
	def find_classifiers(folder):
		"""Find all files ending with .classifier in the provided folder, non-recursively"""

		files = next(os.walk(folder, followlinks=True))[2]
		return [os.path.join(folder, f) for f in files if os.path.splitext(f)[1] == ".classifier"]
