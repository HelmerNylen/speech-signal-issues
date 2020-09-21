import numpy as np
import sys
import warnings
from classifier.model import Model
from sklearn.mixture import GaussianMixture

# pylint: disable=super-init-not-called,abstract-method
class GMM(Model):
	def __init__(self, config: dict):
		self.gmm = GaussianMixture(**config["parameters"])
		self.iepoch = 1
		self.rescale = config["train"].get("rescale_samples", False)
		if self.rescale:
			self.means = None
			self.stddevs = None

	def train(self, train_data, index, labels, config=None):
		if Model.is_concatenated(train_data):
			train_data = Model.split(*train_data)
		train_data = [d for d, l in zip(train_data, labels[index]) if l]
		train_data = Model.concatenated(train_data)
		if not all(length == 1 for length in train_data[1]) and config["train"].get("verbose", True):
			print("The GMM cannot model time-dependence")
		train_data, _ = train_data

		if self.rescale:
			if self.iepoch == 1:
				self.means = np.mean(train_data, axis=0)
				self.stddevs = np.std(train_data, axis=0)
			train_data = (train_data - self.means) / self.stddevs

		self.gmm.fit(train_data)
		self.iepoch += 1
	
	def score(self, test_data, index):
		if Model.is_concatenated(test_data):
			if not all(length == 1 for length in test_data[1]):
				print("The GMM cannot model time-dependence.", file=sys.stderr)

			if self.rescale:
				test_data = ((test_data[0] - self.means) / self.stddevs, test_data[1])
			res = np.zeros(max(index) + 1)
			sums = np.zeros(res.shape[0])
			ptr = 0
			for i, l in enumerate(test_data[1]):
				sequence = test_data[0][ptr:ptr + l, :]
				res[index[i]] = np.sum(self.gmm.score(sequence)) * l
				sums[index[i]] += l
				ptr += l
		else:
			if not all(sequence.shape[0] == 1 for sequence in test_data):
				print("The GMM cannot model time-dependence.", file=sys.stderr)

			res = np.zeros(max(index) + 1)
			sums = np.zeros(res.shape[0])
			if self.rescale:
				for i, sequence in enumerate(test_data):
					res[index[i]] += self.gmm.score((sequence - self.means) / self.stddevs) * sequence.shape[0]
					sums[index[i]] += sequence.shape[0]

			else:
				for i, sequence in enumerate(test_data):
					res[index[i]] += self.gmm.score(sequence) * sequence.shape[0]
					sums[index[i]] += sequence.shape[0]

		mask = sums != 0
		if not mask.all():
			s = (~mask).sum()
			warnings.warn(f"Attempting to label {s} sample{'' if s == 1 else 's'} without any data")

		res[mask] = res[mask] / sums[mask]

		return res
