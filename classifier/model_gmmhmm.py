import numpy as np
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GMMHMM as _GMMHMM
from classifier.model import Model

# pylint: disable=super-init-not-called,protected-access,abstract-method
class GMMHMM(Model):
	def __init__(self, config: dict):
		if "tol" in config["train"] and isinstance(config["train"]["tol"], str):
			config["train"]["tol"] = {"-inf": -np.inf, "inf": np.inf}[config["train"]["tol"]]
		
		self.gmm_hmm = _GMMHMM(**config["parameters"])
		self.gmm_hmm.monitor_ = ConvergenceMonitor(
			*(config["train"][key] for key in ("tol", "n_iter", "verbose"))
		)
		self.iepoch = 1
		self.rand_inits = (
			config["train"].get("weight_rand_init", 0),
			config["train"].get("mean_rand_init", 0),
			config["train"].get("covar_rand_init", 0)
		)
	
	def __rand_init(self, train_data):
		self.gmm_hmm._init(*train_data)
		w_add = self.rand_inits[0] * np.random.randn(*self.gmm_hmm.weights_.shape)
		m_add = self.rand_inits[1] * np.random.randn(*self.gmm_hmm.means_.shape)
		c_add = self.rand_inits[2] * np.abs(np.random.randn(*self.gmm_hmm.covars_.shape))

		if 'w' not in self.gmm_hmm.init_params:
			self.gmm_hmm.weights_ = w_add
			if self.rand_inits[0] == 0:
				self.gmm_hmm.weights_ += 1
		else:
			self.gmm_hmm.weights_ += w_add
		self.gmm_hmm.weights_ = np.abs(self.gmm_hmm.weights_)
		self.gmm_hmm.weights_ = self.gmm_hmm.weights_ / self.gmm_hmm.weights_.sum(axis=1)[:, None]
		
		if 'm' not in self.gmm_hmm.init_params:
			self.gmm_hmm.means_ = m_add
		else:
			self.gmm_hmm.means_ += m_add

		if 'c' not in self.gmm_hmm.init_params:
			self.gmm_hmm.covars_ = c_add
		else:
			self.gmm_hmm.covars_ += c_add

	def train(self, train_data, labels, config=None):
		if Model.is_concatenated(train_data):
			train_data = Model.split(*train_data)
		train_data = [d for d, l in zip(train_data, labels) if l]
		train_data = Model.concatenated(train_data)
		
		if self.iepoch == 1:
			self.__rand_init(train_data)

		self.gmm_hmm.fit(train_data[0], lengths=train_data[1])
		self.iepoch += 1
	
	def score(self, test_data):
		if Model.is_concatenated(test_data):
			res = np.zeros(len(test_data[1]))
			ptr = 0
			for i, l in enumerate(test_data[1]):
				sequence = test_data[0][ptr:ptr + l, :]
				res[i] = self.gmm_hmm.score(sequence)
				ptr += l
			return res
		else:
			return np.array([self.gmm_hmm.score(sequence) for sequence in test_data])