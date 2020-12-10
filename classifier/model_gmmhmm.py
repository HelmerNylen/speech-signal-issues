import numpy as np
import warnings
from hmmlearn.base import ConvergenceMonitor
from hmmlearn import hmm
from .model import Model

from sklearn.utils import check_array
from hmmlearn.utils import iter_from_X_lengths

import logging
_log = logging.getLogger(hmm.__name__)

# pylint: disable=super-init-not-called,protected-access,abstract-method,attribute-defined-outside-init
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
		self.limit_inits = (
			config["train"].get("weight_min_init", 0),
			config["train"].get("covar_min_init", 0),
		)
		self.rescale = config["train"].get("rescale_samples", False)
		if self.rescale:
			self.means = None
			self.stddevs = None
	
	def __rand_init(self, train_data):
		self.gmm_hmm._init(*train_data)
		w_add = self.rand_inits[0] * np.random.randn(*self.gmm_hmm.weights_.shape)
		m_add = self.rand_inits[1] * np.random.randn(*self.gmm_hmm.means_.shape)
		c_add = self.rand_inits[2] * np.abs(np.random.randn(*self.gmm_hmm.covars_.shape))
		
		#for attr in ("weights_", "means_", "covars_"):
		#	arr = getattr(self.gmm_hmm, attr)
		#	print("Initial", attr + ":", arr.shape, "from", np.min(arr), "to", np.max(arr))
		#	print(arr)

		if 'w' not in self.gmm_hmm.init_params:
			self.gmm_hmm.weights_ = w_add
			if self.rand_inits[0] == 0:
				self.gmm_hmm.weights_ += 1
		else:
			self.gmm_hmm.weights_ += w_add
		self.gmm_hmm.weights_ = np.abs(self.gmm_hmm.weights_)
		self.gmm_hmm.weights_ = self.gmm_hmm.weights_ + self.limit_inits[0] * self.gmm_hmm.weights_.sum(axis=1)[:, None]
		self.gmm_hmm.weights_ = self.gmm_hmm.weights_ / self.gmm_hmm.weights_.sum(axis=1)[:, None]
		
		if 'm' not in self.gmm_hmm.init_params:
			self.gmm_hmm.means_ = m_add
		else:
			self.gmm_hmm.means_ += m_add

		if 'c' not in self.gmm_hmm.init_params:
			self.gmm_hmm.covars_ = c_add
		else:
			self.gmm_hmm.covars_ += c_add
		self.gmm_hmm.covars_[self.gmm_hmm.covars_ < self.limit_inits[1]] = self.limit_inits[1]

		#for attr in ("weights_", "means_", "covars_"):
		#	arr = getattr(self.gmm_hmm, attr)
		#	print("Final", attr + ":", arr.shape, "from", np.min(arr), "to", np.max(arr))
		#	print(arr)

	def train(self, train_data, index, labels, config=None):
		if Model.is_concatenated(train_data):
			train_data = Model.split(*train_data)
		train_data = [d for d, l in zip(train_data, labels[index]) if l]
		train_data = Model.concatenated(train_data)

		if self.rescale:
			if self.iepoch == 1:
				self.means = np.mean(train_data[0], axis=0)
				self.stddevs = np.std(train_data[0], axis=0)
			train_data = ((train_data[0] - self.means) / self.stddevs, train_data[1])
		
		if self.iepoch == 1:
			self.__rand_init(train_data)

		self.gmm_hmm.fit(train_data[0], lengths=train_data[1])
		self.iepoch += 1
	
	def score(self, test_data, index):
		did_warn = False
		def degenerateWarningFilter(record):
			if record.getMessage() != "Degenerate mixture covariance":
				return True
			else:
				nonlocal did_warn
				did_warn = True
				return False

		_log.addFilter(degenerateWarningFilter)


		if Model.is_concatenated(test_data):
			if self.rescale:
				test_data = ((test_data[0] - self.means) / self.stddevs, test_data[1])
			res = np.zeros(max(index) + 1)
			lengths = np.zeros(res.shape[0])
			ptr = 0
			for i, l in enumerate(test_data[1]):
				sequence = test_data[0][ptr:ptr + l, :]
				# TODO: Should we rather label each sequence and then cast the votes,
				# rather than multiply the scores by the length? Will the length affect the
				# sequence's score anyway?
				res[index[i]] += self.gmm_hmm.score(sequence) * l
				lengths[index[i]] += l
				ptr += l

		else:
			res = np.zeros(max(index) + 1)
			lengths = np.zeros(res.shape[0])
			if self.rescale:
				for i, sequence in enumerate(test_data):
					res[index[i]] += self.gmm_hmm.score((sequence - self.means) / self.stddevs) * sequence.shape[0]
					lengths[index[i]] += sequence.shape[0]
			else:
				for i, sequence in enumerate(test_data):
					res[index[i]] += self.gmm_hmm.score(sequence) * sequence.shape[0]
					lengths[index[i]] += sequence.shape[0]

		mask = lengths != 0
		if not mask.all():
			s = (~mask).sum()
			warnings.warn(f"Attempting to label {s} empty feature sequence{'' if s == 1 else 's'}")

		res[mask] = res[mask] / lengths[mask]

		_log.removeFilter(degenerateWarningFilter)
		if did_warn:
			_log.warning("Degenerate mixture covariance")

		return res

# Override to ensure numerical stability in _do_mstep()
# pylint: disable=unused-variable
class _GMMHMM(hmm.GMMHMM):
	def _do_mstep(self, stats):
		if self.covariance_type == 'diag':
			# Call hmmlearn.base._BaseHMM._do_mstep(stats)
			super(hmm.GMMHMM, self)._do_mstep(stats) #pylint: disable=bad-super-call
			
			# Directly copied from hmm.GMMHMM._do_mstep(stats)
			nc = self.n_components
			nf = self.n_features
			nm = self.n_mix

			n_samples = stats['n_samples']

			# Maximizing weights
			alphas_minus_one = self.weights_prior - 1
			new_weights_numer = stats['post_mix_sum'] + alphas_minus_one
			new_weights_denom = (
				stats['post_sum'] + np.sum(alphas_minus_one, axis=1)
			)[:, np.newaxis]
			new_weights = new_weights_numer / new_weights_denom

			# Maximizing means
			lambdas, mus = self.means_weight, self.means_prior
			new_means_numer = (
				np.einsum('ijk,il->jkl', stats['post_comp_mix'], stats['samples'])
				+ lambdas[:, :, np.newaxis] * mus
			)
			new_means_denom = (stats['post_mix_sum'] + lambdas)[:, :, np.newaxis]
			new_means = new_means_numer / new_means_denom

			# Maximizing covariances
			centered_means = self.means_ - mus
			centered2 = stats['centered'] ** 2
			centered_means2 = centered_means ** 2

			alphas = self.covars_prior
			betas = self.covars_weight

			new_cov_numer = (
				np.einsum('ijk,ijkl->jkl', stats['post_comp_mix'], centered2)
				+ lambdas[:, :, np.newaxis] * centered_means2
				+ 2 * betas
			)
			# In hmm.GMMHMM this is written as 
			# new_cov_denom = stats['post_mix_sum'][:, :, np.newaxis] + 1 + 2 * (alphas + 1)
			# If stats['post_mix_sum'] is close to 0 and alphas equal to -1.5,
			# this would cause new_cov_denom to equal 0 and a subsequent division by zero
			new_cov_denom = (
				stats['post_mix_sum'][:, :, np.newaxis] + 2 * (alphas + 1.5)
			)
			new_cov = new_cov_numer / new_cov_denom

			# Assigning new values to class members
			self.weights_ = new_weights
			self.means_ = new_means
			self.covars_ = new_cov

		else:
			super()._do_mstep(stats)

 	# pylint: disable=redefined-builtin
	def fit(self, X, lengths=None):
		"""Estimate model parameters.

		An initialization step is performed before entering the
		EM algorithm. If you want to avoid this step for a subset of
		the parameters, pass proper ``init_params`` keyword argument
		to estimator's constructor.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.

		lengths : array-like of integers, shape (n_sequences, )
			Lengths of the individual sequences in ``X``. The sum of
			these should be ``n_samples``.

		Returns
		-------
		self : object
			Returns self.
		"""
		X = check_array(X)
		self._init(X, lengths=lengths)
		self._check()

		self.monitor_._reset()
		for iter in range(self.n_iter):
			stats = self._initialize_sufficient_statistics()
			curr_logprob = 0
			for i, j in iter_from_X_lengths(X, lengths):
				framelogprob = self._compute_log_likelihood(X[i:j])
				logprob, fwdlattice = self._do_forward_pass(framelogprob)
				curr_logprob += logprob
				bwdlattice = self._do_backward_pass(framelogprob)
				posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
				try:
					with np.errstate(invalid="raise"):
						self._accumulate_sufficient_statistics(
							stats, X[i:j], framelogprob, posteriors, fwdlattice,
							bwdlattice)
				except FloatingPointError as e:
					print(f"{type(e).__name__}: {e}")
					print("Divergence detected, stopping training")
					return self
					

			# XXX must be before convergence check, because otherwise
			#	 there won't be any updates for the case ``n_iter=1``.
			self._do_mstep(stats)

			self.monitor_.report(curr_logprob)
			if self.monitor_.converged:
				break

		if (self.transmat_.sum(axis=1) == 0).any():
			_log.warning("Some rows of transmat_ have zero sum because no "
						 "transition from the state was ever observed.")

		return self