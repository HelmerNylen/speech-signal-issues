import torch
import os
import sys
import numpy as np
import warnings
from torch.utils.data import DataLoader

from .model import Model

GM_HMM_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gm_hmm")

if os.path.exists(GM_HMM_FOLDER):
	sys.path.append(os.path.abspath(os.path.join(GM_HMM_FOLDER, os.pardir)))
else:
	raise FileNotFoundError("gm_hmm folder does not exist: " + GM_HMM_FOLDER)

try:
	from gm_hmm.src import genHMM
	from gm_hmm.src.utils import TheDataset
except ModuleNotFoundError as e:
	msg = "gm_hmm not found in any of the following folders:\n" + "\n".join(sys.path)
	raise FileNotFoundError(msg) from e

# pylint: disable=super-init-not-called,abstract-method,arguments-differ,signature-differs
class GenHMM(Model):
	def __init__(self, config: dict):
		self.genhmm = None
		if config["parameters"].get("net_D", None) is not None:
			self.init_genhmm(config)

		self.__last_batch_size = None

	def init_genhmm(self, config, net_D=None):
		if net_D is not None:
			config["parameters"]["net_D"] = net_D
		self.genhmm = genHMM.GenHMM(**config["parameters"])
		self.genhmm.iepoch = 1
		self.genhmm.iclass = None
	
	def __create_DataLoader(self, data, batch_size):
		lengths = [sequence.shape[0] for sequence in data]
		maxlen = max(lengths)
		padded_data = [np.pad(sequence, ((0, maxlen - sequence.shape[0]), (0, 0))) for sequence in data]
		return DataLoader(
			TheDataset(padded_data, lengths, device=self.genhmm.device),
			batch_size=batch_size,
			shuffle=True
		)

	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.genhmm.device = device
				self.genhmm.pushto(self.genhmm.device)
				return True
			except: #pylint: disable=bare-except
				print("Unable to push to cuda device. Proceeding on CPU.", file=sys.stderr)
				self.genhmm.device = 'cpu'
				self.genhmm.pushto(self.genhmm.device)

		return False

	# Adaptation of gm_hmm/bin/train_class_gen.py
	def train(self, train_data, index, labels, config):
		# Init data
		if Model.is_concatenated(train_data):
			train_data = Model.split(*train_data)
		train_data = [d for d, l in zip(train_data, labels[index]) if l]
		# TODO: gmmhmm and lstm both have an option to rescale the data to mean 0/var 1,
		# which could be considered for the genhmm as well

		# Init model
		if self.genhmm is None:
			net_D = (train_data[0].shape[1]//2)*2 # Must be even and less than or equal to the number of features
			print("Inferring net_D =", net_D, "from training data")
			self.init_genhmm(config, net_D)

		# Optionally push to GPU
		self.genhmm.device = 'cpu'
		if not config["train"].get("force_cpu", False):
			self.__try_push_cuda()
		print("Using", self.genhmm.device)

		# Dataloader
		self.__last_batch_size = config["train"]["batch_size"]
		data = self.__create_DataLoader(train_data, self.__last_batch_size)
		self.genhmm.number_training_data = len(train_data)

		# Train
		self.genhmm.train()
		for i in range(config["train"]["n_iter"]):
			if "verbose" not in config["train"] or config["train"]["verbose"]:
				print(f"\tIteration {i}")
			self.genhmm.fit(data)
		
		self.genhmm.device = 'cpu'
		self.genhmm.pushto(self.genhmm.device)
		self.genhmm.iepoch += 1
	
	def score(self, test_data, index, batch_size=None, use_gpu=True):
		if Model.is_concatenated(test_data):
			test_data = Model.split(*test_data)
		self.genhmm.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128
		data = self.__create_DataLoader(test_data, batch_size)

		self.genhmm.old_eval()
		self.genhmm.eval()
		scores = torch.cat([self.genhmm.pred_score(x) for x in data]).cpu().numpy()

		res = np.zeros(max(index) + 1)
		lengths = np.zeros(res.shape[0])
		for i, sequence in enumerate(test_data):
			res[index[i]] += scores[i] * sequence.shape[0]
			lengths[index[i]] += sequence.shape[0]

		mask = lengths != 0
		if not mask.all():
			s = (~mask).sum()
			warnings.warn(f"Attempting to label {s} empty feature sequence{'' if s == 1 else 's'}")

		res[mask] = res[mask] / lengths[mask]

		return res