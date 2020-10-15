import torch
import warnings
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
from .model import Model

# pylint: disable=super-init-not-called,signature-differs,arguments-differ
class LSTM(Model):
	MULTICLASS = True
	def __init__(self, config: dict, noise_types: list):
		self.hidden_dim = config["parameters"]["hidden_dim"]
		self.num_layers = config["parameters"]["num_layers"]
		self.dropout = config["parameters"]["dropout"]
		self.learning_rate = config["train"]["learning_rate"]
		self.binary = config["parameters"].get("binary", True)
		self.lstm = None
		self.loss = None
		self.optimizer = None
		self.noise_types = list(noise_types)
		self.device = 'cpu'

		self.__last_batch_size = None

		if "input_dim" in config["parameters"]:
			self.init_lstm(
				config["parameters"]["input_dim"],
				config["parameters"].get("pos_weight", None)
			)

		self.rescale = config["train"].get("rescale_samples", False)
		if self.rescale:
			self.means = None
			self.stddevs = None
	
	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.device = device
				self.lstm.to(device=self.device)
				return True
			except: #pylint: disable=bare-except
				print("Unable to push to cuda device", file=sys.stderr)
				self.device = 'cpu'
				self.lstm.to(device=self.device)

		return False
	
	def __push_packed_sequence(self, sequence):
		#PackedSequence.to(device) is apparently bugged, see https://discuss.pytorch.org/t/cannot-move-packedsequence-to-gpu/57901
		if self.device == 'cpu':
			return sequence.cpu()
		else:
			return sequence.cuda(device=self.device)
	
	def init_lstm(self, input_dim, pos_weight=None):
		if self.lstm is not None:
			raise Exception("LSTM already initialized")
		self.lstm = LSTMModule(input_dim, self.hidden_dim, 1 if self.binary else len(self.noise_types),
				self.num_layers, self.dropout)
		self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

	def get_noise_types(self):
		return self.noise_types

	def train(self, train_data, index, labels, config):
		# Prepare data
		if Model.is_concatenated(train_data):
			train_data = Model.split(*train_data)
		if self.rescale:
			if self.means == None:
				self.means, self.stddevs = LSTMDataset.get_scale_params(train_data)
			train_data = LSTMDataset.rescale(train_data, self.means, self.stddevs)

		# Prepare labels
		labels = labels[index]
		if self.binary:
			assert (labels.sum(axis=1) == 1).all()
			labels = labels[:, 0].reshape(-1, 1)
			pos_weight = (1 - labels).sum() / labels.sum() if labels.any() else 1
		else:
			pos_weight = [((1 - labels[:, i]).sum() / labels[:, i].sum() if labels[:, i].any() else 1) for i in range(labels.shape[1])]
		
		pos_weight = torch.tensor(pos_weight) * config["train"].get("pos_weight_factor", 1)
		
		# Prepare dataloader
		self.__last_batch_size = config["train"]["batch_size"]
		dataset = LSTMDataset(train_data, labels)
		dataloader = DataLoader(dataset, batch_size=self.__last_batch_size,
				shuffle=True, collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')

		# view_examples(dataset, 1)

		# Init model
		if self.lstm is None:
			self.init_lstm(train_data[0].shape[1], pos_weight=pos_weight)
		self.lstm.train()
		self.device = 'cpu'
		if not config["train"].get("force_cpu", False):
			self.__try_push_cuda()

		# Print model structure
		verbose = config["train"].get("verbose", False)
		if verbose:
			print("Parameters:")
			total = 0
			for param in self.lstm.parameters():
				print(f"\t{type(param.data).__name__}\t{'x'.join(map(str, param.size()))}")
				total += np.prod(param.size())

			print(f"Total: {total} parameters")
			del total

		# Train
		stats = {"total_loss": [], "F1": [], "accuracy": []}
		for i in range(config["train"]["n_iter"]):
			total_loss = 0
			tp, tn, fp, fn = 0, 0, 0, 0
			for sequences, _labels in dataloader:
				self.optimizer.zero_grad()
				
				_labels = _labels.to(self.device)

				predictions = self.lstm(self.__push_packed_sequence(sequences))
				loss = self.loss(predictions, _labels)
				loss.backward()
				self.optimizer.step()

				if verbose:
					total_loss += loss.detach().data
					if self.binary:
						_pred = predictions.detach() > 0.5
						_true = _labels > 0.5
					else:
						_pred = torch.log_softmax(predictions.detach(), dim=1).argmax(dim=1)
						_true = _labels.argmax(dim=1)
					tp += (_pred & _true).sum().item()
					tn += (~_pred & ~_true).sum().item()
					fp += (_pred & ~_true).sum().item()
					fn += (~_pred & _true).sum().item()

			if verbose:
				stats["total_loss"].append(total_loss)
				stats["accuracy"].append(100 * (tp + tn) / (tp + tn + fp + fn))
				stats["F1"].append(100 * (2 * tp) / (2 * tp + fp + fn))
				print(f"Iteration {i}: {self.loss.__class__.__name__} = {total_loss:.2f}, accuracy (training set): {stats['accuracy'][-1]/100:.2%}, F1-score (training set): {stats['F1'][-1]/100:.2%}")
		# if verbose:
		# 	view_training_stats(stats.values(), stats.keys(), self.noise_types[0] + " LSTM")
		
		self.device = 'cpu'
		self.lstm.to(self.device)

	def score(self, test_data, index, batch_size=None, use_gpu=True):
		if self.rescale:
			test_data = LSTMDataset.rescale(test_data, self.means, self.stddevs)
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)

		self.lstm.eval()
		self.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128
		
		# During testing we send in the index and sequence length instead of a label
		dataset = LSTMDataset(test_data, list(zip(index, map(len, test_data))))
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
			collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')
		scores = np.zeros((max(index) + 1, len(self.noise_types)))
		lengths = np.zeros(scores.shape[0])

		with torch.no_grad():
			for sequences, idxs_and_lens in dataloader:
				_scores = self.lstm(self.__push_packed_sequence(sequences))
				if self.binary:
					_scores = torch.cat((_scores-1, -_scores), dim=1).cpu().numpy()
				else:
					_scores = torch.log_softmax(_scores, dim=1).cpu().numpy()

				for score, idx, l in zip(_scores, idxs_and_lens[:, 0], idxs_and_lens[:, 1]):
					# TODO: Not sure if we should really multiply by the length here
					scores[int(idx)] += score * int(l)
					lengths[int(idx)] += int(l)


		mask = lengths != 0
		if not mask.all():
			s = (~mask).sum()
			warnings.warn(f"Attempting to label {s} empty feature sequence{'' if s == 1 else 's'}")

		scores[mask, :] = scores[mask, :] / lengths[mask, None]

		self.device = 'cpu'
		self.lstm.to(self.device)

		return scores

class LSTMModule(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
		super(LSTMModule, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim,
			num_layers=num_layers, dropout=dropout, batch_first=True)
		self.linear = nn.Linear(self.hidden_dim, output_dim)
		#self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def forward(self, data):
		x, _ = self.lstm(data)
		x, l = pad_packed_sequence(x, batch_first=True)
		x = x[range(x.shape[0]), l-1, :]
		x = self.linear(x)
		#x = self.logsoftmax(x)
		return x

class LSTMDataset(Dataset):
	def __init__(self, data, labels):
		assert not Model.is_concatenated(data)
		assert len(data) == len(labels)
		self.data = data
		self.labels = labels
		
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	
	def __len__(self):
		return len(self.data)

	@staticmethod
	def get_scale_params(data):
		if Model.is_concatenated(data):
			return (
				np.mean(data[0], axis=0),
				np.std(data[0], axis=0)
			)
		else:
			d, _ = Model.concatenated(data)
			return (
				np.mean(d, axis=0),
				np.std(d, axis=0)
			)

	@staticmethod
	def rescale(data, means, stddevs):
		if Model.is_concatenated(data):
			return ((data[0] - means) / stddevs, data[1])
		else:
			return [sequence - means / stddevs for sequence in data]

def LSTMCollate(samples):
	sequences, labels = zip(*samples)
	sequences = pack_sequence([torch.from_numpy(s) for s in sequences], enforce_sorted=False)
	labels = torch.tensor(labels, dtype=torch.float32)
	return sequences, labels



# -- Debug functions --

def view_training_stats(data, legend, title=None):
	from matplotlib import pyplot as plt
	import time, os
	plt.figure()
	for series, label in zip(data, legend):
		plt.plot(range(len(series)), series, label=label)
	plt.legend()
	plt.xlabel("t (frames)")
	plt.ylabel("Statistic value")
	if title:
		plt.title(title)
	plt.savefig(os.path.join(os.getcwd(), "testresults", time.strftime(title + " %Y-%m-%d %H:%M:%S.png", time.localtime())))

def view_examples(dataset, n=2):
	from matplotlib import pyplot as plt
	import random, os
	r = str(random.randint(1, 1000))
	print(os.getcwd(), r)

	def _plot(vec, lbl, i):
		plt.figure()
		for j in range(vec.shape[1]):
			plt.plot(np.arange(vec.shape[0]), vec[:, j], label="Dimension " + str(j))
		plt.legend()
		plt.xlabel("t (frames)")
		plt.ylabel("Feature value")
		t = "".join(str(int(k)) for k in lbl) + "-" + str(i)
		plt.title(t)
		with open(os.path.join(os.getcwd(), "testresults", r + "-" + t + ".png"), "wb") as f:
			plt.savefig(f)

	n_true, n_false = 0, 0
	for i in np.random.permutation(len(dataset)):
		vec, lbl = dataset[i]
		if i == 0:
			print(vec.shape)
			if vec.shape[1] == 13:
				break
		if lbl[0] and n_true < n:
			_plot(vec, lbl, i)
			n_true += 1
		elif not lbl[0] and n_false < n:
			_plot(vec, lbl, i)
			n_false += 1
		elif n_true >= n and n_false >= n:
			break