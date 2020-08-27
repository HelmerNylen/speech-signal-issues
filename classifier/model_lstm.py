import torch
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
		self.lstm = None
		self.loss = None
		self.optimizer = None
		self.noise_types = list(noise_types)
		self.device = 'cpu'

		self.__last_batch_size = None

		if "input_dim" in config["parameters"]:
			self.init_lstm(config["parameters"]["input_dim"])

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
			except:
				print("Unable to push to cuda device")
				raise
		return False
	
	def __push_packed_sequence(self, sequence):
		#PackedSequence.to(device) is apparently bugged, see https://discuss.pytorch.org/t/cannot-move-packedsequence-to-gpu/57901
		if self.device == 'cpu':
			return sequence.cpu()
		else:
			return sequence.cuda(device=self.device)
	
	def init_lstm(self, input_dim):
		if self.lstm is not None:
			raise Exception("LSTM already initialized")
		self.lstm = LSTMModule(input_dim, self.hidden_dim, len(self.noise_types),
				self.num_layers, self.dropout)
		self.loss = nn.BCEWithLogitsLoss()
		self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

	def get_noise_types(self):
		return self.noise_types

	def train(self, train_data, labels, config):
		if self.lstm is None:
			self.init_lstm(train_data[0].shape[1])
		self.lstm.train()
		self.device = 'cpu'
		if not config["train"].get("force_cpu", False):
			self.__try_push_cuda()

		if config["train"].get("verbose", False):
			print("Parameters:")
			total = 0
			for param in self.lstm.parameters():
				print(f"\t{type(param.data).__name__}\t{'x'.join(map(str, param.size()))}")
				total += np.prod(param.size())

			print(f"Total: {total} parameters")
			del total

		self.__last_batch_size = config["train"]["batch_size"]
		
		if self.rescale:
			if self.means == None:
				self.means, self.stddevs = LSTMDataset.get_scale_params(train_data)
			train_data = LSTMDataset.rescale(train_data, self.means, self.stddevs)

		dataset = LSTMDataset(train_data, labels)
		dataloader = DataLoader(dataset, batch_size=self.__last_batch_size,
				shuffle=True, collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')
		for i in range(config["train"]["n_iter"]):
			total_loss = 0
			for sequences, labels in dataloader:
				self.optimizer.zero_grad()
				
				predictions = self.lstm(self.__push_packed_sequence(sequences))
				loss = self.loss(predictions, labels.to(self.device))
				loss.backward()
				self.optimizer.step()

				total_loss += loss.detach().data

			if config["train"].get("verbose", False):
				print(f"Iteration {i}: {self.loss.__class__.__name__} = {total_loss:.2f}")
		
		self.device = 'cpu'
		self.lstm.to(self.device)

	def score(self, test_data, batch_size=None, use_gpu=True):
		if self.rescale:
			test_data = LSTMDataset.rescale(test_data, self.means, self.stddevs)
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)

		self.lstm.eval()
		self.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128
		
		dataset = LSTMDataset(test_data, np.zeros((len(test_data), 1)))
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
			collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')
		scores = []
		with torch.no_grad():
			for sequences, _ in dataloader:
				# Maybe want to do a sigmoid before returning scores
				scores.extend(self.lstm(self.__push_packed_sequence(sequences))\
					.cpu().numpy())

		self.device = 'cpu'
		self.lstm.to(self.device)

		return np.array(scores)

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

# TODO: kan nog byta till torch.utils.data.TensorDataset eftersom extrafunktionaliteten är borttagen
class LSTMDataset(Dataset):
	def __init__(self, data, labels):
		# TODO: allow concatenated sequences
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
	labels = torch.tensor(labels, dtype=torch.float32) # Bool may be available on newer pytorch versions
	return sequences, labels