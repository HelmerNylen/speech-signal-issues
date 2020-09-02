import json
import numpy as np

class NoiseClass:
	def __init__(self, id, name, degradations, classifiers=None, classification_settings=None, info=None): # pylint: disable=redefined-builtin
		self.id = id
		self.name = name
		self.degradations = degradations
		self.classifiers = classifiers
		self.classification_settings = classification_settings
		self.info = info

		if self.classifiers is not None:
			if len(self.classifiers) == 0:
				self.classifiers = None
			else:
				self.weights()
	
	def __str__(self):
		return "NoiseClass(" + ", ".join(a + "=" + repr(b) for a, b in vars(self).items()) + ")"
	def __repr__(self):
		return str(self)

	def scores_shape(self, n_sequences):
		if hasattr(n_sequences, "__len__"):
			n_sequences = len(n_sequences)
		return (len(self.classifiers), n_sequences, 2)
	
	def weights(self):
		weights = np.array([float(classifier.get("weight", 1)) for classifier in self.classifiers])
		if np.any(weights < 0):
			raise ValueError("Weights must be >= 0")
		if weights.sum() == 0:
			raise ValueError("Weights must sum to > 0")
		return weights / weights.sum()

	def compound_labels(self, scores):
		if scores.shape != self.scores_shape(scores.shape[1]):
			raise ValueError(f"Expected scores to be {'x'.join(map(str, self.scores_shape(scores.shape[1])))}, got {'x'.join(map(str, scores.shape))}")
		average = "score"
		if self.classification_settings is not None and "average" in self.classification_settings:
			average = self.classification_settings["average"]
		if average == "score":
			return np.argmax(np.sum(scores * self.weights()[:, None, None], axis=0), axis=1) < 0.5
		elif average == "label":
			return np.round(np.sum(np.argmax(scores, axis=2) * self.weights()[:, None], axis=0)) < 0.5
		else:
			raise ValueError(f"Unknown value of classifications_settings.average: {average}")


	@staticmethod
	def from_json(data):
		if isinstance(data, dict):
			if "id" not in data or "degradations" not in data:
				raise ValueError("Noise class definition missing id and/or degradations")

			info = Info() if "info" not in data else Info(
				data["info"].get("description", None),
				data["info"].get("cause", None),
				data["info"].get("solution", None)
			)
			return NoiseClass(
				data["id"],
				data.get("name", data["id"]),
				data["degradations"],
				data.get("classifiers", None),
				data.get("classification_settings", None),
				info
			)
		elif isinstance(data, list):
			return [NoiseClass.from_json(d) for d in data]
		else:
			raise ValueError("Not a dict or list: " + str(data))

	@staticmethod
	def from_file(filename):
		with open(filename, "r") as f:
			res = NoiseClass.from_json(json.load(f))

		if isinstance(res, list):
			res = dict((nc.id, nc) for nc in res)
		return res

class Info:
	def __init__(self, description=None, cause=None, solution=None):
		self.description = description
		self.cause = cause
		self.solution = solution
	def __str__(self):
		return "Info(" + ", ".join(a + "=" + repr(b) for a, b in vars(self).items()) + ")"
	def __repr__(self):
		return str(self)