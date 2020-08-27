import sys
import os.path
import numpy as np
import subprocess
import traceback
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(sys.path[0], os.path.pardir))
from classifier.classifier import Classifier
from features.feature_extraction import extract_features

root = os.path.join(os.getcwd(), os.path.pardir)
tempfiles = os.path.join(root, "server", "test_files")
models = os.path.join(root, "classifier", "models")

classifier_types = subprocess.run(
	[os.path.join(root, "classifier_interface.py"), "available-types"],
	check=True,
	stdout=subprocess.PIPE,
	encoding=sys.getdefaultencoding()
).stdout.splitlines()

class Analysis:
	def __init__(self, name, noise_types=None, scores=None, prediction=None, error=None):
		self.name = name
		if self.name in map(lambda c: "latest_" + c + ".classifier", classifier_types):
			self.name = "Latest " + self.name[len("latest_"):-len(".classifier")]
		elif self.name.endswith(".classifier"):
			self.name = self.name[:-len(".classifier")]
		self.noise_types = noise_types
		self.scores = scores
		if self.noise_types is None:
			self.prediction = None
		else:
			self.prediction = self.noise_types[prediction if prediction is not None else np.argmax(scores)]
		self.error = error

def analyze_file(f):
	if os.path.splitext(f.name)[1] != ".wav":
		raise ValueError("Invalid file type. Only .wav files supported.")

	with NamedTemporaryFile(dir=tempfiles, suffix=".wav") as tmp:
		with open(tmp.name, 'wb') as destination:
			for chunk in f.chunks():
				destination.write(chunk)
		

		feats = extract_features([tmp.name], ["mfcc_kaldi"], [{}], cache=False)
		feats = feats[0]
		analyses = []

		for classifierName in Classifier.find_classifiers(models):
			try:
				c = Classifier.from_file(os.path.join(models, classifierName))
				(predicted_class,), noise_types, (scores,) = c.label(feats, True)
				analyses.append(Analysis(os.path.basename(classifierName), noise_types, scores, predicted_class))

			except: #pylint: disable=bare-except
				analyses.append(Analysis(classifierName, error=traceback.format_exc()))
		
		return analyses
