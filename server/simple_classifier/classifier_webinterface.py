import sys
import os.path
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(sys.path[0], os.path.pardir))
from classifier.classifier import Classifier
from features.feature_extraction import extract_features


def analyze_file(f):
	if os.path.splitext(f.name)[1] != ".wav":
		raise ValueError("Invalid file type. Only .wav files supported.")

	with NamedTemporaryFile(dir="test_files", suffix=".wav") as tmp:
		with open(tmp.name, 'wb') as destination:
			for chunk in f.chunks():
				destination.write(chunk)
		

		feats = extract_features([tmp.name], ["mfcc_kaldi"], [{}], cache=False)
		feats = feats[0]
		predictions = {}
		all_scores = {}

		for classifierName in ["latest_GMMHMM.classifier", "latest_LSTM.classifier"]:
			c = Classifier.from_file(os.path.join("../classifier/models/", classifierName))
			(predicted_class,), noise_types, (scores,) = c.label(feats, True)
			predictions[classifierName] = noise_types[predicted_class]
			all_scores[classifierName] = list(zip(noise_types, scores))
		
		return predictions, all_scores
