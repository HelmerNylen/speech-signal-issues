#!/usr/bin/env python3
import argparse
import os
import sys
import json
import numpy as np
import pickle
from time import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
DATASETS = os.path.join(PROJECT_ROOT, "datasets")

sys.path.append(PROJECT_ROOT)
from noise_classes.noise_class import NoiseClass #pylint: disable=import-error
from features.feature_extraction import read_dataset, extract_features
from classifier.classifier import Classifier
from classifier.confusion_table import ConfusionTable

from classifier.model_lstm import LSTM
from classifier.model_gmmhmm import GMMHMM
from classifier.model_genhmm import GenHMM
from classifier.model_gmm import GMM
avaliable_models = (LSTM, GMMHMM, GenHMM, GMM)
setting_categories = ("parameters", "train", "score")

def train(args):
	# Read dataset source.json
	if not os.path.exists(os.path.join(DATASETS, args.dataset_name)):
		print(f"Dataset {args.dataset_name} does not exist", file=sys.stderr)
		sys.exit(1)
	dataset_source_fname = os.path.join(DATASETS, args.dataset_name, "source.json")
	with open(dataset_source_fname, "r") as f:
		dataset_source = json.load(f)
	if "weights" not in dataset_source:
		print(f"Dataset {args.dataset_name} was not created using noise classes.", file=sys.stderr)
		print("An attempt will be made to match the labels to existing classes.", file=sys.stderr, flush=True)
		dataset_source["weights"] = dict((label, 1 / len(dataset_source["labels"])) for label in dataset_source["labels"])
	
	if args.update is not None:
		noise_classes_old = load_noise_classes(args, False)
		if noise_classes_old is None:
			args.update = None
	
	# Iterate over noise classes
	print("Initializing classifiers")
	noise_classes = NoiseClass.from_file(args.noise_classes)
	default_settings = None
	for label in dataset_source["weights"]:
		if label not in noise_classes:
			print(f"Label {label} of dataset {args.dataset_name} is not among the defined noise classes in {args.noise_classes}", file=sys.stderr, flush=True)
			continue

		nc = noise_classes[label]
		if nc.classifiers is None:
			# Pseudo-class
			continue

		train_all = True
		if args.update is not None and nc.id in noise_classes_old:
			nc_old = noise_classes_old[nc.id]
			train_all = False
			if json.dumps(nc.degradations) != json.dumps(nc_old.degradations):
				print(f"Warning: the degradation definition of {nc.id} has changed.", file=sys.stderr)
				print("If you have not already, please generate a new dataset using degradations/create_dataset.py", file=sys.stderr, flush=True)
				print(f"All classifiers in {nc.id} will be retrained from scratch")
				train_all = True
		
		print(label + ":")
		
		for classifier_ind, classifier_spec in enumerate(nc.classifiers):
			print(f"\t {classifier_spec['type']} (feature: {classifier_spec['feature']}", end="")
			if len(nc.classifiers) > 1:
				print(", weight:", classifier_spec.get('weight', 1), end="")
			if "bootstrap" in classifier_spec:
				print(", bootstrap:", classifier_spec["bootstrap"], end="")
			if classifier_spec.get("vad", None):
				print(", VAD:", "unvoiced" if classifier_spec["vad"].get("inverse", False) else "voiced", end="")
			print(")")

			# Setup classifier specifications
			classifier_complete_defaults(classifier_spec, args.classifier_defaults, default_settings)

			# Initialize or copy old classifier
			_type = next((m for m in avaliable_models if m.__name__ == classifier_spec["type"]), None)
			if _type is None:
				print(f"Unrecognized classifier type {classifier_spec['type']}", file=sys.stderr)
				sys.exit(1)
			config = {
				_type.__name__: dict((cat, classifier_spec[cat]) for cat in setting_categories if classifier_spec[cat] is not None)
			}

			if args.update is not None and nc.id in noise_classes_old and classifier_ind < len(nc_old.classifiers):
				classifier_spec_old = nc_old.classifiers[classifier_ind]
				if not train_all and nc.id not in args.update \
						and classifier_specs_equal(classifier_spec, classifier_spec_old):
					classifier_spec["instance"] = classifier_spec_old["instance"]
					classifier_spec["notrain"] = True
					continue

			classifier_spec["instance"] = Classifier([nc.id, nc.id + " (negative)"], _type, config, silent=False)
	
	# Prune noise_classes
	for nc_id in tuple(noise_classes.keys()):
		classifiers = noise_classes[nc_id].classifiers
		if classifiers is None or sum(1 for spec in classifiers if spec.get("instance") is not None) == 0:
			del noise_classes[nc_id]
	
	# Check if skipping training is allowed
	if args.update is not None and set(noise_classes.keys()) != set(noise_classes_old.keys()):
		print("The set of noise classes have changed", file=sys.stderr)
		print("All classifiers for all noise classes must be retrained from scratch", file=sys.stderr, flush=True)
		for nc in noise_classes.values():
			for spec in nc.classifiers:
				if "notrain" in spec:
					del spec["notrain"]

	# Train classifiers grouped by feature
	rng = None
	spec_inds_sorted = _sort_spec_inds(noise_classes)
	filenames, classes, labels = read_dataset(args.dataset_name, "train")
	for i, (spec, nc, feats, idxs) in enumerate(_iterate_classifiers(spec_inds_sorted, filenames, args.recompute)):
		print(f"Training ({i + 1}/{len(spec_inds_sorted)})")
		if args.update is not None and spec.get("notrain", False):
			print("Keeping old classifier")
			del spec["notrain"]
			continue

		label_ind, = np.where(classes == spec["instance"].noise_types[0])[0]
		labels_binary = labels[:, label_ind]
		labels_binary = np.column_stack((labels_binary, ~labels_binary))
		
		# Bootstrapping
		if spec.get("bootstrap", False):
			if len(nc.classifiers) == 1:
				print("Warning: Bootstrapping a single classifier - please use model averaging or the entire training set.", file=sys.stderr, flush=True)
			if rng is None:
				rng = np.random.default_rng()
			sample_inds = rng.choice(np.arange(len(filenames)), len(filenames))
		 
			feats_used = [feats[feat_ind] for sample_ind in sample_inds for feat_ind in np.where(idxs == sample_ind)[0]]
			idxs_used = [idx for idx, sample_ind in enumerate(sample_inds) for _ in range(np.sum(idxs == sample_ind))]
			labels_used = labels_binary[sample_inds, :]

			#for li in range(len(sample_inds)):
			#	for fi in np.where(idxs_used == idxs_used[li])[0]:
			#		assert labels_used[fi] == labels_binary[sample_inds[li]]
		else:
			feats_used = feats
			idxs_used = idxs
			labels_used = labels_binary

		spec["instance"].train(feats_used, idxs_used, labels_used, args.models)
	
	print("Training complete")

	# Save
	fname = os.path.join(args.models, args.dataset_name + ".noiseclasses")
	if os.path.exists(fname):
		print("Overwriting", fname)
	with open(fname, "wb") as f:
		pickle.dump(noise_classes, f)
	print("Saved to", fname)

def classifier_specs_equal(a, b):
	return all(json.dumps(a[field]) == json.dumps(b[field])
			for field in setting_categories + ("type", "feature", "feature_settings")) \
		and all(json.dumps(a.get(field, False)) == json.dumps(b.get(field, False))
			for field in ("bootstrap", "vad"))

def classifier_complete_defaults(classifier_spec, defaults_file, default_settings=None):
	for category in setting_categories:
		if classifier_spec.get(category, "default") == "default":
			# Load defaults if needed
			if default_settings is None:
				if not os.path.exists(defaults_file):
					print(f"Classifier defaults file {defaults_file} does not exist", file=sys.stderr)
					sys.exit(1)
				with open(defaults_file, "r") as f:
					default_settings = json.load(f)

			classifier_spec[category] = default_settings[classifier_spec["type"]].get(category)

def _feat_id(spec):
	return spec["feature"] + "/" + repr(spec["feature_settings"]) + ("/" + repr(spec["vad"]) if spec.get("vad", None) not in (None, {}) else "")
def _sort_spec_inds(noise_classes):
	return sorted(
		((spec_ind, nc) for _, nc in noise_classes.items() for spec_ind, spec in enumerate(nc.classifiers)),
		key=lambda tup: _feat_id(tup[1].classifiers[tup[0]])
	)
def _iterate_classifiers(spec_inds_sorted, filenames, recompute, silent=False, yield_ind=False):
	last_feat_id = None
	for spec_ind, nc in spec_inds_sorted:
		spec = nc.classifiers[spec_ind]
		if _feat_id(spec) != last_feat_id:
			if not silent:
				print(f"Extracting {spec['feature']} features (use cache: {['yes', 'no'][recompute]})")
				print(f"\tSettings: {spec['feature_settings']}")
				print(f"\tVAD settings: {spec.get('vad', 'VAD not used')}")
			feats, idxs = extract_features(
				filenames,
				spec["feature"], spec["feature_settings"],
				spec.get("vad", None),
				concatenate=False,
				cache=not recompute
			)
			last_feat_id = _feat_id(spec)

		yield (spec_ind if yield_ind else spec), nc, feats, idxs

def load_noise_classes(args, exit_if_missing=True):
	fname = os.path.join(args.models, args.dataset_name + ".noiseclasses")
	if not os.path.exists(fname):
		print(f"File {fname} does not exist", file=sys.stderr)
		if exit_if_missing:
			sys.exit(1)
		else:
			return None

	with open(fname, "rb") as f:
		noise_classes = pickle.load(f)
	return noise_classes

def stats(predicted_labels, nc_ids, true_labels, classes):
	mapping = np.array(list(np.where(classes == nc_id)[0][0] for nc_id in nc_ids))
	assert predicted_labels.shape == (true_labels.shape[0], len(mapping))

	confusion_tables = dict()
	for i, nc_id in enumerate(nc_ids):
		ct = ConfusionTable((nc_id, nc_id + " (negative)"), (nc_id, nc_id + " (negative)"))
		ct[0, 0] = np.sum(predicted_labels[:, i] & true_labels[:, mapping[i]])
		ct[0, 1] = np.sum(~predicted_labels[:, i] & true_labels[:, mapping[i]])
		ct[0, ...] = ct[0, 0] + ct[0, 1]
		ct[1, 0] = np.sum(predicted_labels[:, i] & ~true_labels[:, mapping[i]])
		ct[1, 1] = np.sum(~predicted_labels[:, i] & ~true_labels[:, mapping[i]])
		ct[1, ...] = ct[1, 0] + ct[1, 1]
		confusion_tables[nc_id] = ct

	hamming = 0
	intersection = 0
	union = 0
	tot_predicted_labels = 0
	tot_true_labels = 0
	exact = 0

	for filename_ind in range(predicted_labels.shape[0]):
		g = predicted_labels[filename_ind, :]
		t = true_labels[filename_ind, mapping]
		hamming              += np.sum(g ^ t)
		tot_predicted_labels += np.sum(g)
		tot_true_labels      += np.sum(t)
		intersection         += np.sum(g & t)
		union                += np.sum(g | t)
		exact                += np.all(g == t)

	hamming /= predicted_labels.shape[0] * predicted_labels.shape[1]
	jaccard = intersection / union
	precision = intersection / tot_predicted_labels if tot_predicted_labels > 0 else 0
	recall = intersection / tot_true_labels
	f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
	exact /= predicted_labels.shape[0]

	stats_dict = dict((
		("Hamming loss", hamming),
		("Jaccard index", jaccard),
		("Precision", precision),
		("Recall", recall),
		("F1-score", f1_score),
		("Subset accuracy", exact),
		("Predicted label cardinality", tot_predicted_labels / predicted_labels.shape[0]),
		("True label cardinality", tot_true_labels / true_labels.shape[0])
	))

	return confusion_tables, stats_dict
	
def test(args):
	if not os.path.exists(os.path.join(DATASETS, args.dataset_name)):
		print(f"Dataset {args.dataset_name} does not exist", file=sys.stderr)
		sys.exit(1)

	noise_classes = load_noise_classes(args)
	
	# Extract features and test each classifier
	filenames, classes, labels = read_dataset(args.dataset_name, "test")
	if not args.skip_classifiers:
		print("Testing individual classifiers")
		for nc in noise_classes.values():
			print(nc.name)
			for spec in nc.classifiers:
				feats, idxs = extract_features(
					filenames,
					spec["feature"], spec["feature_settings"],
					spec.get("vad", None),
					concatenate=False,
					cache=not args.recompute
				)
				
				label_ind, = np.where(classes == spec["instance"].noise_types[0])[0]
				labels_binary = labels[:, label_ind]
				labels_binary = np.column_stack((labels_binary, labels_binary != 1))
				print(spec["instance"].test(feats, idxs, labels_binary))
				print()
	
	# Stats for the noise class based labeling
	args.filenames = filenames
	predicted_labels, nc_ids = get_labels(args, noise_classes)

	print("Combined classifier statistics")
	confusion_tables, stats_dict = stats(predicted_labels, nc_ids, labels, classes)

	for nc_id, ct in confusion_tables.items():
		print(noise_classes[nc_id].name)
		if not args.skip_classifiers and len(noise_classes[nc_id].classifiers) == 1:
			print("(see above)")
			print()
			continue

		print(ct)
		print()

	print("Multilabel statistics")
	for name, value in stats_dict.items():
		print(f"{name + ':  ':<18}{value:.3}")
		

def get_labels(args, noise_classes=None):
	noise_classes = load_noise_classes(args) if noise_classes is None else noise_classes

	# Score all files for all classifiers
	all_scores = dict(
		(nc_id, np.zeros(nc.scores_shape(len(args.filenames))))
			for nc_id, nc in noise_classes.items()
	)
	for spec_ind, nc, feats, idxs in _iterate_classifiers(
			_sort_spec_inds(noise_classes),
			args.filenames,
			args.recompute,
			silent=args.silent,
			yield_ind=True):
		classifier = nc.classifiers[spec_ind]["instance"]
		_, _, scores = classifier.label(feats, idxs, return_scores=True, silent=args.silent)
		all_scores[nc.id][spec_ind] = scores

	# Compute labels from the scores
	labels = np.column_stack(tuple(nc.compound_labels(all_scores[nc_id]) for nc_id, nc in noise_classes.items()))
	return (labels, tuple(nc_id for nc_id in noise_classes))

def do_label(args):
	labels, nc_ids = get_labels(args)
	if args.silent:
		print(json.dumps(((1 * labels).tolist(), nc_ids), separators=(',', ':')))
	else:
		for i, filename in enumerate(args.filenames):
			print(filename)
			for j, nc_id in enumerate(nc_ids):
				print(nc_id + ":", ["[ ]", "[x]"][int(labels[i, j])], end="  ")
			print()

def check(args):
	if not os.path.exists(args.noise_classes):
		print(f"Noise classes file {args.noise_classes} does not exist", file=sys.stderr)
		nc_defs = None
	else:
		nc_defs = NoiseClass.from_file(args.noise_classes)

	dataset_source = None
	if not os.path.exists(os.path.join(DATASETS, args.dataset_name)):
		print(f"Dataset {args.dataset_name} does not exist", file=sys.stderr)
	else:
		dataset_source_fname = os.path.join(DATASETS, args.dataset_name, "source.json")
		with open(dataset_source_fname, "r") as f:
			dataset_source = json.load(f)

	nc_trained = load_noise_classes(args, False)
	
	# Test if noise classes definition file and dataset source file agree on degradations
	defs_dataset_ok = None
	if nc_defs is not None and dataset_source is not None:
		def_nc_degs = dict((nc_id, nc_defs[nc_id].degradations) for nc_id in dataset_source["noise_class_degradations"] if nc_id in nc_defs)
		defs_dataset_ok = def_nc_degs == dataset_source["noise_class_degradations"]
	
	# Test if dataset source file and stored classifiers agree on noise class ids and degradations
	dataset_trained_ok = None
	if dataset_source is not None and nc_trained is not None:
		if set(dataset_source["weights"].keys()) == set(nc_trained.keys()):
			trained_nc_degs = dict((nc_id, nc_trained[nc_id].degradations) for nc_id in dataset_source["noise_class_degradations"])
			dataset_trained_ok = dataset_source["noise_class_degradations"] == trained_nc_degs
		else:
			dataset_trained_ok = False
	
	# Test if noise classes definition file and stored classifiers agree on classifier specs
	defs_trained_ok = None
	if nc_defs is not None and nc_trained is not None:
		default_settings = None
		for nc in nc_trained.values():
			if nc.id not in nc_defs or len(nc_defs[nc.id].classifiers) != len(nc.classifiers):
				defs_trained_ok = False
				break

			for classifier_ind, spec_trained in enumerate(nc.classifiers):
				spec_def = nc_defs[nc.id].classifiers[classifier_ind]
				classifier_complete_defaults(spec_def, args.classifier_defaults, default_settings)

				if not classifier_specs_equal(spec_trained, spec_def):
					defs_trained_ok = False
					break
			if nc.classification_settings != nc_defs[nc.id].classification_settings:
				defs_trained_ok = False

			if defs_trained_ok is False:
				break
		else:
			defs_trained_ok = True
	
	return defs_dataset_ok, dataset_trained_ok, defs_trained_ok

def do_check(args):
	defs_dataset_ok, dataset_trained_ok, defs_trained_ok = check(args)
	if args.silent:
		print(json.dumps((defs_dataset_ok, dataset_trained_ok, defs_trained_ok)))
	else:
		if defs_dataset_ok is False:
			print("Discrepancy between noise classes definitions and dataset detected")
			print("Suggested action: generate a new dataset")
		if dataset_trained_ok is False:
			print("Discrepancy between dataset and trained noise classes detected")
			print("Suggested action: retrain classifiers on the new dataset")
		if defs_trained_ok is False:
			print("Discrepancy between noise classes definitions and trained noise classes detected")
			print("Suggested action: retrain classifiers with the new settings")
		if all((defs_dataset_ok, dataset_trained_ok, defs_trained_ok)):
			print("No discrepancies detected")
		# Other cases (where one or more test is None) are reported to sys.stderr from check()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="interface.py",
		description="Train and test classifiers for noise classes"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--models", help="Path to classifier models save folder (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "classifier", "models"))

	parser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")


	subparser = subparsers.add_parser("train", help="Perform training, and save the result to the models folder")
	subparser.set_defaults(func=train)

	subparser.add_argument("dataset_name", help="Name of the dataset to train on", metavar="MyDataset")

	subparser.add_argument("--noise-classes", help="Path to noise class definitions (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "noise_classes", "noise_classes.json"), metavar="myfile.json")
	subparser.add_argument("--classifier-defaults", help="Path to classifier default settings (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "classifier", "defaults.json"), metavar="myfile.json")
	subparser.add_argument("-u", "--update", help="Update previously trained classifiers with changes to the noise class definitions. If no classifiers are found this argument is ignored. If no changes to the definitions have been made, no classifiers are changed. If any noise class IDs are provided, the classifiers corresponding these IDs are always retrained.",
			nargs="*", metavar="ID")


	subparser = subparsers.add_parser("test", help="Perform testing")
	subparser.set_defaults(func=test, silent=False)

	subparser.add_argument("dataset_name", help="Name of the dataset to test on", metavar="MyDataset")

	subparser.add_argument("-s", "--skip-classifiers", help="Do not test the individual classifiers on each noise class - only display the summarized stats", action="store_true")


	subparser = subparsers.add_parser("label", help="Label files for which the true noise classes are not known")
	subparser.set_defaults(func=do_label)

	subparser.add_argument("dataset_name", help="Name of the dataset on which the noise classes used for labeling are trained", metavar="MyDataset")
	subparser.add_argument("filenames", help="The files that are to be labeled", metavar="file.wav", nargs="+")
	
	subparser.add_argument("-s", "--silent", help="Output on stdout only the results of the labeling, as a JSON-serialized array and an index", action="store_true")


	subparser = subparsers.add_parser("check", help="Check that the noise classes file, the dataset and the trained models are up-to-date with each other")
	subparser.set_defaults(func=do_check)

	subparser.add_argument("dataset_name", help="Name of the dataset to check", metavar="MyDataset")

	subparser.add_argument("--noise-classes", help="Path to noise class definitions (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "noise_classes", "noise_classes.json"), metavar="myfile.json")
	subparser.add_argument("--classifier-defaults", help="Path to classifier default settings (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "classifier", "defaults.json"), metavar="myfile.json")
	subparser.add_argument("-s", "--silent", help="Output on stdout only the results of the check, as a JSON-serialized array", action="store_true")

	args = parser.parse_args()
	start = time()
	args.func(args)
	if not vars(args).get("silent", False):
		print(f"Total time: {time() - start:.1f} s")