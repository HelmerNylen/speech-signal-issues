import os
import csv
import numpy as np
from .mfcc import mfcc_kaldi, mfcc_librosa
from .utils import root_folder, concat_samples

CACHE_SIZE = 10

def get_available_features() -> tuple:
	feats = (mfcc_kaldi, mfcc_librosa)
	return tuple(f.__name__ for f in feats), feats

def read_dataset(name: str, partition: str, label: str=None) -> tuple:
	if partition.lower() not in ("train", "test"):
		raise ValueError("Partition must be 'train' or 'test'")
	files_root = os.path.join(root_folder(), "datasets", name, partition)
	if not os.path.exists(files_root):
		raise ValueError(f"Dataset does not exist: {name} ({partition})")

	with open(os.path.join(files_root, "labels.csv"), newline='') as f:
		reader = csv.reader(f)
		classes = next(reader)[1:]
		filenames, labels = zip(*((line[0], line[1:]) for line in reader))

	filenames = np.array(filenames)
	classes = np.array(classes)
	labels = np.array(labels) != '0'

	if label is None:
		return filenames, classes, labels
	else:
		try:
			mask = labels[:, np.nonzero(classes == label)]
			return filenames[mask], classes, labels[mask]
		except ValueError as e:
			raise ValueError(f"Label {label} not found in dataset {name}") from e

def _filename(filenames: list, features: list, feature_args: list):
	_hash = hash((tuple(filenames), tuple(features), tuple(tuple(d.items()) for d in feature_args))) 
	return os.path.join(root_folder(), "features", "cache", str(_hash) + ".features")

# TODO: combine features (how?)
def extract_features(filenames: list, features: list, feature_args: list,
		concatenate: bool=False, cache: bool=True) -> list:

	cache_filename = _filename(filenames, features, feature_args)
	if cache and os.path.exists(cache_filename):
		from pickle import load
		with open(cache_filename, "rb") as f:
			cached_tuple, feature_vals = load(f)
		if cached_tuple == (filenames, features, feature_args):
			return feature_vals
		else:
			del cached_tuple, feature_vals

	available_features = get_available_features()
	feature_vals = []
	for feature_ind, feature_name in enumerate(features):
		if feature_name not in available_features[0]:
			raise ValueError(f"Unrecognized feature {feature_name}")
		feature_func = available_features[1][available_features[0].index(feature_name)]

		if feature_ind >= len(feature_args):
			raise ValueError(f"No arguments to feature {feature_ind} ({feature_name}) were provided. "
					+ "If none are needed, specify an empty dict.")
		args = feature_args[feature_ind]

		feature_vals[feature_ind] = feature_func(filenames, **args)
	
	if cache:
		from pickle import dump
		with open(cache_filename, "wb") as f:
			dump(((filenames, features, feature_args), feature_vals), f)

	if concatenate:
		for i in range(len(feature_vals)):
			feature_vals[i] = concat_samples(feature_vals[i])
	
	return feature_vals

def extract_dataset_features(name: str, partition: str, features: list, feature_args: list,
		concatenate: bool=False, cache: bool=True) -> dict:

	filenames, classes, labels = read_dataset(name, partition, None)
	feature_vals = extract_features(filenames, features, feature_args, False, cache)
	res = dict()
	for label in classes:
		mask = labels[:, classes.index(label)]
		f_vals = []
		for i in range(len(feature_vals)):
			f_vals.append([sequence for sequence, bit in zip(feature_vals[i], mask) if bit])
		res[label] = f_vals

	del filenames, feature_vals, classes, labels

	if concatenate:
		for label in res:
			for i in range(len(res[label])):
				res[label][i] = concat_samples(res[label][i])

	return res
