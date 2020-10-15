import os
import csv
import numpy as np
from .mfcc import mfcc_kaldi, mfcc_librosa
from .custom import acf, histogram, rms_energy, rms_energy_infra
from .utils import root_folder, concat_samples

CACHE_SIZE = 20

def get_available_features() -> tuple:
	feats = (mfcc_kaldi, mfcc_librosa, acf, histogram, rms_energy, rms_energy_infra)
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

	filenames = np.array([os.path.join(files_root, fn) for fn in filenames])
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

def _filename(filenames: list, feature: str, feature_args: dict, vad: dict):
	from pickle import dumps
	from hashlib import sha256
	obj = (tuple(filenames), feature, tuple(sorted(feature_args.items())), tuple() if vad is None else tuple(sorted(vad.items())))
	_hash = sha256(dumps(obj)).hexdigest()
	return os.path.join(root_folder(), "features", "cache", str(_hash) + ".features")

def extract_features(filenames: list, feature: str, feature_args: dict, vad: dict=None,
		concatenate: bool=False, cache: bool=True):

	if cache:
		# Lookup features in cache and return them if found
		cache_filename = _filename(filenames, feature, feature_args, vad)

		if os.path.exists(cache_filename):
			from pickle import load
			with open(cache_filename, "rb") as f:
				cached_tuple, feature_vals, index = load(f)
			if (np.array(cached_tuple[0]) == filenames).all() and cached_tuple[1:] == (feature, feature_args, vad):
				return feature_vals, index
			else:
				del cached_tuple, feature_vals, index

	# Get feature function
	available_features = get_available_features()
	if feature not in available_features[0]:
		raise ValueError(f"Unrecognized feature {feature}")
	feature_func = available_features[1][available_features[0].index(feature)]

	if vad is None or len(vad) == 0:
		# Compute features on original files
		index = np.arange(len(filenames))
		feature_vals = feature_func(filenames, **feature_args)
	else:
		# Split files according to VAD output and perform feature extraction on each part
		from .vad import analyze_gen, smooth, splitaudio, write_tempfiles, clean_tempdir
		if "frame_length" not in vad or "aggressiveness" not in vad:
			raise ValueError("Argument 'vad' must contain 'frame_length' and 'aggressiveness'")

		gen = analyze_gen(filenames, vad["frame_length"], vad["aggressiveness"])
		if "smooth_n" in vad:
			gen = (
				(audio, smooth(activity, n=vad["smooth_n"], threshold=vad.get("smooth_treshold")))
					for audio, activity in gen)

		# TODO: this essentially copies the entire dataset to disk, which is ugly
		tempfiles = []
		tempdir = None
		for audio, activity in gen:
			segments = splitaudio(audio, vad["frame_length"], activity, vad.get("inverse", False), vad.get("min_length", 0))
			_tf, tempdir = write_tempfiles(segments, tempdir)
			tempfiles.append(_tf)

		index = np.array([i for i in range(len(tempfiles)) for j in range(len(tempfiles[i]))])

		tempfiles = [t for l in tempfiles for t in l]
		feature_vals = feature_func(tempfiles, **feature_args)
		clean_tempdir(tempdir)
		
	empty = sum(np.prod(v.shape) == 0 for v in feature_vals if isinstance(v, np.ndarray))
	if empty > 0:
		raise ValueError(f"{empty} feature value{'' if empty == 1 else 's'} of dimension 0")
	
	if cache:
		# Remove least recently used cache entries if needed
		cache_files = next(os.walk(os.path.join(root_folder(), "features", "cache")))
		cache_files = [os.path.join(cache_files[0], fn) for fn in cache_files[2] if fn.endswith(".features")]
		cache_files = sorted(cache_files, key=lambda fn: os.stat(fn).st_atime, reverse=True)
		if len(cache_files) > CACHE_SIZE - 1:
			for fn in cache_files[CACHE_SIZE-1:]:
				try:
					os.remove(fn)
				except OSError:
					pass

		from pickle import dump
		with open(cache_filename, "wb") as f:
			dump(((filenames, feature, feature_args, vad), feature_vals, index), f)

	if concatenate:
		feature_vals = concat_samples(feature_vals)
	
	return feature_vals, index

# For compability with older classifiers
def extract_dataset_features(name: str, partition: str, features: list, feature_args: list,
		concatenate: bool=False, cache: bool=True) -> dict:

	filenames, classes, labels = read_dataset(name, partition, None)
	feature_vals, _ = [extract_features(filenames, f, f_a, concatenate=False, cache=cache) for f, f_a in zip(features, feature_args)]
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
