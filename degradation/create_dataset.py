#!/usr/bin/env python3
import os
import sys
import argparse
import math
import csv
import json
from time import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def _noise_files(noise_folder, keep=(".wav",)):
	walk = os.walk(noise_folder, followlinks=True)
	# Skip .WAV (uppercase) files as these are unprocessed timit sphere files (i.e. don't use os.path.splitext(fn)[-1].lower())
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1] in keep])
			for dp, dn, filenames in walk)
	walk = ((os.path.relpath(dirpath, noise_folder), (dirpath, filenames))
			for dirpath, dirnames, filenames in walk
			if dirpath != noise_folder and len(filenames) > 0)
	return walk

def _speech_files(speech_folder, keep=(".wav",)):
	walk = os.walk(speech_folder, followlinks=True)
	# Skip .WAV (uppercase) files as these are unprocessed timit sphere files (i.e. don't use os.path.splitext(fn)[-1].lower())
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1] in keep])
			for dp, dn, filenames in walk)
	walk = ((dirpath, filenames)
			for dirpath, dirnames, filenames in walk
			if len(filenames) > 0)
	return walk

def read_noise(noise_folder: str) -> tuple:
	noise = dict(_noise_files(noise_folder))
	n_noisefiles = sum(len(noise[noisetype][1]) for noisetype in noise)
	return noise, n_noisefiles

def read_speech(speech_folder: str) -> tuple:
	speech = list(_speech_files(speech_folder))
	n_speechfiles = sum(len(f) for d, f in speech)
	return speech, n_speechfiles

def _filenames(folder: str, num: int, ext: str):
	for i in range(1, num + 1):
		yield os.path.join(folder, str(i).rjust(math.ceil(math.log10(num)), '0') + ext)

def _labels(_type: dict) -> list:
	if "label" in _type:
		return [_type["label"]]
	else:
		return _type.get("labels", [])

def create(args):
	# Matlab must be loaded before any module which depends on random
	print("Importing Matlab ...", end="", flush=True)
	import matlab.engine
	print(" Done")

	import random
	from degradations import setup_matlab_degradations

	# Begin starting Matlab asynchronously
	m_eng = matlab.engine.start_matlab(background=True)

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.noise)
	speech, n_speechfiles = read_speech(args.speech)
	
	if n_noisefiles == 0:
		print("No noise files found", file=sys.stderr)
		sys.exit(1)
	if n_speechfiles == 0:
		print("No speech files found", file=sys.stderr)
		sys.exit(1)

	if not os.path.exists(args.specification):
		print("File not found:", args.specification, file=sys.stderr)
		sys.exit(1)
	
	with open(args.specification, "r") as f:
		spec = json.load(f)

	# Verify partition
	ds_train = spec["train"]
	ds_test = spec["test"]
	if 0 <= ds_train <= 1:
		ds_train = round(ds_train * n_speechfiles)
	else:
		ds_train = int(ds_train)
	if 0 <= ds_test <= 1:
		ds_test = round(ds_test * n_speechfiles)
	else:
		ds_test = int(ds_test)
	if ds_train < 0 or ds_test < 0 or ds_train + ds_test > n_speechfiles:
		print(f"Invalid partition of files: {ds_train} train, {ds_test} test, {ds_train + ds_test} total", file=sys.stderr)
		sys.exit(1)
	print(f"Speech: {ds_train} training files and {ds_test} testing files")
	
	# Partition speech set
	speech = [os.path.join(dirpath, filepath) for dirpath, filepaths in speech for filepath in filepaths]
	random.shuffle(speech)
	speech_train = speech[:ds_train]
	speech_test = speech[ds_train:ds_train + ds_test]
	del speech

	# Partition noise set
	noise_train = dict()
	noise_test = dict()
	print("Noise:")
	for noise_type, tup in noise.items():
		folder, files = tup
		random.shuffle(files)
		split = round(len(files) * ds_train / (ds_train + ds_test))
		noise_train[noise_type] = (folder, files[:split])
		noise_test[noise_type] = (folder, files[split:])
		print(f"\t{noise_type}: {split} training files and {len(files) - split} testing files")
	del noise

	# Verify degradation type labels
	for i, _type in enumerate(spec["types"]):
		for label in _labels(_type):
			if label not in spec["labels"]:
				print(f"Warning: Label '{label}' on type {i} not among the dataset's labels ({', '.join(spec['labels'])})", file=sys.stderr)
	for label in spec["labels"]:
		if isinstance(label, str):
			has_double = sum(label.lower() == l.lower() for l in spec["labels"] if isinstance(l, str)) > 1
		else:
			has_double = sum(label == l for l in spec["labels"]) > 1
		if has_double:
			print(f"Label {label} has a{' case-insensitive' if isinstance(label, str) else ''} double")
			sys.exit(1)


	# Assign degradation types indexes to speech files
	# TODO: testa
	import numpy as np
	n_types = len(spec["types"])
	weights = [t.get("weight", 1) for t in spec["types"]]
	weight_sum = sum(weights)

	types_train = [i for i, t in enumerate(spec["types"]) for _ in range(int(t["weight"] * len(speech_train) // weight_sum))]
	types_train = types_train + random.choices(range(len(spec["types"])), weights, k=len(speech_train) - len(types_train))
	print(np.bincount(types_train) / np.bincount(types_train)[0])
	random.shuffle(types_train)

	types_test = [i for i, t in enumerate(spec["types"]) for _ in range(int(t["weight"] * len(speech_test) // weight_sum))]
	types_test = types_test + random.choices(range(len(spec["types"])), weights, k=len(speech_test) - len(types_test))
	print(np.bincount(types_test) / np.bincount(types_test)[0])
	random.shuffle(types_test)

	# Setup Matlab
	print("Setting up Matlab ...", end="", flush=True)
	m_eng = m_eng.result()
	m_eng.cd(os.path.join(PROJECT_ROOT, "degradation"), nargout=0)
	args.adt = os.path.join(args.adt, "AudioDegradationToolbox")
	if not os.path.exists(args.adt):
		print("\nAudio Degradation Toolbox folder does not exist:", args.adt, file=sys.stderr)
		sys.exit(1)
	print(" Done")

	# Create output directory
	dataset_folder = os.path.join(args.datasets, spec["name"])
	output_train = os.path.join(dataset_folder, "train")
	output_test = os.path.join(dataset_folder, "test")
	if os.path.exists(dataset_folder) and sum(len(fns) for _, _, fns in os.walk(dataset_folder, followlinks=True)) > 1:
		if args.overwrite:
			print("Dataset", spec["name"], "already exists. Removing existing files.")
			c = 0
			for folder in (output_train, output_test):
				if not os.path.exists(folder):
					continue
				for filename in next(os.walk(folder))[2]:
					os.remove(os.path.join(folder, filename))
					c += 1
			print(f"Removed {c} file{'s' if c != 1 else ''}")
		else:
			print("Dataset", spec["name"], "already exists", file=sys.stderr)
			sys.exit(1)
	else:
		for folder in (dataset_folder, output_train, output_test):
			try:
				os.mkdir(folder)
				print("Created", folder)
			except FileExistsError:
				print(folder, "already exists")

	# Create datasets
	for t, speech_t, types_t, noise_t, output_t in [
		["train", speech_train, types_train, noise_train, output_train],
		["test", speech_test, types_test, noise_test, output_test]
	]:
		print(f"Creating {t}ing data")

		print("\tSetting up Matlab arguments")
		degradations = [spec["types"][_type]["degradations"] for _type in types_t]

		# Load degradation instructions into Matlab memory
		# The overhead for passing audio data between Python and Matlab is extremely high,
		# so Matlab is only sent the filenames and left to figure out the rest for itself
		# Further, certain datatypes (struct arrays) cannot be sent to/from Python
		setup_matlab_degradations(speech_t, degradations, noise_t, m_eng, "degradations")

		# Store all variables in Matlab memory
		output_files = list(_filenames(output_t, len(speech_t), ".wav"))
		m_eng.workspace["speech_files"] = speech_t
		m_eng.eval("speech_files = string(speech_files);", nargout=0)
		m_eng.workspace["output_files"] = output_files
		m_eng.eval("output_files = string(output_files);", nargout=0)
		m_eng.workspace["use_cache"] = True
		m_eng.workspace["adt_root"] = args.adt

		print("\tCreating samples")
		try:
			# Actual function call to create_samples.m, which in turn uses the ADT
			m_eng.eval("create_samples(speech_files, degradations, output_files, use_cache, adt_root);", nargout=0)
			# Save degradation instructions so that a sample can be recreated if needed
			m_eng.save(os.path.join(output_t, "degradations.mat"), "speech_files", "degradations", "output_files", nargout=0)
			
		except matlab.engine.MatlabExecutionError as e:
			print(e, file=sys.stderr)
			print("A Matlab error occurred", file=sys.stderr)
			print("Launching Matlab desktop so you may debug. Press enter to exit.", end="", flush=True, file=sys.stderr)
			m_eng.desktop(nargout=0)
			input()
			raise e
		
		# Save the labels.csv file
		print("\tSaving labels")
		label_file = os.path.join(output_t, "labels.csv")
		with open(label_file, "w", newline='') as f:
			writer = csv.writer(f)
			writer.writerow(("Filename",)
					+ tuple(str(l).lower() for l in spec["labels"]))
			for filename, typeind in zip(output_files, types_t):
				labels = _labels(spec["types"][typeind])
				writer.writerow((os.path.basename(filename),)
						+ tuple(int(l in labels) for l in spec["labels"]))

		print("Done")
	
	with open(os.path.join(dataset_folder, "source.json"), "w") as f:
		json.dump(spec, f)
	print("Wrote source.json")
		
	m_eng.exit()
	print("Dataset created")

def list_files(args):
	# This currently imports Matlab (in degradations) and doesn't need to
	from degradations import DEGRADATIONS

	if not os.path.exists(args.noise):
		print(f"Noise folder {args.noise} does not exist", file=sys.stderr)
		return

	if not os.path.exists(args.speech):
		print(f"Speech folder {args.speech} does not exist", file=sys.stderr)
		return

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.noise)
	speech, n_speechfiles = read_speech(args.speech)
	
	print(f"Found {n_noisefiles} noise files")
	for noisetype in noise:
		print(f"\t{len(noise[noisetype][1])} of type \"{noisetype}\"")
	print(f"Found {n_speechfiles} speech files")
	for t in ("test", "train"):
		print(f"\t{sum(len(fs) for d, fs in speech if t in d.lower())} in set \"{t}\"")
	print(f"Noise types: {', '.join(s for s in DEGRADATIONS if s != 'pad')}")

def prepare(args):
	from preparations import prep_folder

	if not os.path.exists(args.noise):
		print(f"Noise folder {args.noise} does not exist", file=sys.stderr)
		return

	if not os.path.exists(args.speech):
		print(f"Speech folder {args.speech} does not exist", file=sys.stderr)
		return

	tot = prep_folder(
		args.noise,
		recursive=True,
		prompt=args.prompt,
		skip_if_fixed=not args.no_skip,
		to_mono=not args.keep_stereo,
		downsample=not args.no_downsample
	)
	print(f"Prepared {tot} noise files")
	tot = prep_folder(
		args.speech,
		recursive=True,
		prompt=args.prompt,
		skip_if_fixed=not args.no_skip,
		to_mono=not args.keep_stereo,
		downsample=not args.no_downsample
	)
	print(f"Prepared {tot} speech files")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="create_dataset.py",
		description="Introduce problems into speech recordings"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--profile", help="Profile the dataset creator", action="store_true")

	group = parser.add_argument_group("input folders")
	group.add_argument("--speech", help="The folder containing speech recordings. This is explored recursively. Default: %(default)s",
			default=os.path.join(PROJECT_ROOT, "timit"), metavar="FOLDER")
	group.add_argument("--noise", help="The folder containing noise recordings. Noise files should be grouped by type in subfolders (e.g. noise/air-conditioning/, noise/electric-hum/). Default: %(default)s",
			default=os.path.join(PROJECT_ROOT, "noise"), metavar="FOLDER")

	subparser = subparsers.add_parser("create", help="Create a dataset from available speech and noise")
	subparser.set_defaults(func=create)

	subparser.add_argument("specification", help="JSON file with a dataset specification", metavar="myfile.json")
	# TODO: Ability to override these on the command line? (Why?)
	#subparser.add_argument("-n", "--name", help="Override name of the new dataset", metavar="MyDataset", default=None)
	#subparser.add_argument("--train", help="Override ratio/number of files in training set", default=None, type=float)
	#subparser.add_argument("--test", help="Override ratio/number of files in testing set", default=None, type=float)
	# TODO: maybe add ability to filter which labels should be used in the dataset? (Label groups?)

	# TODO: implement?
	# Currently, noise is down- or upsampled to match speech
	# Supposedly already taken care of by 'prepare'
	# subparser.add_argument("--downsample-speech", help="Downsample speech signal if noise sample rate is lower", action="store_true")

	subparser.add_argument("-o", "--overwrite", help="If the dataset already exists, overwrite it. CAUTION: Existing files are deleted without prompt.", action="store_true")
	subparser.add_argument("-d", "--datasets", help="The folder containing all datasets (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "datasets"), metavar="FOLDER")
	subparser.add_argument("--adt", help="Path to Audio Degradation Toolbox root folder (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "degradation", "adt"), metavar="FOLDER")
	subparser.add_argument("--no-cache", help="Disable caching noise files. Increases runtime but decreases memory usage.", action="store_true")
	
	subparser = subparsers.add_parser("prepare", help="Prepare the audio files in the dataset (convert from nist to wav, stereo to mono etc.)")
	subparser.set_defaults(func=prepare)
	
	subparser.add_argument("-p", "--prompt", help="List commands and ask for confirmation before performing preparation.", action="store_true")
	subparser.add_argument("--no-skip", help="Process a file even if a processed .wav file with the same name already exists.", action="store_true")
	subparser.add_argument("--keep-stereo", help="Do not convert stereo files to mono (kaldi requires mono for feature extraction)", action="store_true")
	subparser.add_argument("--no-downsample", help="Do not downsample files to 16 kHz (kaldi and the WebRTC VAD require 16kHz audio)", action="store_true")

	subparser = subparsers.add_parser("list", help="List available files and degradations")
	subparser.set_defaults(func=list_files)

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("args.func(args)", sort="cumulative")
	else:
		start = time()
		args.func(args)
		print(f"Total time: {time() - start:.1f} s")