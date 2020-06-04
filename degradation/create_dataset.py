#!/usr/bin/env python3
import os
import sys
import argparse
import math
import csv
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

def create(args):
	# Matlab must be loaded before any module which depends on random
	print("Importing Matlab ...", end="", flush=True)
	import matlab.engine
	print(" Done")

	import random
	from degradations import get_degradations, setup_matlab_degradations

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

	# Parse and verify degradation classes
	available_classes = [d for d in get_degradations(noise) if d not in ("pad",)]
	if len(args.classes) == 0 or args.classes == ["all"]:
		args.classes = map(str, available_classes)
	tmp = []
	for c in args.classes:
		try:
			tmp.append(available_classes[available_classes.index(None if c.lower() == "none" else c.lower())])
		except ValueError:
			print(f"Unknown degradation class \"{c}\"", file=sys.stderr)
			print("Available:", ", ".join(available_classes), file=sys.stderr)
			print(file=sys.stderr)
			print("(Stopping Matlab ...", end="", flush=True)
			m_eng.cancel()
			print(" Done)")
			sys.exit(1)
	args.classes = tmp
	n_classes = len(args.classes)
	del tmp

	# Verify partition
	if 0 <= args.train <= 1:
		args.train = round(args.train * n_speechfiles)
	else:
		args.train = int(args.train)
	if 0 <= args.test <= 1:
		args.test = round(args.test * n_speechfiles)
	else:
		args.test = int(args.test)
	if args.train < 0 or args.test < 0 or args.train + args.test > n_speechfiles:
		print(f"Invalid partition of files: {args.train} train, {args.test} test, {args.train + args.test} total", file=sys.stderr)
		sys.exit(1)
	print(f"Speech: {args.train} training files and {args.test} testing files")
	
	# Partition speech set
	speech = [os.path.join(dirpath, filepath) for dirpath, filepaths in speech for filepath in filepaths]
	random.shuffle(speech)
	speech_train = speech[:args.train]
	speech_test = speech[args.train:args.train + args.test]
	del speech

	# Partition noise set
	noise_train = dict()
	noise_test = dict()
	print("Noise:")
	for noise_type, tup in noise.items():
		folder, files = tup
		random.shuffle(files)
		split = round(len(files) * args.train / (args.train + args.test))
		noise_train[noise_type] = (folder, files[:split])
		noise_test[noise_type] = (folder, files[split:])
		print(f"\t{noise_type}: {split} training files and {len(files) - split} testing files")
	del noise

	# Assign noise types to speech files
	labels_train = list(range(n_classes)) * (len(speech_train) // n_classes)
	labels_train = labels_train + ([n_classes - 1] * (len(speech_train) - len(labels_train)))
	random.shuffle(labels_train)
	labels_test = list(range(n_classes)) * (len(speech_test) // n_classes)
	labels_test = labels_test + ([n_classes - 1] * (len(speech_test) - len(labels_test)))
	random.shuffle(labels_test)

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
	dataset_folder = os.path.join(args.datasets, args.name)
	output_train = os.path.join(dataset_folder, "train")
	output_test = os.path.join(dataset_folder, "test")
	if os.path.exists(dataset_folder) and sum(len(fns) for _, _, fns in os.walk(dataset_folder, followlinks=True)) > 1:
		if args.overwrite:
			print("Dataset", args.name, "already exists. Removing existing files.")
			c = 0
			for folder in (output_train, output_test):
				if not os.path.exists(folder):
					continue
				for filename in next(os.walk(folder))[2]:
					os.remove(os.path.join(folder, filename))
					c += 1
			print(f"Removed {c} file{'s' if c != 1 else ''}")
		else:
			print("Dataset", args.name, "already exists", file=sys.stderr)
			sys.exit(1)
	else:
		for folder in (dataset_folder, output_train, output_test):
			try:
				os.mkdir(folder)
				print("Created", folder)
			except FileExistsError:
				print(folder, "already exists")

	# Create datasets
	for t, speech_t, labels_t, noise_t, output_t in [
		["train", speech_train, labels_train, noise_train, output_train],
		["test", speech_test, labels_test, noise_test, output_test]
	]:
		print(f"Creating {t}ing data")

		print("\tSetting up Matlab arguments")
		degradations = [args.classes[label] for label in labels_t]
		if args.pad is not None:
			degradations = [["pad", degradation] for degradation in degradations]
		# Load degradation instructions into Matlab memory
		# The overhead for passing audio data between Python and Matlab is extremely high,
		# so Matlab is only sent the filenames and left to figure out the rest for itself
		# Further, certain datatypes (struct arrays) cannot be sent to/from Python
		setup_matlab_degradations(noise_t, speech_t, degradations, m_eng, vars(args), "degradations")

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
					+ tuple(str(c).lower() for i, c in enumerate(args.classes) if i in labels_t))
			for filename, label in zip(output_files, labels_t):
				writer.writerow((os.path.basename(filename),)
						+ tuple(int(label == i) for i in range(len(args.classes))))

		print("Done")
	
	with open(os.path.join(dataset_folder, "info.txt"), "w") as f:
		print("Arguments:", vars(args), file=f)
		
	m_eng.exit()
	print("Dataset created")

def list_files(args):
	# This currently imports Matlab (in degradations) and doesn't need to
	from degradations import get_degradations

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
	print(f"Noise types: {', '.join(s for s in map(str, get_degradations(noise)) if s != 'pad')}")

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

	subparser.add_argument("name", help="Name of the new dataset", metavar="MyDataset")
	subparser.add_argument("-o", "--overwrite", help="If the dataset already exists, overwrite it. CAUTION: Existing files are deleted without prompt.", action="store_true")
	subparser.add_argument("-d", "--datasets", help="The folder containing all datasets (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "datasets"), metavar="FOLDER")
	subparser.add_argument("--adt", help="Path to Audio Degradation Toolbox root folder (default: %(default)s)",
			default=os.path.join(PROJECT_ROOT, "degradation", "adt"), metavar="FOLDER")

	# TODO: definiera degradations via jsonfiler
	group = subparser.add_argument_group("degradation parameters")
	group.add_argument("--snr", help="Signal-to-noise ratio in dB (speech is considered signal)", type=float, default=None)
	# TODO: implement?
	# Currently, noise is down- or upsampled to match speech
	# group.add_argument("--downsample-speech", help="Downsample speech signal if noise sample rate is lower", action="store_true")
	group.add_argument("-p", "--pad", help="Pad the speech with PAD seconds of silence at the beginning and end", type=float, default=None)
	group.add_argument("--clip-amount", help="Percentage of samples which will be considered out of range", type=float, default=None)
	group.add_argument("--clip-type", help="Type of clipping to use, either 'soft' or 'hard'", default="hard", choices=("soft", "hard"), type=str.lower)
	group.add_argument("--mute-percent", help="Percentage of samples which will be muted", type=float, default=None)
	group.add_argument("--total-mute-length", help="Total time in seconds which will be muted", type=float, default=None)
	group.add_argument("--mute-length-min", help="Minimum length of a mute segment in seconds", type=float, default=None)
	group.add_argument("--mute-length-max", help="Maximum desired length of a mute segment in seconds", type=float, default=None)
	group.add_argument("--mute-pause-min", help="Minimum pause between mute segments in seconds", type=float, default=None)

	subparser.add_argument("-c", "--classes", help="The class types to use (default: all)", metavar="CLASS", nargs="+", default=[])
	subparser.add_argument("--train", help="Ratio/number of files in training set (default: %(default).2f)", default=11/15, type=float)
	subparser.add_argument("--test", help="Ratio/number of files in testing set (default: %(default).2f)", default=4/15, type=float)
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