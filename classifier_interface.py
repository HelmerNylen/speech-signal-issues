#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
import pickle
from contextlib import redirect_stdout
from time import time

from classifier.classifier import Classifier

from features import read_dataset, extract_features

def train(args):
	from classifier.model_gmmhmm import GMMHMM
	from classifier.model_lstm import LSTM
	from classifier.model_gmm import GMM
	supported_classifiers = [LSTM, GMMHMM, GMM]

	# Defaults for data and models folders
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)
	
	# Determine how the user wants their classifiers saved
	if len(args.write) == 1:
		args.write = args.write[0]
	if isinstance(args.write, str):
		if "<type>" not in args.write.lower() and len(args.classifiers) != 1 \
					and (args.read is not None and len(args.read) > 1 and "type" not in args.read[0].lower()):
			print("Invalid write specifier")
			sys.exit(1)
	elif len(args.write) != 0 and len(args.write) != (len(args.classifiers) or len(supported_classifiers)):
		print(f"Invalid number of write files ({len(args.write)}, expected 0 or {len(args.classifiers) or len(supported_classifiers)})")
		sys.exit(1)

	if args.config_stdin:
		config = json.loads(sys.stdin.buffer.read())

	elif args.read is not None:
		# Read classifiers from disk
		config = dict()
		if len(args.read) == 0:
			args.read = args.write
		if len(args.read) == 1:
			args.read = args.read[0]
		if isinstance(args.read, str):
			if "<type>" in args.read.lower():
				args.read = [re.sub("<type>", c.__name__, args.read, flags=re.IGNORECASE)
					for c in supported_classifiers
						if len(args.classifiers) == 0 or c.__name__.lower() in args.classifiers]
			else:
				args.read = [args.read]
		classifiers = []
		for fn in args.read:
			classifiers.append(Classifier.from_file(os.path.join(args.models, fn)))
			if classifiers[-1].model_type.__name__ in config:
				raise ValueError(f"Can only train one instance of each type of classifier at once (duplicate {classifiers[-1].model_type.__name__})")
			if len(args.classifiers) != 0 and classifiers[-1].__name__.lower() not in args.classifiers:
				raise ValueError(f"The classifier in {fn} is a {classifiers[-1].model_type.__name__}, which is disallowed by --classifiers ({args.classifiers})")
			
			config[classifiers[-1].model_type.__name__] = classifiers[-1].config

	else:
		# Create new classifiers using the provided config
		with open(args.config, "r") as f:
			config = json.load(f)
	
	args.silent = args.silent or args.write_stdout
	
	if args.override:
		# Override any specified config values
		# This also updates the classifiers' configs which have been read from disk
		for path, value in args.override:
			d = config
			path = path.split(".")
			for i, part in enumerate(path[:-1]):
				possible = [k for k in d.keys() if k.lower() == part]
				if len(possible) == 0:
					if not args.silent:
						print("Creating config key", path[:i+1])
					d[part] = dict()
					possible = [part]
				d = d[possible[0]]
			d[path[-1]] = json.loads(value)

	# Get features
	with redirect_stdout(sys.stderr if args.silent else sys.stdout):
		filenames, classes, labels = read_dataset(args.dataset_name, "train")
		feats, index = extract_features(filenames, "mfcc_kaldi", {}, cache=not args.recompute)

	# Create classifiers (unless previously read from disk)
	if args.read is None:
		classifiers = []
		for supported_classifier in supported_classifiers:
			if len(args.classifiers) == 0 or supported_classifier.__name__.lower() in args.classifiers:
				classifiers.append(Classifier(classes, supported_classifier, config, silent=args.silent))

	# Train the classifiers for the specified number of epochs
	for epoch in range(1, args.epochs + 1):
		for i, classifier in enumerate(classifiers):
			# Train
			classifier.train(feats, index, labels, silent=args.silent, models_folder=args.models)
			# Save classifier
			if isinstance(args.write, str):
				classifier.save_to_file(os.path.join(
					args.models,
					re.sub(
						"<type>",
						classifier.model_type.__name__,
						args.write,
						flags=re.IGNORECASE
					)
				))
			elif len(args.write) == len(classifiers):
				classifier.save_to_file(os.path.join(
					args.models,
					args.write[i]
				))
		if not args.silent and args.epochs != 1:
			print("Epoch", epoch)

	if args.write_stdout:
		# Send trained classifiers on stdout
		sys.stdout.buffer.write(pickle.dumps(classifiers))

	if not args.silent:
		print("Done")


def test(args):
	from classifier.model_gmmhmm import GMMHMM
	from classifier.model_lstm import LSTM
	from classifier.model_gmm import GMM
	supported_classifiers = [LSTM, GMMHMM, GMM]

	# Defaults for data and models folders
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)
	
	args.silent = args.silent or args.write_stdout

	# Read classifiers, either from disk or stdin
	if not args.silent:
		print("Reading classifiers ...", end="", flush=True)
	if args.read_stdin:
		classifiers = [c for c in Classifier.from_bytes(sys.stdin.buffer.read())]
	else:
		classifiers = []
	for f in args.read:
		if "<type>" in f.lower():
			f = [re.sub("<type>", c.__name__, f, flags=re.IGNORECASE) for c in supported_classifiers]
		else:
			f = [f]
		for fn in f:
			classifiers.append(Classifier.from_file(os.path.join(args.models, fn)))
	if not args.silent:
		print(" Done")

	if len(classifiers) == 0:
		print("Found no classifiers")
		sys.exit(1)
	if not args.silent:
		print(f"Loaded {len(classifiers)} classifier{'' if len(classifiers) == 1 else 's'}.")
	
	# Get features
	with redirect_stdout(sys.stderr if args.silent else sys.stdout):
		filenames, _, labels = read_dataset(args.dataset_name, "test")
		feats, index = extract_features(filenames, "mfcc_kaldi", {}, cache=not args.recompute)

	if not args.silent:
		print("Calculating scores ...", flush=True)
	# Score all classifiers
	confusion_tables = []
	for classifier in classifiers:
		if not args.silent:
			print(classifier.model_type.__name__)
		confusion_tables.append(classifier.test(feats, labels))
	if not args.silent:
		print("Done")

		# Print confusion tables to console
		for classifier, confusion_table in zip(classifiers, confusion_tables):
			print("Confusion table for", classifier.model_type.__name__)
			print(confusion_table)
			print()
	
	if args.write_stdout:
		# Send confusion tables on stdout
		sys.stdout.buffer.write(pickle.dumps(confusion_tables))

def _list(args):
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)

	# Get information about all classifiers in the models folder
	classifiers = []
	for f in Classifier.find_classifiers(args.models):
		c = Classifier.from_file(os.path.join(args.models, f))
		classifiers.append([
			os.path.basename(f),
			c.model_type.__name__,
			str(os.path.getsize(f))
		])
	
	if len(classifiers) > 0:
		classifiers = [["Filename", "Model Type", "Size in Bytes"]] + classifiers
		widths = [max(len(classifiers[row][i]) for row in range(len(classifiers))) for i in range(3)]
		print("  ".join(classifiers[0][i].ljust(widths[i]) for i in range(3)))
		for row in classifiers[1:]:
			print("  ".join(row[i].rjust(widths[i]) for i in range(3)))
	else:
		print(f"No classifiers in {args.models}")

if __name__ == "__main__":
	supported_types = ["LSTM", "GMMHMM", "GMM"]
	parser = argparse.ArgumentParser(
		prog="classifier.py",
		description="Train or test a classifier. Supported types: " + ", ".join(supported_types)
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--profile", help="Profile the script", action="store_true")
	parser.add_argument("--models", help="Path to classifier models save folder (default: $PWD/classifier/models)", default=os.path.join(os.getcwd(), "classifier", "models"))

	subparser = subparsers.add_parser("train", help="Perform training")
	subparser.set_defaults(func=train)

	subparser.add_argument("dataset_name", help="Name of the dataset to train on", metavar="MyDataset")

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")
	subparser.add_argument("-c", "--classifiers", help=f"Classes to train. If none are specified, all types are trained. Available: {', '.join(supported_types)}.", metavar="TYPE",
							nargs="*", choices=list(map(str.lower, supported_types))+[[]], type=str.lower, default=[])
	subparser.add_argument("-o", "--override", help="Override a classifier parameter. PATH takes the form 'classifier.category.parameter'. VALUE is a JSON-parsable object.", nargs=2, metavar=("PATH", "VALUE"), action="append")
	subparser.add_argument("-s", "--silent", help="Suppress all informational output on stdout (certain output is instead routed to stderr)", action="store_true")
	subparser.add_argument("-e", "--epochs", help="Number of epochs to run. Repeatedly trains each classifier <TYPE>.train.n_iter times and may save the intermediate classifiers (see --write). No effect on SVM training. Default: %(default)d", metavar="NUM_EPOCHS", type=int, default=1)
	
	out = subparser.add_argument_group("Classifier output")
	out.add_argument("-w", "--write", help="Files relative to <MODELS> to save classifier(s) to. Must be 0, 1 (containing <TYPE>), or same as number of classes. Default: %(default)s", metavar="FILE", nargs="*", default="latest_<TYPE>.classifier")
	out.add_argument("--write-stdout", help="Write resulting classifiers to stdout. Implies --silent.", action="store_true")

	configs = subparser.add_mutually_exclusive_group()
	configs.add_argument("--config", help="Path to classifier config file (default: $PWD/classifier/defaults.json)", default=os.path.join(os.getcwd(), "classifier", "defaults.json"))
	configs.add_argument("--read", help="Continue training existing classifiers for another epoch. Note that learning rates and iteration counts can be overridden by -o.\n"
		+ "Specify files relative to <MODELS> to read classifier(s) from. Must be 0, 1 (containing <TYPE>), or same as number of classes. If 0 files are provided the value of --write is used.", metavar="FILE", nargs="*", default=None)
	configs.add_argument("--config-stdin", help="Read config from stdin", action="store_true")

	subparser = subparsers.add_parser("test", help="Perform testing")
	subparser.set_defaults(func=test)

	subparser.add_argument("dataset_name", help="Name of the dataset to train on", metavar="MyDataset")

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")
	subparser.add_argument("-s", "--silent", help="Suppress all informational output on stdout (certain output is instead routed to stderr)", action="store_true")
	subparser.add_argument("--write-stdout", help="Write resulting confusion tables to stdout. Implies --silent.", action="store_true")

	inp = subparser.add_argument_group("Classifier input")
	inp.add_argument("read", help="Files relative to <MODELS> to read classifier(s) from. '<TYPE>' can be used in FILE to match all classifier type names. Default: %(default)s", metavar="FILE", nargs="*", default=["latest_<TYPE>.classifier"])
	inp.add_argument("--read-stdin", help="Read classifiers from stdin", action="store_true")

	subparser = subparsers.add_parser("list", help="List all classifiers in <MODELS>")
	subparser.set_defaults(func=_list)

	subparser = subparsers.add_parser("available-types", help="List the available classifier types")
	subparser.set_defaults(func=lambda a: print(*supported_types, sep='\n'), silent=True)

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("args.func(args)", sort="cumulative")
	else:
		start = time()
		args.func(args)
		if not vars(args).get("silent", False) and not vars(args).get("write_stdout", False):
			print(f"Total time: {time() - start:.1f} s")