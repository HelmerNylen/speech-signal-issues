#!/usr/bin/env python3
import sys, os
import json
import subprocess
import time

from noise_classes import interface
from features.feature_extraction import read_dataset
from classifier.confusion_table import ConfusionTable

from functools import reduce
from argparse import Namespace
from contextlib import redirect_stdout

root = interface.PROJECT_ROOT
# The name of the dataset (suffixed by 2, 3, etc. if there are multiple realizations)
DATASET_NAME = "BasicDataset"
ds_json = os.path.join(root, "degradation", "basic_dataset.json")
# The number of realizations of the dataset, i.e. copies with the same settings but different random initializations
n_realizations = 2
# The number of trials for each realization
n_trials = 3

ds_names = [DATASET_NAME + (str(i+1) if i > 0 else "") for i in range(n_realizations)]

def assert_datasets_exist():
	for ds_name in ds_names:
		res = json.loads(subprocess.run(
			(os.path.join(root, "noise_classes", "interface.py"), "check", ds_name, "--silent"),
			encoding=sys.getdefaultencoding(),
			stdout=subprocess.PIPE,
			check=True
		).stdout)
		
		cmd = (os.path.join(root, "degradation", "create_dataset.py"), "create", ds_json, "--name", ds_name)
		if res[0] is None:
			# Dataset does not exist, create it
			subprocess.run(cmd, check=True)
		elif res[0] is False:
			# Dataset exists but is outdated, overwrite it
			subprocess.run((*cmd, "--overwrite"), check=True)
		else:
			print(ds_name, "exists and is up-to-date")

def train_test(ds_name=DATASET_NAME, silent=False, recompute=False):
	kwargs = {"stdout": subprocess.DEVNULL} if silent else dict()
	try:
		subprocess.run((os.path.join(root, "noise_classes", "interface.py"), "train", ds_name), check=True, **kwargs)
	except subprocess.CalledProcessError:
		# We may have been unlucky and ran out of VRAM so try again
		subprocess.run((os.path.join(root, "noise_classes", "interface.py"), "train", ds_name), check=True, **kwargs)

	args = Namespace(
		models=os.path.join(interface.PROJECT_ROOT, "classifier", "models"),
		dataset_name=ds_name,
		recompute=recompute,
		silent=silent)
	noise_classes = interface.load_noise_classes(args)
	args.filenames, classes, true_labels = read_dataset(ds_name, "test")
	predicted_labels, nc_ids = interface.get_labels(args, noise_classes)
	confusion_tables, stats_dict = interface.stats(predicted_labels, nc_ids, true_labels, classes)

	return confusion_tables, stats_dict

def repeat_trial(N, ds_name=DATASET_NAME, silent=True, recompute=False):
	confusion_tables_dicts, stats_dicts = zip(*(train_test(ds_name, silent=silent, recompute=recompute) for _ in range(N)))
	avg_cts, avg_stats = average(confusion_tables_dicts, stats_dicts)
	return avg_cts, avg_stats

def average(confusion_tables_dicts, stats_dicts):
	avg_stats = dict(
		(key, sum(d[key] for d in stats_dicts) / len(stats_dicts))
			for key in stats_dicts[0]
	)
	avg_cts = dict(
		(nc_id, reduce(merge_confusion_tables, map(lambda d: d[nc_id], confusion_tables_dicts)))
			for nc_id in confusion_tables_dicts[0]
	)
	return avg_cts, avg_stats

def merge_confusion_tables(a, b):
	assert (a.true_labels == b.true_labels).all() and (a.confused_labels == b.confused_labels).all()
	ct = ConfusionTable(a.true_labels, a.confused_labels)
	ct.data = a.data + b.data
	ct.totals = a.totals + b.totals
	return ct

if __name__ == "__main__":
	assert_datasets_exist()

	testresults = os.path.join(root, "testresults")
	if not os.path.exists(testresults):
		os.mkdir(testresults)

	with open(os.path.join(testresults, "log.txt"), "w") as f:
		with redirect_stdout(f):
			start = time.time()

			realizations_avg_cts, realizations_avg_stats = [], []
			for ds_name in ds_names:
				trials_avg_cts, trials_avg_stats = repeat_trial(n_trials, silent=False)
				realizations_avg_cts.append(trials_avg_cts)
				realizations_avg_stats.append(trials_avg_stats)

			realizations_avg_cts, realizations_avg_stats = average(realizations_avg_cts, realizations_avg_stats)
			print(f"Done after: {time.time() - start:.1f} s")

	with open(os.path.join(testresults, time.strftime("Test %Y-%m-%d %H:%M:%S", time.localtime())), "w") as f:
		print("Writing to", f.name)
		
		print("Datasets:", *ds_names, file=f)
		print("n_trials:", n_trials, file=f)

		print(" " * 15, "TPR", "TNR", sep="\t", file=f)
		for ct in realizations_avg_cts.values():
			print(
				f"{ct.true_labels[0]:<15}",
				f"{ct[0, 0] / ct[0, ...]:.2%}",
				f"{ct[1, 1] / ct[1, ...]:.2%}",
				sep="\t",
				file=f
			)

		for ct in realizations_avg_cts.values():
			print(ct, file=f)
			print(file=f)
			print(ct)
			print()

		for key, val in realizations_avg_stats.items():
			print(f"{key + ':  ':<18}{val:.3}", file=f)
			print(f"{key + ':  ':<18}{val:.3}")

		print(file=f)
		print("Noise class definitions:", file=f)
		with open(os.path.join(root, "noise_classes", "noise_classes.json")) as f_nc:
			for line in f_nc:
				print(line, file=f, end="")

		print(file=f)
		print(file=f)
		print(f"Dataset source ({DATASET_NAME}):", file=f)
		with open(os.path.join(root, "datasets", DATASET_NAME, "source.json")) as f_nc:
			for line in f_nc:
				print(line, file=f, end="")