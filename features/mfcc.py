import sys
import subprocess

def mfcc_kaldi(filenames, **kwargs):
	from .utils import check_kaldi_root
	check_kaldi_root()

	import kaldi_io
	from tempfile import NamedTemporaryFile

	with NamedTemporaryFile(suffix=".ark") as ark:
		cmd = [
			'compute-mfcc-feats',
			*(f"--{key}={val}" for key, val in kwargs.items()),
			'scp:-',
			'ark:' + ark.name
		]
		#print(*cmd)
		result = subprocess.run(cmd,
			input="\n".join(fn + " " + fn
				for fn in filenames) + "\n",
			encoding=sys.getdefaultencoding(),
			stderr=subprocess.PIPE,
			check=False
		)
		if result.returncode:
			print(result.stderr)
			result.check_returncode()
		else:
			for line in result.stderr.splitlines(False):
				if not line.startswith("LOG") and line.strip() != " ".join(result.args).strip():
					print(line)
	
		return [mat for _, mat in kaldi_io.read_mat_ark(ark)]

def mfcc_librosa(filenames, **kwargs):
	from librosa.core import load
	from librosa.feature import mfcc

	res = []
	for fn in filenames:
		y, fs = load(fn, None, None)
		res.append(mfcc(
			y, fs, **kwargs
			# n_mfcc=13,
			# n_fft=round(0.025 * fs),
			# hop_length=round(0.010 * fs),
			# window="hamming",
			# n_mels=23,
			# fmin=20,
			# lifter=22
		).T.copy())
	return res