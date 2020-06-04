import numpy as np

def acf(filenames, **_):
	import librosa
	res = []
	for fn in filenames:
		y, _ = librosa.load(fn, None)
		autocorr = np.correlate(y, y, mode="same")
		res.append(autocorr[len(autocorr)//2 - 1])

	return res