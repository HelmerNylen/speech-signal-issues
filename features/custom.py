import numpy as np

def acf(filenames, *, require_max: bool=False, ratio: bool=True, threshold: float=0.1):
	"""The number of times the signal value does not change between samples
	
	Parameters
    ----------
    Filenames : list of strings
        path to the input files

    require_max : bool
        if true, abs(y[i]) == max(abs(y)) is also a requirement for an instant i to be counted

    ratio : bool
        divide the result with the total number of samples

    threshold : float
        unless require_max is set, abs(y[i]) > threshold * max(abs(y)) is also a requirement for an instant i to be counted
		
	Returns
    -------
    values    : list of np.ndarray [shape=(1,1)]
        the computed scalar feature for each filename, as a 1x1 matrix
	"""

	import librosa

	res = []
	for fn in filenames:
		y, _ = librosa.load(fn, None)
		y_abs = np.abs(y)[:-1]
		if require_max:
			val = ((np.diff(y) == 0) & (y_abs == y_abs.max())).sum()
		else:
			val = ((np.diff(y) == 0) & (y_abs > threshold * y_abs.max())).sum()
		if ratio:
			val = val / len(y)
		res.append(np.array([[val]]))

	return res

def histogram(filenames, *, ratio: bool=True, n_bins: int=20, relative_bins: bool=False):
	import librosa
	
	bins = np.linspace(-1, 1, n_bins)
	res = []
	for fn in filenames:
		y, _ = librosa.load(fn, None)
		if relative_bins:
			bins = np.linspace(y.min(), y.max(), n_bins)
		hist, _ = np.histogram(y, bins)
		if ratio:
			hist = hist / len(y)

		res.append(hist.reshape(1, -1).astype('float32'))

	return res

def rms_energy(filenames, *, frame_length: int=30, hop_length: int=None, delta: bool=False, normalized=False):
	import librosa
	
	if hop_length is None:
		hop_length = frame_length / 2
	
	res = []
	for fn in filenames:
		y, fs = librosa.load(fn, None)
		frame_length_used = int(fs * frame_length / 1000)
		hop_length_used = int(fs * hop_length / 1000)
		rms = librosa.feature.rms(y=y, frame_length=frame_length_used, hop_length=hop_length_used)\
			.reshape(-1, 1)
		if normalized:
			rms /= np.abs(rms).max() or 1
		if delta:
			diff = np.diff(rms, axis=0)
			diff = np.pad(diff, ((1, 0), (0, 0)))
			rms = np.column_stack((rms, diff))
		
		res.append(rms)

	return res

def rms_energy_infra(filenames, *, frame_length: int=500, hop_length: int=None, threshold: int=20):
	import librosa

	if hop_length is None:
		hop_length = frame_length / 4

	res = []
	for fn in filenames:
		y, fs = librosa.load(fn, None)
		frame_length_used = int(fs * frame_length / 1000)
		hop_length_used = int(fs * hop_length / 1000)
		S, _ = librosa.magphase(librosa.stft(y, n_fft=frame_length_used, hop_length=hop_length_used))
		freqs = librosa.fft_frequencies(sr=fs, n_fft=frame_length_used)
		S[freqs > threshold, :] = 0

		res.append(librosa.feature.rms(S=S, frame_length=frame_length_used, hop_length=hop_length_used)\
			.reshape(-1, 1).astype('float32'))
	
	return res