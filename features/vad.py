import webrtcvad
import numpy as np
import os
import warnings
from pydub import AudioSegment
from tempfile import TemporaryDirectory

vad = webrtcvad.Vad()

# TODO: add an option to convert the file to conform to the requirements as good as possible,
# and then perform the VAD on that file
def analyze_gen(filenames: list, frame_length: int, aggressiveness: int):
	if frame_length not in (10, 20, 30):
		raise ValueError("Only frame lengths of 10, 20 or 30 ms are allowed")
	if aggressiveness not in (0, 1, 2, 3):
		raise ValueError("VAD mode must be between 0 and 3")

	vad.set_mode(aggressiveness)

	for fn in filenames:
		audio = AudioSegment.from_wav(fn)

		if audio.channels != 1:
			raise ValueError(f"Only mono audio is supported, but file {fn} has {audio.channels} channels")
		if audio.sample_width != 2:
			raise ValueError(f"Audio must be 16 bit, but file {fn} is {audio.sample_width * 8}-bit")
		if audio.frame_rate not in (8000, 16000, 32000, 48000):
			raise ValueError(f"Audio must be sampled at 8, 16, 32 or 48 kHz, but file {fn} is sampled at {audio.frame_rate} Hz.")
		n_bytes_per_frame = audio.sample_width * int(audio.frame_rate * frame_length / 1000)
		data = audio.raw_data
		n_frames = len(data) // n_bytes_per_frame
		activity = np.full(n_frames, False)

		for ptr in range(n_frames):
			frame = data[ptr*n_bytes_per_frame : (ptr+1)*n_bytes_per_frame]
			activity[ptr] = vad.is_speech(frame, audio.frame_rate)

		yield (audio, activity)

def smooth(activity: np.ndarray, n: int, threshold: float=None):
	conv = np.convolve(activity, np.ones(n) / n, mode="same")

	if threshold is None or threshold == 0.5:
		return conv >= 0.5
	else:
		pos, = (conv >= threshold).nonzero()
		neg, = (conv <= 1 - threshold).nonzero()
		res = np.full(activity.shape, False)
		ptr = 0

		while ptr < len(activity):
			starts = pos[pos > ptr]
			if len(starts) == 0:
				return res
			ends = neg[neg > starts[0]]
			if len(ends) == 0:
				ends = [len(activity)]
			res[starts[0]:ends[0]] = True
			ptr = ends[0]

		return res

def analyze_all(filenames: list, frame_length: int, aggressiveness: int):
	return [(audio, activity) for audio, activity in analyze_gen(filenames, frame_length, aggressiveness)]

def splitaudio(audio: AudioSegment, frame_length: int, activity: np.ndarray, inverse: bool=False, min_length: float=0):
	if audio.channels != 1:
		raise ValueError(f"Only mono audio is supported, but supplied audio has {audio.channels} channels")
	
	n_bytes_per_frame = audio.sample_width * int(audio.frame_rate * frame_length / 1000)
	data = audio.raw_data
	n_frames = len(data) // n_bytes_per_frame
	segments = []

	start = None
	for ptr in range(n_frames):
		if activity[ptr] ^ inverse:
			if start is None:
				start = ptr * n_bytes_per_frame
		else:
			if start is not None:
				segment = AudioSegment(
					data[start : ptr*n_bytes_per_frame],
					sample_width=audio.sample_width,
					frame_rate=audio.frame_rate,
					channels=audio.channels
				)
				segments.append(segment)
				start = None

		if ptr == n_frames - 1 and start is not None:
			segment = AudioSegment(
				data[start : n_frames * n_bytes_per_frame],
				sample_width=audio.sample_width,
				frame_rate=audio.frame_rate,
				channels=audio.channels
			)
			segments.append(segment)
	
	if all(len(segment) < min_length for segment in segments):
		if len(segments) == 0:
			warnings.warn("VAD filters out all audio for at least one file. Try changing the VAD or smoothing parameters.")
		else:
			warnings.warn("No VAD filtered segments were long enough for at least one file. Try changing the VAD or smoothing parameters, or decreasing min_length.")

	if min_length > 0:
		segments = [segment for segment in segments if len(segment) >= min_length]

	return segments

def write_tempfiles(segments: list, directory: TemporaryDirectory=None):
	if directory is None:
		directory = TemporaryDirectory()
	dname = directory.name
	fname = 0

	tmp = None
	res = []
	for segment in segments:
		while tmp is None or os.path.exists(tmp):
			fname += 1
			tmp = os.path.join(dname, str(fname) + ".wav")

		segment.export(tmp, format="wav")
		res.append(tmp)
	
	return res, directory

def clean_tempdir(directory: TemporaryDirectory):
	directory.cleanup()