import os
import re
from matlab.engine import MatlabEngine
from random import choice

additive_noise_prefix = "additive-"

# Define degradation names here
def get_degradations(noise_files: dict) -> list:
	res = [None, "pad", "white", "clipping", "mute"]
	res.extend(additive_noise_prefix + k for k in noise_files.keys())
	return res

def setup_matlab_degradations(noise_files: dict, speech_files: list, degradations_by_file: list,
		m_eng: MatlabEngine, degradation_parameters: dict, workspace_var: str="degs") -> str:
	"""Setup the structure in Matlab needed to perform the specified degradations
	Returns: the Matlab workspace variable in which the structure is stored"""

	if len(speech_files) != len(degradations_by_file):
		raise ValueError(f"Different number of speech files ({len(speech_files)}) and degradations ({len(degradations_by_file)})")

	allowed_degradations = get_degradations(noise_files)
	matlab_degradations_by_file = []
	max_num_degradations = 0
	null_degradation = {"name": "", "params": ""}

	normalize_audio = degradation_parameters.get("normalizeOutputAudio", True)
	
	for degradations in degradations_by_file:
		if isinstance(degradations, str) or not hasattr(degradations, '__iter__'):
			degradations = degradations,

		matlab_degradations_by_file.append([])
		has_normalized = False
		for degradation in degradations:
			# No degradation.
			# Normalize audio if this has not already been done.
			# All other degradations normalize audio except if specifically told not to.
			# TODO: make normalize audio, snr etc. non-global settings
			if degradation is None:
				if normalize_audio and not has_normalized:
					name = "normalize"
					params = ""
					has_normalized = True

					matlab_degradations_by_file[-1].append({"name": name, "params": params})
			
			# Add additive noise from a recording.
			elif degradation.startswith(additive_noise_prefix) and degradation in allowed_degradations:
				if degradation_parameters.get("snr", None) is None:
					raise ValueError("Please specify SNR to apply additive noise")
				
				name = "addSound"

				folder, files = noise_files[degradation[len(additive_noise_prefix):]]
				path = os.path.join(folder, choice(files))
				params = {
					"addSoundFile": path,
					"addSoundFileStart": "random",
					"snrRatio": degradation_parameters["snr"],
					"normalizeOutputAudio": normalize_audio
				}
				has_normalized = normalize_audio

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			# Pad a sample with silence.
			# Does not degrade the sample in the same sense, but still an operation we want to do as part of the same pipeline.
			elif degradation == "pad":
				if degradation_parameters.get("pad", None) is None:
					raise ValueError("Please specify padding amount to apply padding")

				name = "pad"
				params = {
					"before": degradation_parameters["pad"],
					"after": degradation_parameters["pad"]
				}

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			# Add additive white noise.
			elif degradation == "white":
				if degradation_parameters.get("snr", None) is None:
					raise ValueError("Please specify SNR to apply additive white noise")

				name = "addNoise"
				params = {
					"noiseColor": "white",
					"snrRatio": degradation_parameters["snr"],
					"normalizeOutputAudio": normalize_audio
				}
				has_normalized = normalize_audio

				matlab_degradations_by_file[-1].append({"name": name, "params": params})
			
			# Apply soft or hard clipping.
			elif degradation == "clipping":
				if degradation_parameters.get("clipAmount", None) is None:
					raise ValueError("Please specify clipAmount to apply clipping")

				name = {
					"hard": "applyClippingAlternative",
					"soft": "applySoftClipping"
				}.get(degradation_parameters.get("clip_type", "hard").lower(), None)
				if name is None:
					raise ValueError(f"Invalid clip_type: {degradation_parameters['clip_type']}. Must be one of 'soft', 'hard'.")
				
				params = {
					"percentOfSamples": degradation_parameters["clip_amount"]
				}
				# Soft clipping reduces the maximum signal value to 2/3
				has_normalized = True

				matlab_degradations_by_file[-1].append({"name": name, "params": params})
			
			# Apply random muting.
			elif degradation == "mute":
				if degradation_parameters.get("mute_percent", None) is None \
						and degradation_parameters.get("total_mute_length", None) is None:
					raise ValueError("Please specify mute_percent or total_mute_length to apply clipping")
				
				name = "applyMute"
				params = {re.sub("_.", lambda s: s[0][1].upper(), field): degradation_parameters[field]
					for field in ("mute_percent", "total_mute_length",
						"mute_length_min", "mute_length_max", "mute_pause_min")
							if degradation_parameters.get(field, None) is not None}

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			else:
				raise ValueError(f'Unrecognized degradation "{degradation}"')

		max_num_degradations = max(max_num_degradations, len(matlab_degradations_by_file[-1]))
	
	# Pad the degradation list with identity degradations so that it is a proper rectangular matrix
	for degradations in matlab_degradations_by_file:
		if len(degradations) < max_num_degradations:
			degradations.extend([null_degradation] * (max_num_degradations - len(degradations)))

	# Create a cell array of {name: "", params: {}} structs
	m_eng.workspace[workspace_var] = matlab_degradations_by_file
	# Cast it to a struct array
	m_eng.eval(workspace_var + " = cell2mat(cellfun(@cell2mat, " + workspace_var + ", 'UniformOutput', false)');", nargout=0)
	
	return workspace_var

if __name__ == "__main__":
	print("Please use create_dataset.py")