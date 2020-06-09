import os
from matlab.engine import MatlabEngine
from random import choice, randint, uniform

# Define degradations here
DEGRADATIONS = (
	"", "pad", "addSound", "applyImpulseResponse", "adaptiveEqualizer", "applyMfccMeanAdaption",
	"normalize", "applyMute", "applySoftClipping", "addNoise", "applyAliasing", "applyClipping",
	"applyClippingAlternative", "applyDelay", "applyDynamicRangeCompression", "applyHarmonicDistortion",
	"applyHighpassFilter", "applyLowpassFilter", "applySpeedup", "applyWowResampling"
)

def setup_matlab_degradations(speech_files: list, degradations_by_file: list,
		noise_files: dict, m_eng: MatlabEngine, workspace_var: str="degradations") -> str:
	"""Setup the structure in Matlab needed to perform the specified degradations
	Returns: the Matlab workspace variable in which the structure is stored"""

	if len(speech_files) != len(degradations_by_file):
		raise ValueError(f"Different number of speech files ({len(speech_files)}) and degradations ({len(degradations_by_file)})")

	matlab_degradations_by_file = []
	max_num_degradations = 0
	null_degradation = {"name": "", "params": ""}
	
	for degradations in degradations_by_file:
		matlab_degradations_by_file.append([])
		
		for degradation in degradations:
			name = degradation["name"]
			params = dict()
			
			if degradation["name"] not in DEGRADATIONS:
				raise ValueError(f'Unrecognized degradation "{degradation}".')
			
			for key, value in degradation.items():
				if key == "name": continue
				if isinstance(value, dict):
					if tuple(value.keys()) == ("randomNoiseFile",):
						folder, files = noise_files[value["randomNoiseFile"]]
						params[key] = os.path.join(folder, choice(files))
					elif tuple(value.keys()) == ("randomChoice",):
						params[key] = choice(value["randomChoice"])
					elif tuple(value.keys()) == ("randomRange",):
						low, high = value["randomRange"]
						params[key] = uniform(low, high)
					elif tuple(value.keys()) == ("randomRangeInt",):
						low, high = value["randomRangeInt"]
						params[key] = randint(low, high)
					else:
						raise ValueError("Unknown replacement " + str(value))
				else:
					params[key] = float(value) if isinstance(value, int) else value
			
			if len(params) == 0:
				params = ""

			matlab_degradations_by_file[-1].append({"name": name, "params": params})

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