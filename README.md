# Detecting Signal Corruptions in Voice Recordings for Speech Therapy
Code for my degree project [_Detecting Signal Corruptions in Voice Recordings for Speech Therapy_](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-291429). Note that this is for the multi-label code, the single-label code can be found at [https://github.com/HelmerNylen/prestudy](https://github.com/HelmerNylen/prestudy).

To get started with the tool you do not need to read this entire document as much of the content is here for reference. [**Installation**](#installation) and the first section of [**Running Experiments**](#running-experiments) are recommended.

## Table of contents
1. [**Installation**](#installation)
2. [**Terminology**](#terminology)
3. [**Structure**](#structure)
	1. [_classifier_](#classifier)
	2. [_datasets_](#datasets)
	3. [_degradation_](#degradation)
	4. [_features_](#features)
	5. [_noise_](#noise)
	6. [_noise\_classes_](#noise_classes)
	7. [_timit_](#timit)

4. [**Running Experiments**](#running-experiments)
	1. [Generating Datasets](#generating-datasets)
	2. [Adding Corruptions](#adding-corruptions)
	3. [Adding Features](#adding-features)
		- [Voice Activity Detection](#voice-activity-detection)
	4. [Adding Classifiers](#adding-classifiers)
		- [Ensemble Classification](#ensemble-classification)

## Installation
Ubuntu 18.04.5 was used during development.
1. Clone or download this repository.
2. Install [Python 3.6](https://www.python.org/downloads/). Version 3.6.9 was used during development.
3. Install [NVidia CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
4. Install LLVM via `sudo apt-get install llvm`.
5. Install SoX and additional audio format handlers via `sudo apt-get install sox libsox-fmt-all`.
6. Install Matlab.
7. Acquire TIMIT and place the root directory (containing `DOC`, `TEST`, `TRAIN` and `README.DOC`) in the folder [_timit_](timit).
8. Install [Kaldi](https://github.com/kaldi-asr/kaldi).
9. Decide if you need the GenHMM:
	- If you want to use the GenHMM, clone or download the [gm\_hmm repository](https://github.com/FirstHandScientist/genhmm) and place its root directory in the folder _classifier/gm\_hmm_. Follow its installation instructions. The dataset preparation steps etc. are not needed but can be useful for verifying the installation.
	- If you are not interested in using the GenHMM, proceed to the next step. It can be installed at a later time if you change your mind.
10. Open a terminal in this project's root folder (where _README.md_ is located). Create a virtual environment with a Python 3.6 interpreter via `python3.6 -m venv pyenv`.
11. Activate the environment via `source pyenv/bin/activate`.
12. Go to `[your Matlab root]/extern/engines/python/` and run `python setup.py install`.
13. Return to this project's root folder and install the dependencies listed in [`requirements.txt`](requirements.txt), via e.g. `pip install -r requirements.txt`.
14. Download the [Audio Degradation Toolbox](https://code.soundsoftware.ac.uk/projects/audio-degradation-toolbox/files) (ADT). Place the content of _audio-degradation-toolbox_ in [_degradation/adt_](degradation/adt). The structure should be that _degradation/adt_ now contains three folders: _AudioDegradationToolbox_, _html\_gen_, and _testdata_, as well as a number of files.
15. Run `degradation/create_dataset.py prepare` to convert noise files (recordings for air-conditioning noise and electric hum are included) to the right formats, convert the TIMIT files from NIST sphere to standard wavefile etc.
16. Given that the above steps completed successfully, you should now be all set to run `./experiments.py`.

To run experiments in a new terminal window, or after restarting your computer, you need to activate the virtual environment using `source pyenv/bin/activate`.


## Terminology
The terminology used in the codebase is slightly different from that in the report.
- In general, a _corruption_ (report) is here referred to as a _noise type_ or _noise class_ (code). _Degradations_ in the code are generally the smaller operations which constitute a _corruption_, but can also be used as a third synonym as there is usually only one degradation per corruption.
- A _multiclass_ model in the code is essentially a _discriminative_ model in the report, as the models which are not _multiclass_ only model a single class in a _generative_ way.


## Structure
The root folder contains some project-wide settings such as the module requirements and linting options. The highest-level interface, `experiments.py`, is also located here and helps perform more complex tests. It can generate the required datasets and train multiple classifier instances and average their results. Results are saved to a folder called _testresults_.

### [_classifier_](classifier)
_classifier_ contains interfaces to or implementations of the various classifier algorithms. [`classifier.py`](classifier/classifier.py) is an interface for passing data to algorithms, saving/loading trained algorithms, and performing labeling. Using generative models (GMM-HMM or GenHMM), the classifier trains one model for each possible class (i.e. positive and negative). Using discriminative models such as the LSTM, only one model is trained and the Classifier class is mostly just a wrapper.

Each algorithm inherits from Model, defined in [`model.py`](classifier/model.py). The algorithms themselves are stored in `model_x.py` files. Algorithm implementations should override the `train()` and `score()` methods. Discriminative models should also override `get_noise_types()` and set the class property `MULTICLASS` to `True`. Note that for discriminative models the constructor also receives a `noise_types` parameter, which is a list mapping label index to corruption ids. This should be stored and returned verbatim by `get_noise_types()`.

Default hyperparameters for the algorithm are stored in [`defaults.json`](classifier/defaults.json). Each algorithm should define the `"parameters"` and `"train"` fields. These two fields are passed in a dictionary to the constructor and the `train()` method of a model via the `config` parameter. Optionally, the `"score"` field can be used to pass keyword arguments to the `score()` method.

[`confusion_table.py`](classifier/confusion_table.py) defines a confusion table on which precision and other evaluation metrics can be computed. They are used to pass test results from the Classifier class to higher-level interfaces but can also be printed to the terminal directly.

The [_models_](classifier/models) folder is the suggested and default place to store trained classifier algorithms. Classifiers are stored here with the `.classifier` file extension. The Noise Classes interface (see below) also stores its classifier bundles here, using the `.noiseclasses` extension.

The _gm\_hmm_ folder, if present, should contain a clone of the GenHMM repository.

### [_datasets_](datasets)
This is the suggested and default folder to store all datasets. Assuming you generate a dataset named _MyDataset_, a folder _datasets/MyDataset_ will be generated. This will contain the training set under _datasets/MyDataset/train_ and the testing set under _datasets/MyDataset/test_. These two folders each contain a large number of `.wav` files as well as `labels.csv` containing comma-separated labels for each file, and `degradations.mat` which is a complete list of the arguments sent to the Audio Degradation Toolbox. The latter can be useful for debugging if you have a sample and want to know what TIMIT file it originated from or what value was chosen for a random parameter.

Finally, the dataset definition is copied to _datasets/MyDataset/source.json_. More on how these work below.

### [_degradation_](degradation)
This contains the scripts needed to generate datasets. These are accessed via the [`create_dataset.py`](degradation/create_dataset.py) tool, the usage of which is explained under section [**Generating Datasets**](#generating-datasets).

[`balanced_dataset.json`](degradation/balanced_dataset.json) and [`basic_dataset.json`](degradation/basic_dataset.json) contain the definitions of the _Balanced_ and _Realistic_ datasets, respectively, used in the report. These are just examples and can be replaced with your own definition or removed.

[`create_samples.m`](degradation/create_samples.m) is a Matlab script that receives instructions form `create_dataset.py`, invokes the degradations in the ADT, and saves the resulting samples to disk.

[`degradations.py`](degradation/degradations.py) contains a list of all valid degradations. It helps convert the JSON definition of a dataset into proper arguments for `create_samples.m`.

[`preparations.py`](degradation/preparations.py) is an interface to SoX, which helps convert the audio files (TIMIT and noise files) to valid 16-bit 16 kHz mono wavefiles.

The [_adt_](degradation/adt) folder contains the Audio Degradation Toolbox, which is a Matlab tool with scripts for common degradations.

The [_custom-degradations_](degradations/custom-degradations) folder contains additional degradations. Custom degradations should follow the naming convention and argument signature of the built-in degradations to be detected by `create_samples.m`.

### [_features_](features)
The _features_ folder contains tools to extract, filter, and cache various types of features.

[`mfcc.py`](features/mfcc.py) contains hooks to external libraries for MFCC features. [`custom.py`](features/custom.py) contains definitions for other types of features.

[`feature_extraction.py`](features/feature_extraction.py) is responsible for loading datasets, resolving caching, and invoking the features defined in `mfcc.py` and `custom.py`. The extracted features are generally returned as a tuple `(feats, index)`. `feats` is a list of Numpy arrays containing feature values, usually with the time dimension along axis 0 (if applicable). `index` is a list mapping the arrays in `feats` to the corresponding file in the dataset. It is usually equal to `np.arange(len(feats))`, but if VAD filtering is used it may look more like `[0, 0, 1, 1, 1, 2, 3, 3, ...]`.

[`utils.py`](features/utils.py) contains helper functions for the other files.

[`vad.py`](features/vad.py) performs the voice activity detection filtering. If used it writes each detected segment to a temporary file, and `feature_extraction.py` extracts the features for each segment and organizes the index variable.

The [_cache_](degradation/cache) folder stores computed features so that they can be reused without having to recompute them. The size of the cache can be adjusted in `feature_extraction.py`.

_kaldi_ contains the Kaldi library, used for MFCC extraction.

### [_noise_](noise)
_noise_ is where the audio files for additive noise are stored. The name of each subfolder serves as an identifier, and during dataset generation each subfolder is partitioned into a training and a testing set. This does not, however, affect the contents in the _noise_ subfolders on disk.

Only `.wav` files are considered valid, so to include for example an MP3 file you need to run `degradation/create_dataset.py prepare`. This creates a `.wav` copy of each file which conforms to the requirements (16 bits, 16 kHz etc.), however if the original file is also a `.wav` file it is overwritten.

### [_noise\_classes_](noise_classes)
This folder contains the main tools for working with classifier and feature combinations, as well as definitions of different corruptions (herein referred to as noise classes).

[`noise_class.py`](noise_classes/noise_class.py) contains tools for classification using one or more classifiers for a specific corruption. A `NoiseClass` stores trained classifiers, feature and ensemble classification settings, and the degradations which consitute a corruption.

[`interface.py`](noise_classes/interface.py) is used for working with `NoiseClass` instances. It can be used to train and test classifiers on specific datasets, perform classification of external files using trained classifiers, and to check whether the `noise_classes.json` file, the trained classifiers in _classifier/models_, and the dataset definition in `datasets/MyDataset/source.json` are in sync, taking into account `classifier/defaults.json`.

[`noise_classes.json`](noise_classes/noise_classes.json) is where most of the editing goes when performing different experiments. It contains an array of corruption specifications, which in turn consist of an identifier, a readable name, the degradations needed to create it, and the classifier(s) and features that should be used to detect it. The identifiers in this file are used in the dataset definitions. See [**Adding Corruptions**](#adding-corruptions) for more details.

### [_timit_](timit)
The _timit_ folder contains the uncorrupted speech samples used to make the datasets. All sound files encountered in this folder (searching all subfolders recursively) are converted to conforming `.wav` files during the data preparation step. The separation into test and training set in TIMIT is ignored when new datasets are created. If you want to add other speech files to the datasets you can place them here, but note that the new files should also be free from any corruptions.

## Running Experiments
If you have followed the installation instructions it should be possible to run the `experiments.py` script in the root folder. In the beginning of this file there are four constants that you can change to run different experiments.
```python
# Dataset definition file
ds_json = os.path.join(root, "degradation", "basic_dataset.json")
# Noise classes definition file
nc_json = os.path.join(root, "noise_classes", "noise_classes.json")
# The number of realizations of the dataset, i.e. copies with the same settings but different random initializations
n_realizations = 2
# The number of trials for each realization
n_trials = 3
```
For example, if you want to average the results over more repeated trials you can increase `n_trials`, or if you want to work with a different dataset you can specify its definition in `ds_json`.

The script will automatically check for changes to the dataset parameters and definitions of the noise classes (corruptions) and regenerate the datasets if they are outdated. The results will be written to the console and `testresults/Test [current time]`. Additional output may be written to `testresults/log.txt`.

### Generating Datasets
Datasets are defined through both a dataset definition file and a noise classes specification. A [dataset definition](degradation/basic_dataset.json) file contains an object with the following fields:

- `"name"`, the name of the dataset. This is used as the name of the folder that contains the dataset and the filename of trained `.noiseclasses` files in _classifier/models_.
- `"train"`, the number of files in the training set. If less than 1 it is used as a fraction.
- `"test"`, the equivalent for the testing set.
- `"weights"`, a mapping of noise class identifiers to fractions. Describes the ratio of files which are given the corruption.
- `"pipeline"`, a list of noise class or operation (see below) identifiers. The list describes the order in which degradations are applied. All identifiers in the `weights` mapping must occur in the `pipeline` list.

Additionally, two additional fields may be present:
- `"operations"`, a list of objects similar to noise classes. An operation has an identifier (`"name"`) and a list of degradations. If an operation is present in `pipeline` it is applied to all samples at that point unless the sample has been assigned incompatible labels.
- `"incompatible"`, a list of noise class and operation identifier combinations. These are resolved as follows:
	1. Labels are randomly assigned to samples according to `weights`.
	2. The samples which are assigned all the noise class identifiers present in a combination have those labels replaced with only one of them. The probability of a label being chosen to remain is proportional to the label's weight.
	3. If there is an operation identifier present in the combination, that operation is omitted for all samples in step 2.

	A combination must have at least two identifiers, of which at most one may be an operation identifier.

The degradations in each noise class is not defined in the dataset definition but in a [noise classes specification](noise_classes/noise_classes.json). (In retrospect this may have been a poor design choice, but here we are.) This is a file containing a list of noise class objects. The parts relevant for dataset generation are the `"id"` and `"degradations"` fields.

The `degradations` list contains one or more objects specifying the degradation function in the ADT that will be invoked, as well as its parameters. `name` must be one of the names listed in [`degradation/degradations.py`](degradation/degradations.py), for example `"applyHighpassFilter"` or `"addSound"`. The arguments accepted varies by degradation (for example, `addSound` takes `normalizeOutputAudio` and `snrRatio`) and are listed in the degradation's Matlab file. Additionally, the intermediate script [`degradation/create_samples.m`](degradation/create_samples.m) interprets some arguments (such as `addSoundFile`, i.e. the soundfile being added to the sample). Arguments are generally floats or strings, but certain special objects can be used to get an element of randomness. These are:
- `{"randomNoiseFile": [string]}`, will get a random file from _noise/[string]_ belonging the corresponding set (training or testing).
- `{"randomChoice": [array]}`, will pick a random element from the array provided.
- `{"randomRange": [array of length 2]}`, will pick a random float in the provided range.
- `{"randomRangeInt": [array of length 2]}`, will pick a random integer in the provided range.

The `degradations` list is interpreted the same way whether it is part of a noise class or an operation.

If you are using the `experiments.py` script to run your tests, datasets are generated and updated for you. However, you can also use [`degradation/create_dataset.py`](degradation/create_dataset.py) directly to create your datasets yourself. Try `degradation/create_dataset.py --help` or `degradation/create_dataset.py create --help` for information on specific usage.

### Adding Corruptions
Assume that we want to add a corruption that applies a DC offset to the waveform. As there is no such degradation in the ADT we need to implement a custom degradation. We start by creating a copy of [`template.m`](degradation/custom-degradations/template.m) in _degradation/custom-degradations_. Let us name this copy `degradationUnit_addDCOffset.m`. Open the file in a text editor or Matlab.

Replace `template` with `degradationUnit_addDCOffset` in the function name on the first row and in the comment below. Add the current date, your name and a description of what the degradation will do. Reasonable arguments to our degradation is the amount of bias and whether we take into account the existing bias or not. We add the two arguments `bias`, with a default of `0.05`, and `relative`, with a default of `0`.

To add the bias we replace `f_audio_out = f_audio;` in the main program section with the following:
```matlab
f_audio_out = f_audio + parameter.bias;
if ~parameter.relative
    f_audio_out = f_audio_out - mean(f_audio);
end
```

The entire file should now look like this:
```matlab
function [f_audio_out,timepositions_afterDegr] = degradationUnit_addDCOffset(f_audio, samplingFreq, timepositions_beforeDegr, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: degradationUnit_addDCOffset
% Date: 2020-12
% Programmer: Your Name
%
% Description:
% - adds a DC bias to the audio
%
% Input:
%   f_audio      - audio signal \in [-1,1]^{NxC} with C being the number of
%                  channels
%   timepositions_beforeDegr - some degradations delay the input signal. If
%                             some points in time are given via this
%                             parameter, timepositions_afterDegr will
%                             return the corresponding positions in the
%                             output. Set to [] if unavailable. Set f_audio
%                             and samplingFreq to [] to compute only
%                             timepositions_afterDegr.
%
% Input (optional): parameter
%   .bias     = 0.05 - The amount of offset.
%   .relative = 0    - Whether the new bias replaces the previous or
%                      is added to it.
%
%   timepositions_beforeDegr - some degradations delay the input signal. If
%                              some points in time are given via this
%                              parameter, timepositions_afterDegr will
%                              return the corresponding positions in the
%                              output
%
% Output:
%   f_audio_out  - audio output signal
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<4
    parameter=[];
end
if nargin<3
    timepositions_beforeDegr=[];
end
if nargin<2
    error('Please specify input data');
end

if isfield(parameter,'bias')==0
    parameter.bias = 0.05;
end
if isfield(parameter,'relative')==0
    parameter.relative = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_audio_out = [];
if ~isempty(f_audio)
    f_audio_out = f_audio + parameter.bias;
    if ~parameter.relative
        f_audio_out = f_audio_out - mean(f_audio);
    end
end

% This degradation does not impose a delay
timepositions_afterDegr = timepositions_beforeDegr;

end
```

As we have named our file with the prefix `degradationUnit_` and put it in the _custom-degradations_ folder, `create_dataset.m` will automatically find our Matlab function for us. However, we also need to make sure the Python script knows our degradation exists. To do this we open [`degradation/degradations.py`](degradation/degradations.py) and add `"addDCOffset"` to the `DEGRADATIONS` constant. It should now read:
```python
DEGRADATIONS = (
	"", "pad", "addSound", "applyImpulseResponse", "adaptiveEqualizer", "applyMfccMeanAdaption",
	"normalize", "applyMute", "applySoftClipping", "addNoise", "applyAliasing", "applyClipping",
	"applyClippingAlternative", "applyDelay", "applyDynamicRangeCompression", "applyHarmonicDistortion",
	"applyHighpassFilter", "applyLowpassFilter", "applySpeedup", "applyWowResampling", "addInfrasound", "addDCOffset"
)
```

Next, we have to add our new corruption to [`noise_classes/noise_classes.json`](noise_classes/noise_classes.json). To introduce some variety we set the `bias` argument to a random value between `0.02` and `0.07`. As the offset is likely more detectable in the time domain than the frequency domain we use a histogram feaure and a GMM classifier for now. Add the following to the end of the list, right before the final `]` symbol:
```json
, {
	"id": "dc-offset",
	"name": "DC Offset",
	"degradations": [
		{
			"name": "addDCOffset",
			"bias": {"randomRange": [0.02, 0.07]}
		}
	],
	"classifiers": [
		{
			"type": "GMM",
			"parameters": "default",
			"train": "default",
			"feature": "histogram",
			"feature_settings": {}
		}
	]
}
```

Finally, we also have to update our dataset specification with the new noise class - otherwise no samples will have the corruption, and no classifiers can be trained to detect it. Open [`degradation/basic_dataset.json`](degradation/basic_dataset.json) and add `"dc-offset":		0.125` to `"weights"`. Also add `"dc-offset"` to `"pipeline"` before `"clipping-soft"`. The result should look like:
```json
...
	"weights": {
		"add-white":		0.125,
		"add-aircon":		0.125,
		"add-hum":			0.125,
		"add-infra":		0.125,
		"clipping-hard":	0.125,
		"clipping-soft":	0.125,
		"dc-offset":		0.125,
		"mute":				0.125
	},
	"pipeline": [
		"pad-1",
		"add-white",
		"add-aircon",
		"add-hum",
		"add-infra",
		"mute",
		"dc-offset",
		"clipping-soft",
		"clipping-hard",
		"normalize-rand"
	],
...
```

Now you are all set to try out the new corruption! Run `experiments.py` to generate the dataset and train/test the classifiers. For reference I get a 96.9% balanced accuracy. (Note, however, that we have only allowed a positive bias.)

To summarize, the steps are as follows:
1. If needed, create a Matlab script to perform some operation on the audio signal. You can use [`degradation/custom-degradations/template.m`](degradation/custom-degradations/template.m) as a starting point.
2. Register the script in [`degradation/degradations.py`](degradation/degradations.py).
3. Create a new noise class in [`noise_classes/noise_classes.json`](noise_classes/noise_classes.json).
4. Add the noise class identifier to a dataset's `weights` and `pipeline`.

### Adding Features
In the previous section we used a histogram feature to detect the DC offset corruption. While this certainly may be a good starting point, a more straightforward approach would be to use the mean of the signal as a feature. We thus need to implement it and add it to the appropriate files.

Open [`features/custom.py`](features/custom.py). Every feature function receives the list of files from which features will be extracted as the first argument. Settings for the computation can be provided via keyword arguments, which will be filled in by the fields in the noise class' `"feature_settings"` object. For instance, we can specify the number of bins in the histogram feature by writing e.g.
```python
"feature_settings": {"n_bins": 10}
```
In our current case, with the mean feature, there is not really any more information we need to compute the feature, so we accept no keyword arguments. At the bottom of `custom.py` we add the following method definition:
```python
def mean(filenames):
	import librosa
	
	res = []
	for fn in filenames:
		y, _ = librosa.load(fn, None)
		value = np.mean(y).reshape(-1, 1)
		res.append(value)
	
	return res
```
We use the `librosa` module to read each sample, and `numpy` to compute the mean. The return value of a feature function must be a list of `numpy` arrays, one for each file. The arrays should have the shape `(T, dims)`, where `T` is the length of the time sequence and `dims` number of feature dimensions. `dims` should be the same for each array, but `T` may vary. In our case we are only computing a scalar, so we convert it into a 1x1 array. (The arrays also need to be of the `float32` type. While this is the default in many cases, certain `librosa` spectrum computations return `float64`.)

The next step is to let [`features/feature_extraction.py`](features/feature_extraction.py) know of our new method. At the start we add `mean` to the import statement from `.custom`:
```python
from .custom import acf, histogram, rms_energy, \
	rms_energy_infra, mfcc_kaldi_full, histogram_local, \
	mfcc_kaldi_delta, mean
```
We also add it to the `FEATURES` constant:
```python
FEATURES = (mfcc_kaldi, mfcc_librosa, acf, histogram,
	rms_energy, rms_energy_infra, mfcc_kaldi_full,
	histogram_local, mfcc_kaldi_delta, mean)
```
While we are editing this file it is worth to note the `CACHE_SIZE` constant defined on the row below `FEATURES`. Extracting features can take a long time, and if we are training multiple classifiers on the same datasets and using the same features it would be redundant to keep recomputing them. Instead we save a number of computed features to the _cache_ folder (at most `CACHE_SIZE`, after which the oldest ones are removed). While this is practical during experiments it is less so if you are debugging a feature method, in which case it is suggested to set `CACHE_SIZE` to `0` or to clear it manually by deleting the files after each run.

Now, the only thing to do is to swap out the histogram and replace it with the mean. In [`noise_classes/noise_classes.json`](noise_classes/noise_classes.json), replace `"feature": "histogram"` with `"feature": "mean"` in the DC offset noise class. If you re-run the experiment you should see that `experiment.py` does not need to regenerate the dataset, and that the GMM classifier for `dc-offset` now uses `mean` features. When testing this the balanced accuracy increases to 99.97% for me, but as there are elements of randomness the results may vary slightly.

#### Voice Activity Detection
For corruptions that only arise during speech it may be interesting to filter out silent segments, and for those that are related to background noise it may be beneficial to remove segments where the speech is dominant. To this end there is a way to split the recordings using a Voice Activity Detector (VAD). By adding a `"vad"` field to an object in the `"classifiers"` array of a noise class you indicate that a VAD will be used, and can specify settings which are passed to [`features/vad.py`](features/vad.py).

Both `frame_length` and `aggressiveness` are required arguments to the VAD. Available settings are:
- `frame_length`, the length of the VAD analysis frame in milliseconds. It must be one of 10, 20, or 30.
- `aggressiveness`, how likely the VAD is to consider a frame non-speech. Must be one of 0, 1, 2, or 3.
- `inverse`, whether to return the voiced (`false`) or the unvoiced (`true`) segments. Defaults to `false`.
- `smooth_n`, the length of the smoothing window used. If not set, smoothing is not used at all. Smoothing computes a running average of the VAD verdict of the `smooth_n` nearest frames. This ensures that a single mislabeled frame does not affect the segmentation.
- `smooth_threshold`, the threshold that must be reached by the running average to toggle between the voiced/unvoiced state. For example, if `smooth_threshold = 0.8` then 80% of the frames must be labeled voiced for a voiced segment to start, after which 80% of frames must be labeled unvoiced to end the segment. Defaults to `0.5`.
- `min_length`, the minimum length of a segment in milliseconds. Segments that are shorter than this are discarded. Defaults to `0`, which will include all segments.

### Adding Classifiers

Assume that we want to see how a new classification algorithm fares against the others in the test. We choose the Decision Tree (DT) algorithm which conveniently is implemented in scikit-learn, a module we are already using. To interface with the feature extraction and other parts of the tool all algorithms must inherit from [`classifier/model.py`](classifier/model.py). We start by creating a new file, `model_dt.py` in the [_classifier_](classifier) folder with the following content:
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from .model import Model

class DT(Model):
	MULTICLASS = True
	def __init__(self, config: dict, noise_types: list):
		self.dtc = DecisionTreeClassifier(**config["parameters"])
		self.noise_types = noise_types
	
	def get_noise_types(self):
		return self.noise_types
	
	def train(self, train_data, index, labels, config: dict=None):
		# Ensure all samples are single vectors (disallow sequences of vectors)
		assert all(sequence.shape[0] == 1 for sequence in train_data)

		X = np.concatenate(train_data)
		self.dtc.fit(X, labels)
	
	def score(self, test_data, index):
		# Ensure all samples are single vectors (disallow sequences of vectors)
		assert all(sequence.shape[0] == 1 for sequence in test_data)
		# Ensure the VAD is not used
		assert all(np.arange(len(index)) == index)

		X = np.concatenate(test_data)
		return self.dtc.predict(X)
```

For brevity we take some shortcuts in this classifier by disallowing sequences of vectors (such as MFCC features) and VAD splitting.

To make the classifier usable we need to add it to [`noise_classes/interface.py`](noise_classes/interface.py), where the names are resolved. In the beginning of the file there are a number of rows where the other classifiers are imported. Add a line importing the DT classifier and add it to `available_models` like so:
```python
from classifier.model_lstm import LSTM
from classifier.model_gmmhmm import GMMHMM
from classifier.model_gmm import GMM
from classifier.model_dt import DT
available_models = (LSTM, GMMHMM, GMM, DT)
```
Before we can use it, however, we also need to add an entry specifying the (default) arguments to the classifier.

According to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) the `DecisionTreeClassifier` takes a number of possible keyword arguments, such as `criterion` or `max_depth`. We can provide our own defaults to these by adding them to [`classifier/defaults.json`](classifier/defaults.json), which is done by adding the following before the last curly brace:
```json
...
	,
	"DT": {
		"parameters": {
			"criterion": "gini",
			"max_depth": 10
		},
		"train": {}
	}
...
```

Next, open [`noise_classes/noise_classes.json`](noise_classes/noise_classes.json) and change the classifier type of DC offset from `GMM` to `DT`. Also change the feature type back to `histogram` if you modified it in the last step. If you did not implement the DC offset corruption you can use one of the clipping corruptions instead.

A decision tree will now be used to detect the corruption. Note that the classifier specification also contains `"parameters"` and `"train"`, just like `defaults.json`. Setting these to `"default"` will copy the settings from `defaults.json`, but if we want to override a setting specifically for a certain corruption it can be done here. In that case we need to specify all the fields in the category we are replacing, for example setting
```json
"parameters": {
	"criterion": "entropy"
}
```
in `noise_classes.json` would change the `criterion` argument to `"entropy"`, but also unset the `max_depth` argument.

Now, having implemented `DT`, added it to `defaults.json` and `interface.py`, and specified it as the classifier for the DC offset, we are all set to test it out. Try running `experiments.py` and see if the DT classifier performs as well as the GMM. For reference, I get a 99.7% balanced accuracy on DC offset.

The arguments `train_data` and `test_data` in the `test()` and `score()` methods of `model_dt.py` are the vector sequences returned by the feature extraction methods. Recall that they consist of numpy arrays of shape `(T, dims)`, where `dims` is the same for all arrays. If the VAD is used there are in general multiple arrays per sample, corresponding to the different voiced or unvoiced segments in the recording. The `index` variable can be used to keep track of these: array `test_data[i]` belongs to file number `index[i]`. During labeling most existing classifiers score segments separately and then compute the weighted mean of the scores, which is used as the score for that recording. (Note that `score()` should return one score per recording, not per segment, so `test_data` is generally longer than the returned score array when VAD filtering is used.)

#### Ensemble Classification
The keen reader will have noticed that the `"classifiers"` field in a noise class is actually an array of objects. This is because you can specify multiple classifiers for the corruption which will vote on the final label. Simply add another classifier specification to the list - these do not have to use the same algorithm or even features. You can specify the weights of individual classifiers by adding a `"weight"` field (the default is `1`), or indicate that the classifier should train on a bootstrapped sample by adding `"bootstrap": true`. If you have multiple classifiers of the same type and want to average the score rather than the label during voting, you can add a field to the noise class specifying `"classification_settings": {"average": "score"}`.

Note that since training is done separately for each classifier, the training time will increase considerably when ensemble classification is used.
