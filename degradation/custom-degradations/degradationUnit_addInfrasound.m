function [f_audio_out,timepositions_afterDegr] = degradationUnit_addInfrasound(f_audio, samplingFreq, timepositions_beforeDegr, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: degradationUnit_addInfrasound
% Date: 2020-09
% Programmer: Helmer Nylen
%
% Description:
% - create an infrasound consisting of noise and sinusoids and add it to
% the original
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
%   .sinusoidFreqs = [r1, r2],    - frequencies of the infrasound sine
%      where r = 2+rand()*16        tones
%   .sinusoidAmps = 1             - amplitudes of the infrasound sine tones
%   .noiseSNR = 0                 - SNR of the waveforms to the noise
%   .noiseColor = 'white'         - noise color
%   .stopFrequency = 20           - argument to the low pass filter
%   .passFrequency                - argument to the low pass filter
%   .normalizeOutputAudio = 1     - peak normalize audio after adding the
%                                   signals
%   .snrRatio = 20  - in dB.
%                     combined noise is scaled such that a signal to noise ratio snrRatio is obtained.
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

if isfield(parameter,'sinusoidFreqs')==0
    parameter.sinusoidFreqs = [2 + rand() * 16, 2 + rand() * 16];
end
if isfield(parameter,'sinusoidAmps')==0
    parameter.sinusoidAmps = ones(size(parameter.sinusoidFreqs));
end
if isfield(parameter,'noiseSNR')==0
    parameter.noiseSNR = 0;
end
if isfield(parameter,'noiseColor')==0
    parameter.noiseColor = 'pink';
end
if isfield(parameter,'snrRatio')==0
    parameter.snrRatio = 20;
end
if isfield(parameter,'stopFrequency')==0
    parameter.stopFrequency = 20;
end
% passFrequency handled by lowpassfilter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_audio_out = [];
if ~isempty(f_audio)
    if iscell(parameter.sinusoidFreqs)
        parameter.sinusoidFreqs = cell2mat(parameter.sinusoidFreqs);
    end
    if iscell(parameter.sinusoidAmps)
        parameter.sinusoidAmps = cell2mat(parameter.sinusoidAmps);
    end
    
    sinusoids = zeros(size(f_audio, 1), length(parameter.sinusoidFreqs));
    for i = 1:length(parameter.sinusoidFreqs)
        sinusoids(:,i) = parameter.sinusoidAmps(i) * sin(2 * pi * parameter.sinusoidFreqs(i) * (1:size(f_audio, 1)) / samplingFreq)';
    end
    sinusoids = adthelper_normalizeAudio(sum(sinusoids, 2), samplingFreq);
    
    total_noise = degradationUnit_addNoise(sinusoids, samplingFreq, [], struct(...
        'noiseColor', parameter.noiseColor, 'snrRatio', parameter.noiseSNR, ...
        'normalizeOutputAudio', 1));
    total_noise = degradationUnit_applyLowpassFilter(total_noise, samplingFreq, [], parameter);
    
    parameter.loadInternalSound = 0;
    parameter.addSound = total_noise;
    parameter.addSoundSamplingFreq = samplingFreq;
    f_audio_out = degradationUnit_addSound(f_audio, samplingFreq, [], parameter);
end

% This degradation does not impose a delay
timepositions_afterDegr = timepositions_beforeDegr;

end
