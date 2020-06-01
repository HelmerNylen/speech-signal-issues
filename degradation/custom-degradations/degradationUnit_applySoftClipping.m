function [f_audio_out,timepositions_afterDegr] = degradationUnit_applySoftClipping(f_audio, ~, timepositions_beforeDegr, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: degradation_applySoftClipping
% Date: 2020-05
% Programmer: Helmer Nylen
%             Adapted from degradation_applyClippingAlternative by Sebastian Ewert
%
% Description:
% - applies clipping by over-normalising
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
%   .clipAPercentageOfSamples = 1 - if set to zero a fixed number of
%                                   samples will be clipped
%   .percentOfSamples = 1         - used only if clipAPercentageOfSamples
%                                   is 1
%   .numSamplesClipped = 1        - used only if clipAPercentageOfSamples
%                                   is 0
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

if isfield(parameter,'clipAPercentageOfSamples')==0
    parameter.clipAPercentageOfSamples = 1;
end
if isfield(parameter,'percentOfSamples')==0
    parameter.percentOfSamples = 1;
end
if isfield(parameter,'numSamplesClipped')==0
    parameter.numSamplesClipped = 10000;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_audio_out = [];
if ~isempty(f_audio)
    sortedValues = sort(abs(f_audio(:)));
    numSamples = length(sortedValues);
    if parameter.clipAPercentageOfSamples
        idxStartSample = round( (1-parameter.percentOfSamples/100) * numSamples);
        divisor = min(sortedValues(idxStartSample:numSamples));
    else
        if parameter.numSamplesClipped > numSamples
            numSamplesClipped = numSamples;
        else
            numSamplesClipped = parameter.numSamplesClipped;
        end
        divisor = min(sortedValues(numSamples-numSamplesClipped+1:numSamples));
    end
    clear sortedValues
    
    divisor = max(divisor,eps);
    f_audio_out = f_audio * 0.999 / divisor;
    
    % https://ccrma.stanford.edu/~jos/pasp/Soft_Clipping.html
    transfer_func = @(x) x - x.^3/3;
    range = abs(f_audio_out) <= 1;
    f_audio_out(range) = transfer_func(f_audio_out(range));
    f_audio_out(f_audio_out < -1) = -2/3;
    f_audio_out(f_audio_out > 1) = 2/3;
    
end

% This degradation does not impose a delay
timepositions_afterDegr = timepositions_beforeDegr;

end
