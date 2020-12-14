function [f_audio_out,timepositions_afterDegr] = template(f_audio, samplingFreq, timepositions_beforeDegr, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: template
% Date: 2020-12
% Programmer: Your Name
%
% Description:
% - a description
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
%   .someParameter = 'defaultValue' - a description
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

if isfield(parameter,'someParameter')==0
    parameter.someParameter = 'defaultValue';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_audio_out = [];
if ~isempty(f_audio)
    f_audio_out = f_audio;
end

% This degradation does not impose a delay
timepositions_afterDegr = timepositions_beforeDegr;

end
