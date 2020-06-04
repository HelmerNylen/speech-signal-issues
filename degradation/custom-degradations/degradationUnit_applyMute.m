function [f_audio_out,timepositions_afterDegr] = degradationUnit_applyMute(f_audio, samplingFreq, timepositions_beforeDegr, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: degradation_applyMute
% Date: 2020-06
% Programmer: Helmer Nylen
%
% Description:
% - randomly mutes audio
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
%   .mutePercent     = 10  - amount of the recording which is muted (only
%                            used if .totalMuteLength is not set)
%   .totalMuteLength       - total mute time in seconds
%   .muteLengthMin   = 0.2 - minimum length in seconds of a mute segment
%   .muteLengthMax   = 1   - maximum length in seconds of a mute segment
%   .mutePauseMin    = 0.2 - minimum length in seconds between mute
%                            segments
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

if isfield(parameter,'muteLengthMin')==0
    parameter.muteLengthMin = 0.2;
end
if isfield(parameter,'muteLengthMax')==0
    parameter.muteLengthMax = 1;
end
if isfield(parameter,'mutePauseMin')==0
    parameter.mutePauseMin = 0.2;
end
if isfield(parameter,'totalMuteLength')==0
    if isfield(parameter,'mutePercent')==0
        parameter.mutePercent = 10;
    end
    parameter.totalMuteLength = (size(f_audio, 1) / samplingFreq) * parameter.mutePercent / 100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_audio_out = [];
if ~isempty(f_audio)
    tot_audio_len = size(f_audio, 1);
    tot_mute_len = floor(parameter.totalMuteLength * samplingFreq);
    min_mute = floor(parameter.muteLengthMin * samplingFreq);
    max_mute = floor(parameter.muteLengthMax * samplingFreq);
    min_pause = floor(parameter.mutePauseMin * samplingFreq);
    
    assert(min_mute <= max_mute);
    assert(tot_mute_len <= tot_audio_len);
    
    if tot_mute_len <= 2 * min_mute
        mute_lens = [tot_mute_len];
        n_mutes = 1;
    else
        min_mutes_needed = ceil(tot_mute_len / max_mute);
        max_mutes_possible = floor(min(tot_mute_len / max(1, min_mute), (min_pause + tot_audio_len) / max(1, min_pause + min_mute)));
        if min_mutes_needed > max_mutes_possible
            warn("Degenerate mute case: min needed " + string(min_mutes_needed) ...
                + ", max possible " + string(max_mutes_possible));
            n_mutes = max(1, max_mutes_possible);
        else
            n_mutes = randi([min_mutes_needed, max_mutes_possible]);
        end
        mute_lens = ones(n_mutes, 1) * min_mute;
        add = zeros(n_mutes, 1);
        s = 0;
        for i=1:n_mutes
            add(i) = randi([0 min(max_mute - min_mute, max(0, tot_mute_len - s - min_mute * n_mutes))]);
            s = s + add(i);
        end
        mute_lens = mute_lens + add(randperm(n_mutes));
        s = sum(mute_lens);
        while s ~= tot_mute_len
            d = min(n_mutes, abs(tot_mute_len - s));
            add(1:d) = sign(tot_mute_len - s);
            add(d+1:end) = 0;
            mute_lens = min(max_mute, max(min_mute, mute_lens + add(randperm(n_mutes))));
            s = sum(mute_lens);
        end
    end
    pause_lens = [0; min_pause * ones(n_mutes - 1, 1); 0];
    add = zeros(n_mutes + 1, 1);
    s = 0;
    for i=1:(n_mutes + 1)
        add(i) = randi([0 max(0, tot_audio_len - tot_mute_len - s - min_pause * (n_mutes - 1))]);
        s = s + add(i);
    end
    pause_lens = pause_lens + add(randperm(n_mutes + 1));
    s = sum(pause_lens);
    while s ~= tot_audio_len - tot_mute_len
        d = min(n_mutes + 1, abs(tot_audio_len - tot_mute_len - s));
        add(1:d) = sign(tot_audio_len - tot_mute_len - s);
        add(d+1:end) = 0;
        pause_lens = max(min_pause, pause_lens + add(randperm(n_mutes + 1)));
        s = sum(pause_lens);
    end
    
    mute_lens = mute_lens(randperm(n_mutes));
    pause_lens(2:n_mutes) = pause_lens(1 + randperm(n_mutes - 1));
    pause_lens([1 length(pause_lens)]) = circshift(pause_lens([1 length(pause_lens)]), randi([0 1]));
    
    mask = arrayfun(@(x) ones(x, 1, 'logical'), pause_lens, 'UniformOutput', false);
    mask = [mask, arrayfun(@(x) zeros(x, 1, 'logical'), [mute_lens; 0], 'UniformOutput', false)];
    mask = cell2mat(reshape(mask', [], 1));
    
    f_audio_out = f_audio .* mask;
    
end

% This degradation does not impose a delay
timepositions_afterDegr = timepositions_beforeDegr;

end
