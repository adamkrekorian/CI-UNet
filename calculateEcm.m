% calculateEcm.m
% Author: Kevin Chu
% Last Modified: 10/12/2020

function ECM = calculateEcm(FTMclean, FTMnoisy, f_stim, ecm_type)
    % Calculates the envelope-correlation based measure between a noisy (or
    % reverberant) FTM and a clean FTM. Method from Yousefian and Loizou,
    % 2012
    %
    % Args:
    %   -FTMclean (nChannels x nFrames): FTM of clean signal
    %   -FTMnoisy (nChannels x nFrames): FTM of noisy or reverberant signal
    %   -f_stim (double): stimulation frequency of CI in Hz
    %   -ecm_type (string): choices: mean or electrode. Mean calculates the
    %    ecm across all electrodes, whereas electrode calculates the ECM for
    %    each electrode separately
    %
    % Returns:
    %   -ECM (double): envelope-correlation based measure
    %
    % Reference:
    % -Predicting the speech reception threshold of cochlear implant
    % listeners using an envelope-correlation based measure

    % Downsample envelopes to 50Hz
    f_env = 50;
    FTMclean = resample(FTMclean',f_env,f_stim)';
    FTMnoisy = resample(FTMnoisy',f_env,f_stim)';
    
    % Mean across time
    muClean = repmat(mean(FTMclean,2),1,size(FTMclean,2));
    muNoisy = repmat(mean(FTMnoisy,2),1,size(FTMnoisy,2));
    
    % Covariance for each frequency band
    rk = sum((FTMclean-muClean).*(FTMnoisy-muNoisy),2)./(sqrt(sum((FTMclean-muClean).^2,2).*sum((FTMnoisy-muNoisy).^2,2)));    
   
    if strcmp(ecm_type, 'mean')
        ECM = nanmean(rk.^2);
    elseif strcmp(ecm_type, 'electrode')
        ECM = rk.^2;
    end
    
end
