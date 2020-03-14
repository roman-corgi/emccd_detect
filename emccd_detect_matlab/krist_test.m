% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
clear; close all; clc; format compact;
addpath('./util');
jMon = 2; fsz = 400*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

% Input fluxmap
npix_across = 100;
flux = 0.008;  % photns/pix/s
fluxmap = flux * ones(npix_across);

% Simulation inputs
frameTime = 10.;  % Frame time (seconds)
em_gain = 4000.;  % CCD EM gain (e-/photon)
full_well_image  = 60000.;  % Image area full well capacity (e-)
full_well_serial = 90000.;  % Serial (gain) register full well capacity (e-)
dark_current = 0.00028;  % Dark  current rate (e-/pix/s)
cic = 0.02;  % Clock induced charge (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 1.;  % Quantum effiency
cr_rate = 0.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)


zeroFrame = zeros(size(fluxmap)); %#ok<*NOPTS>
npts = 14;
pc_thresh = linspace(200, 800, npts);
for ithr = 1:npts
    
    % Threshold and photon count
    
    nthr(ithr) = pc_thresh(ithr) / read_noise;
    %  dark frame
    darkFrame = emccd_detect(zeroFrame, frameTime, em_gain, full_well_image,...
        full_well_serial, dark_current, cic, read_noise, bias,...
        qe, cr_rate, pixel_pitch, true);
    dark_an_mn(ithr) = mean(darkFrame(:));
    % photon-count the dark frame
    dark_PC = zeroFrame;
    dark_PC(darkFrame > pc_thresh(ithr)) = 1;
    
    % bright frame
    brightFrame = emccd_detect(fluxmap, frameTime, em_gain, full_well_image,...
        full_well_serial, dark_current, cic, read_noise, bias,...
        qe, cr_rate, pixel_pitch, true);
    bright_an_mn(ithr) = mean(brightFrame(:));
    bright_PC = zeroFrame;
    bright_PC(brightFrame > pc_thresh(ithr)) = 1;
    %     figure, imagesc(bright_PC); colorbar; colormap gray;
    
    % analysis of photon counted frames
    
    % Number of zero elements
    r_df(ithr) = (1/frameTime) * (nnz(dark_PC) / npix_across^2); %#ok<*SAGROW>
    
    
    % observed mean rate after photon counting
    n_obs(ithr) = nnz(bright_PC) / npix_across^2;
    
    eps_thr(ithr) = exp(- pc_thresh(ithr) / em_gain);
    
    lambda = -log(1-(n_obs(ithr)/eps_thr(ithr)));
    
    rtrue(ithr) = lambda / frameTime;
    
    % photo-electron rate
    r_phe(ithr) = rtrue(ithr) - r_df(ithr);
    
        figure;
        imagesc(brightFrame); %, [0, 2*em_gain*flux*qe*frameTime]
        colorbar;
    %
    
end

%%
figure
plot(nthr, n_obs/frameTime, nthr, r_phe,  nthr, flux*ones(1, npts))
grid
legend('Observed', 'Corrected', 'Actual')
xlabel('threshold factor')
ylabel('rates, e/pix/s')

figure
plot(nthr, eps_thr)
grid
xlabel('threshold factor')
ylabel('threshold effeciency')
title('Assuming all pixels are 1 or 0 real ph-e''s')

acf = 1/em_gain/frameTime;
figure
darksub_an = (bright_an_mn-dark_an_mn);
plot(nthr, bright_an_mn*acf, nthr, dark_an_mn*acf, nthr, (bright_an_mn-dark_an_mn)*acf);
grid
xlabel('threshold factor')
ylabel('rates, e/pix/s')

mean(darksub_an/em_gain/frameTime)

autoArrangeFigures(nr, nc, iMon); return;

return
% n_frames = 500;
% frames_shot_off = zeros(npix_across, npix_across, n_frames);
% frames_shot_on = zeros(npix_across, npix_across, n_frames);
% for i = 1:n_frames
%     frames_shot_off(:, :, i) = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
%                                             full_well_serial, dark_current, cic, read_noise, bias,...
%                                             qe, cr_rate, pixel_pitch, false);
%     frames_shot_on(:, :, i) = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
%                                            full_well_serial, dark_current, cic, read_noise, bias,...
%                                            qe, cr_rate, pixel_pitch, true);
% end