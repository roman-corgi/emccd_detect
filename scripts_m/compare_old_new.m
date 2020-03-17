clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');
jMon = 2; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

% Input fluxmap
npix_across = 1000;
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



tic
oldFrame = emccd_detect(fluxmap, frameTime, em_gain, full_well_image,...
    full_well_serial, dark_current, cic, read_noise, bias,...
    qe, cr_rate, pixel_pitch, true);
toc

figure
imagesc(oldFrame); colorbar;
title('Old');


tic
newFrame = emccd_detect_new(fluxmap, frameTime, em_gain, full_well_image,...
    full_well_serial, dark_current, cic, read_noise, bias,...
    qe, cr_rate, pixel_pitch, true);
toc
figure
imagesc(newFrame); colorbar;
title('New');


figure
imagesc(newFrame-oldFrame); colorbar;
title('New-Old');

autoArrangeFigures(nr, nc, iMon); return;
%-----------------STOP--------------------
