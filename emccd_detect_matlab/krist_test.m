% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
clear; close all; clc;


% Input fluxmap
npixacross = 10;
flux = 0.1;  % photns/pix/s
fluxmap = flux * ones(npixacross);

% Simulation inputs
exptime = 10.;  % Frame time (seconds)
em_gain = 6000.;  % CCD gain (e-/photon)
full_well_image = 60000.;  % Readout register capacity (e-)
full_well_serial = 10000.;  % Serial register capacity (e-)
dark_current = 1/3600;  % Dark rate (e-/pix/s)
cic = 0.02;  % Charge injection noise (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 1.;  % Quantum effiency
cr_rate = 0.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)
shot_noise_on = true;  % Apply shot noise

npts = 3;
cuts = linspace(2, 6, 10);
for icut = 1:npts
    cut = cuts(icut)

    sim_im = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
                          full_well_serial, dark_current, cic, read_noise, bias,...
                          qe, cr_rate, pixel_pitch, shot_noise_on);

    % Threshold and photon count
    pc_thresh = cut * read_noise;
    pc_image = zeros(size(fluxmap));
    pc_image(sim_im > pc_thresh) = 1;

    % Number of zero elements
    pc_num = numel(pc_image) - nnz(pc_image);

    figure;
    imagesc(sim_im, [0, 2*em_gain*flux*qe*exptime]);
    colorbar;
end
