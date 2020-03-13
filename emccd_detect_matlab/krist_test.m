% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
clear; close all; clc;


% Input fluxmap
npix_across = 10;
flux = 0.1;  % photns/pix/s
fluxmap = flux * ones(npix_across);

% Simulation inputs
exptime = 10.;  % Frame time (seconds)
em_gain = 6000.;  % CCD EM gain (e-/photon)
full_well_image = 60000.;  % Image area full well capacity (e-)
full_well_serial = 10000.;  % Serial (gain) register full well capacity (e-)
dark_current = 0.00028;  % Dark  current rate (e-/pix/s)
cic = 0.02;  % Clock induced charge (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 1.;  % Quantum effiency
cr_rate = 0.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)

n_frames = 500;
frames_shot_off = zeros(npix_across, npix_across, n_frames);
frames_shot_on = zeros(npix_across, npix_across, n_frames);
for i = 1:n_frames
    frames_shot_off(:, :, i) = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
                                            full_well_serial, dark_current, cic, read_noise, bias,...
                                            qe, cr_rate, pixel_pitch, false);
    frames_shot_on(:, :, i) = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
                                           full_well_serial, dark_current, cic, read_noise, bias,...
                                           qe, cr_rate, pixel_pitch, true);
end

npts = 3;
cuts = linspace(2, 6, 10);
for icut = 1:npts
    cut = cuts(icut)

    sim_im = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
                          full_well_serial, dark_current, cic, read_noise, bias,...
                          qe, cr_rate, pixel_pitch, true);

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
