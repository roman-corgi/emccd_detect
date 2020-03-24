% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
clear; close all; clc; format compact;
addpath('../');
addpath('../util');

% Input fluxmap
fits_name = 'ref_frame.fits';
current_path = pwd;
fits_path = fullfile(fileparts(fileparts(current_path)), 'data', fits_name);
fluxmap = fitsread(fits_path);  % Input fluxmap (photons/pix/s)

% Simulation inputs
frametime = 100.;  % Frame time (seconds)
em_gain = 5000.;  % CCD EM gain (e-/photon)
full_well_image = 50000.;  % Image area full well capacity (e-)
full_well_serial = 90000.;  % Serial (gain) register full well capacity (e-)
dark_current = 0.0028;  % Dark current rate (e-/pix/s)
cic = 0.01;  % Clock induced charge (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 0.9;  % Quantum efficiency
cr_rate = 5.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)
shot_noise_on = true;  % Apply shot noise

% Simulate single image
sim_im = emccd_detect(fluxmap, frametime, em_gain, full_well_image,...
                      full_well_serial, dark_current, cic, read_noise, bias,...
                      qe, cr_rate, pixel_pitch, shot_noise_on);

% Plot images
plot_images = true;
if plot_images
    figure;
    imagesc(fluxmap); colorbar;
    title('Input Fluxmap')

    figure;
    imagesc(sim_im); colorbar;
    title({'Output Image',...
          sprintf('EM Gain: %.f   Read Noise: %.fe-   Frame Time: %.fs',...
                  em_gain, read_noise, frametime)})
end

if plot_images
    autoArrangeFigures;
end
