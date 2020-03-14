% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
close all; clear; clc;


% Input fluxmap
fits_name = 'ref_frame.fits';
current_path = pwd;
fits_path = fullfile(fileparts(current_path), 'fits', fits_name);
fluxmap = fitsread(fits_path);  % Input fluxmap (photons/pix/s)

% Simulation inputs
frametime = 100.;  % Frame time (seconds)
em_gain = 1000.;  % CCD EM gain (e-/photon)
full_well_image = 60000.;  % Image area full well capacity (e-)
full_well_serial = 10000.;  % Serial (gain) register full well capacity (e-)
dark_current = 0.0056;  % Dark  current rate (e-/pix/s)
cic = 0.01;  % Clock induced charge (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 0.9;  % Quantum effiency
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
          sprintf('Gain: %.0f   Read Noise: %.0fe-   Frame Time: %.0fs',...
                  em_gain, read_noise, frametime)})
end

%% evaluate simulation
% the point of this is to see how well the distributions look, coming out
% of radnEMGain1
Ntries = 2500;
x = zeros(1, Ntries);
NinValues = [1:3:16,50, 60, 70];
for Nin = NinValues
    tic
    for it = 1:Ntries
        x(it) = rand_em_gain(Nin, em_gain);
    end
    tn = toc; tper = tn / Ntries;
    figure, histbn(x, 80, 'all'); grid;
    title(['Nin = ', num2str(Nin),' EMgain = ',num2str(em_gain), ' mean = ',...
        num2str(mean(x),'%5.0f'),' (',num2str(tper*1000,'%6.3f'),' ms per call)'])
end

autoArrangeFigures
