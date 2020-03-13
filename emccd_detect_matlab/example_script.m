% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020


% Input fluxmap
fits_name = 'ref_frame.fits';
current_path = pwd;
fits_path = fullfile(fileparts(current_path), 'fits', fits_name);
fluxmap = fitsread(fits_path);  % Input fluxmap (photons/pix/s)

% Simulation inputs
exptime = 100.;  % Frame time (seconds)
em_gain = 1000.;  % CCD gain (e-/photon)
full_well_image = 60000.;  % Readout register capacity (e-)
full_well_serial = 10000.;  % Serial register capacity (e-)
dark_current = 0.0056;  % Dark rate (e-/pix/s)
cic = 0.01;  % Charge injection noise (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 0.9;  % Quantum effiency
cr_rate = 5.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)
shot_noise_on = true;  % Apply shot noise

% Simulate single image
sim_im = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
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
                  em_gain, read_noise, exptime)})
end
