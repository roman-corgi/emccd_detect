% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020


% Input fluxmap
fits_name = 'ref_frame.fits';
current_path = pwd;
fits_path = fullfile(fileparts(current_path), 'fits', fits_name);
fluxmap = fitsread(fits_path);  % photons/pix/s

% Simulation inputs
exptime = 100.0;  % Frame time (seconds)
gain = 1000.0;  % CCD gain (e-/photon)
full_well_serial = 10000.0;  % Serial register capacity (e-)
full_well = 60000.0;  % Readout register capacity (e-)
dark_rate = 0.0056;  % Dark rate (e-/pix/s)
cic_noise = 0.01;  % Charge injection noise (e-/pix/frame)
read_noise = 100;  % Read noise (e-/pix/frame)
bias = 0.0;  % Bias offset (e-)
quantum_efficiency = 0.9;
cr_rate = 5;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13 * 10^-6;  % Distance between pixel centers (m)
apply_smear = true;  % Apply LOWFS readout smear
% 
% Simulate single image
sim_im = emccd_detect(fluxmap, exptime, gain, full_well_serial, full_well,...
                      dark_rate, cic_noise, read_noise, bias,...
                      quantum_efficiency, cr_rate, pixel_pitch, apply_smear);

% Plot images
plot_images = true;
if plot_images
    figure;
    imagesc(fluxmap); colorbar;
    title('Input Fluxmap')

    figure;
    imagesc(sim_im); colorbar;
    title({'Output Fluxmap',...
          sprintf('Gain: %.0f    RN: %.0fe-    t_{fr}: %.0fs',...
                  gain, read_noise, exptime)})
end
