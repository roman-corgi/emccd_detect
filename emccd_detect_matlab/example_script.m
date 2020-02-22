% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020


% Input fluxmap
current_path = '/Users/sammiller/Documents/GitHub/emccd_detect';
fits_name = 'ref_frame.fits';
fits_path = fullfile(current_path, 'emccd_detect_matlab', 'fits', fits_name);
fluxmap = fitsread(fits_path);

plot_images = true;

% Simulation inputs
frametime = 100.0;  % seconds
gain = 1000.0;
cr_rate = 0;  % hits/cm^2/s (set to 0 for no cosmics; 5 for L2 expected)
bias = 0.0;
qe = 0.9;  % quantum efficiency
fwc_im = 50000.0;  % full well capacity (image plane)
fwc_gr = 90000.0;  % full well capacity (gain register)
dark_current = 0.005;  % e-/pix/s
cic = 0.02;  % e-/pix/frame
read_noise = 100;  % e-/pix/frame -- amplifier noise (EMCCD CCD201 Type C)

% Simulate single image
sim_im = emccd_detect(fluxmap, cr_rate, frametime, gain, bias, qe, fwc_im,...
                      fwc_gr, dark_current, cic, read_noise);

% Plot image
if plot_images
    figure;
    imagesc(fluxmap); colorbar;
    title('Input Fluxmap')

    figure;
    imagesc(sim_im); colorbar;
    title({'Output Fluxmap',...
          sprintf('Gain: %.0f    RN: %.0fe-    t_{fr}: %.0fs',...
                  gain, read_noise, frametime)})
end
