% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020


% Input fluxmap
fits_name = 'ref_frame.fits';
current_path = pwd;
fits_path = fullfile(fileparts(current_path), 'fits', fits_name);
% fluxmap = fitsread(fits_path);  % photons/pix/s

Npixacross = 10;
flux = 0.1 % ph/s/pix
fluxmap = flux * ones(Npixacross);  % photons/pix/s

% Simulation inputs
exptime = 10.0;  % Frame time (seconds)
gain = 6000.0;  % CCD gain (e-/photon)
full_well_serial = 10000.0;  % Serial register capacity (e-)
full_well = 60000.0;  % Readout register capacity (e-)
dark_rate = 1/3600  % Dark rate (e-/pix/s)
cic_noise = 0.02;  % Charge injection noise (e-/pix/frame)
read_noise = 100;  % Read noise (e-/pix/frame)
bias = 0.0;  % Bias offset (e-)
quantum_efficiency = 1;
cr_rate = 5;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13 * 10^-6;  % Distance between pixel centers (m)
apply_smear = false;  % Apply LOWFS readout smear
%
% Simulate single image
Npts = 3;
cuts = linspace(2,6,10)
for icut = 1:Npts
    cut = cuts(icut);
    
    sim_im = emccd_detect(fluxmap, exptime, gain, full_well_serial, full_well,...
        dark_rate, cic_noise, read_noise, bias,...
        quantum_efficiency, cr_rate, pixel_pitch, apply_smear);
    
    
    
    % threshold and photon count
    threshold = cut * read_noise;
    PCimage = zeros(Npixacross);
    PCimage(sim_im>threshold) = 1;
    
    pcnum = Npixacross^2 - nnz(PCimage)
%     rmeas(icut) = 
    figure, imagesc(sim_im,[0,2*gain*flux*quantum_efficiency*exptime]); colorbar
    
end
 