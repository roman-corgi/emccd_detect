function sim_im = emccd_detect(fluxmap, exptime, gain, full_well_serial,...
                               full_well, dark_rate, cic_noise, read_noise,...
                               bias, quantum_efficiency, cr_rate, pixel_pitch,...
                               apply_smear) 
%EMCCD_DETECT Create an EMCCD-detected image for a given flux map.
%
% NOTES
% The flux map must be in units of photons/pix/s. Read noise is in electrons
% and is the amplifier read noise and not the effective read noise after the
% application of EM gain. Dark current must be supplied in units of e-/pix/s,
% and CIC is the clock induced charge in units of e-/pix/frame.
%
% B. Nemati and S. Miller - UAH - 18-Jan-2019
[frame_h, frame_w] = size(fluxmap);

% detector parameters
pars.CRrate      = cr_rate;
pars.pixelPitch  = pixel_pitch; % distance between pixel centers (m)
pars.pixelRadius = 3; % radius of pixels affected by a single cosmic hit (pixels)
pars.FWCim       = full_well; % full well capacity (image plane) 
pars.FWCgr       = full_well_serial; % full well capacity (gain register)
pars.matrixh     = frame_h;
pars.matrixw     = frame_w;

fixed_pattern = zeros(frame_h, frame_w);  % This will be modeled later

% Dark current is specified in e-/pix/s
meanExpectedDark = dark_rate * exptime;

% Mean expected electrons after inegrating in frameTime
meanExpectedElectrons = fluxmap * exptime * quantum_efficiency + meanExpectedDark + cic_noise;

% Electrons actualized at the pixels
expectedElectrons = poissrnd(meanExpectedElectrons);

if cr_rate ~= 0
    % Cosmic hits on image area
    [expectedElectrons, props] = cosmic_hits(expectedElectrons, pars, exptime);
end

% Electrons capped at full well capacity of imaging area
expectedElectrons(expectedElectrons > pars.FWCim) = pars.FWCim;

% Go through EM register
em_frame = zeros(frame_h, frame_w);
indnz = find(expectedElectrons);

for ii = 1:length(indnz)
    ie = indnz(ii);
    em_frame(ie) = rand_em_gain(expectedElectrons(ie), gain);
end

% if cr_rate ~= 0
%     % Tails from cosmic hits
%     em_frame = cosmic_tails(em_frame, pars, props);
% end

% Cap at full well capacity of gain register
em_frame(em_frame > pars.FWCgr) = pars.FWCgr;

readNoiseMap = read_noise * randn(frame_h, frame_w);

outMatrix = em_frame + readNoiseMap + fixed_pattern + bias;

sim_im = outMatrix;

return
