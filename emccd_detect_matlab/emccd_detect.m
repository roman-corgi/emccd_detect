function sim_im = emccd_detect(fluxmap, cr_rate, frametime, em_gain, bias, qe,...
                               fwc_im, fwc_gr, dark_current, cic, read_noise) 
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
pars.pixelPitch  = 13 * 10^(-6); % distance between pixel centers (m)
pars.pixelRadius = 3; % radius of pixels affected by a single cosmic hit (pixels)
pars.FWCim       = fwc_im; % full well capacity (image plane) 
pars.FWCgr       = fwc_gr; % full well capacity (gain register)
pars.matrixh     = frame_h;
pars.matrixw     = frame_w;

fixed_pattern = zeros(frame_h, frame_w);  % This will be modeled later
if ~exist('QE','var') || isempty(qe)
    qe = 1.0;
end
if ~exist('bias','var') || isempty(bias)
    bias = 0.;
end
% Dark current is specified in e-/pix/s
meanExpectedDark = dark_current * frametime;

% Mean expected electrons after inegrating in frameTime
meanExpectedElectrons = fluxmap * frametime * qe + meanExpectedDark + cic;

% Electrons actualized at the pixels
expectedElectrons = poissrnd(meanExpectedElectrons);

if cr_rate ~= 0
    % Cosmic hits on image area
    [expectedElectrons, props] = cosmic_hits(expectedElectrons, pars, frametime);
end

% Electrons capped at full well capacity of imaging area
expectedElectrons(expectedElectrons > pars.FWCim) = pars.FWCim;

% Go through EM register
em_frame = zeros(frame_h, frame_w);
indnz = find(expectedElectrons);

for ii = 1:length(indnz)
    ie = indnz(ii);
    em_frame(ie) = rand_em_gain(expectedElectrons(ie), em_gain);
end

if cr_rate ~= 0
    % Tails from cosmic hits
    em_frame = cosmic_tails(em_frame, pars, props);
end

% Cap at full well capacity of gain register
em_frame(em_frame > pars.FWCgr) = pars.FWCgr;

readNoiseMap = read_noise * randn(frame_h, frame_w);

outMatrix = em_frame + readNoiseMap + fixed_pattern + bias;

sim_im = outMatrix;

return
