function sim_im = emccd_detect(fluxmap, cr_rate, frametime, em_gain, bias, qe,...
                               fwc_im, fwc_gr, dark_current, cic, read_noise) 
%EMCCD_DETECT Create an EMCCD-detected image corresponding to an input flux map.
%
% NOTES
% The flux map must be in units of photons/pix/s. Read noise is in electrons
% and is the amplifier read noise and not the effective read noise after the
% application of EM gain. Dark current must be supplied in units of e-/pix/s,
% and CIC is the clock induced charge in units of e-/pix/frame.
%
% B. Nemati and S. Miller - UAH - 18-Jan-2019
[nr, nc] = size(fluxmap);

if ~exist('FWCim','var') || isempty(fwc_im)
    fwc_im = 50000;
end
if ~exist('FWCgr','var') || isempty(fwc_gr)
    fwc_gr = 150000;
end
    

% detector parameters
pars.CRrate      = cr_rate;
pars.pixelPitch  = 13 * 10^(-6); % distance between pixel centers (m)
pars.pixelRadius = 3; % radius of pixels affected by a single cosmic hit (pixels)
pars.FWCim       = fwc_im; % full well capacity (image plane) 
pars.FWCgr       = fwc_gr; % full well capacity (gain register)
pars.matrixh     = nr;
pars.matrixw     = nc;

zmtx = zeros(nr, nc);
fixedPattern = zmtx;
if ~exist('QE','var') || isempty(qe)
    qe = 1.0;
end
if ~exist('bias','var') || isempty(bias)
    bias = 0.;
end
% dark current is specified in e-/pix/s
meanExpectedDark = dark_current * frametime;

% mean expected electrons after inegrating in frameTime
meanExpectedElectrons = fluxmap * frametime * qe + meanExpectedDark + cic;

% electrons actualized at the pixels
expectedElectrons = poissrnd(meanExpectedElectrons);

if cr_rate ~= 0
    % cosmic hits on image area
    [expectedElectrons, props] = cosmic_hits(expectedElectrons, pars, frametime);
end

% electrons capped at full well capacity of imaging area
expectedElectrons(expectedElectrons > pars.FWCim) = pars.FWCim;

% go through EM register
emFrame = zmtx;
indnz = find(expectedElectrons);

for ii = 1:length(indnz)
    ie = indnz(ii);
    emFrame(ie) = rand_em_gain(expectedElectrons(ie), em_gain);
end

if cr_rate ~= 0
    % tails from cosmic hits
    emFrame = cosmic_tails(emFrame, pars, props);
end

% cap at full well capacity of gain register
emFrame(emFrame > pars.FWCgr) = pars.FWCgr;

% readNoise
readNoiseMap = read_noise * randn(nr, nc);

outMatrix = emFrame + readNoiseMap + fixedPattern + bias;

sim_im = outMatrix;

return
