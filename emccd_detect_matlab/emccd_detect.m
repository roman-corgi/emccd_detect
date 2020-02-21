function [y, meta] = emccd_detect(fluxMap, readNoise, darkCurrent, CIC, CRrate, frameTime, EMgain, FWCim, FWCgr, QE, fixedPattern, bias)
%  emccd_detect(fluxMap, readNoise, darkCurrent, CIC, CRrate, frameTime, EMgain, FWCim, FWCgr, QE, fixedPattern, bias)
%
% Create an EMCCD-detected image corresponding to an input flux map. The flux map must be in units of
% photons per pixel per second. Read noise is in electrons and is the amplifier read noise and not
% the effective read noise after the application of EM gain. Dark current must be supplied in units
% of electrons per pixel per second, and CIC is the clock induced charge in units of e-/pix/frame. 
%
% B. Nemati and S. Miller - UAH - 18-Jan-2019

[nr, nc] = size(fluxMap);

if ~exist('FWCim','var') || isempty(FWCim)
    FWCim = 50000;
end
if ~exist('FWCgr','var') || isempty(FWCgr)
    FWCgr = 150000;
end
    

% detector parameters
pars.CRrate      = CRrate;
pars.pixelPitch  = 13 * 10^(-6); % distance between pixel centers (m)
pars.pixelRadius = 3; % radius of pixels affected by a single cosmic hit (pixels)
pars.FWCim       = FWCim; % full well capacity (image plane) 
pars.FWCgr       = FWCgr; % full well capacity (gain register)
pars.matrixh     = nr;
pars.matrixw     = nc;

zmtx = zeros(nr, nc);
if ~exist('QE','var') || isempty(QE)
    QE = 1.0;
end
if ~exist('fixedPattern','var') || isempty(fixedPattern)
    fixedPattern = zmtx;
end
if ~exist('bias','var') || isempty(bias)
    bias = 0.;
end
% dark current is specified in e-/pix/s
meanExpectedDark = darkCurrent * frameTime;

% mean expected electrons after inegrating in frameTime
meanExpectedElectrons = fluxMap * frameTime * QE + meanExpectedDark + CIC;

% electrons actualized at the pixels
expectedElectrons = poissrnd(meanExpectedElectrons);

if CRrate ~= 0
    % cosmic hits on image area
    [expectedElectrons, props] = cosmicHits(expectedElectrons, pars, frameTime);
end

% electrons capped at full well capacity of imaging area
expectedElectrons(expectedElectrons > pars.FWCim) = pars.FWCim;

% go through EM register
emFrame = zmtx;
indnz = find(expectedElectrons);

for ii = 1:length(indnz)
    ie = indnz(ii);
    emFrame(ie) = randEMGain(expectedElectrons(ie), EMgain);
end

if CRrate ~= 0
    % tails from cosmic hits
    emFrame = cosmicTails(emFrame, pars, props);
end

% cap at full well capacity of gain register
emFrame(emFrame > pars.FWCgr) = pars.FWCgr;

% readNoise
readNoiseMap = readNoise * randn(nr, nc);

outMatrix = emFrame + readNoiseMap + fixedPattern + bias;

y = outMatrix;

return
