function [outMatrix, props] = cosmicHits(frame, pars, frameTime)
% Inputs:
%           frame            : input frame
%           pars             : detector parameters
%           frameTime        : time of a single frame (seconds)
%
% Output:
%           outmatrix        : input frame + cosmic hits
% 
% S. Miller - 16-Jan-2019

CRrate      = pars.CRrate;
pixelPitch  = pars.pixelPitch;
pixelRadius = pars.pixelRadius;
FWCim       = pars.FWCim;
matrixh     = pars.matrixh;
matrixw     = pars.matrixw;

% find size of frame
framesize = matrixh*pixelPitch * matrixw*pixelPitch; % m^2

% find number of hits/frame
hitsPerSecond = CRrate * framesize/10^(-4);
hitsPerFrame = round(hitsPerSecond * frameTime);

% generate hits
hitsx = rand(1, hitsPerFrame) * matrixh;
hitsy = rand(1, hitsPerFrame) * matrixw;

% describe each hit as a gaussian centered at (hitsx,hitsy), landing on pixel (h,k),
% and having energies described by r (since radius is proportional to energy)  
xx = 1:matrixw;
yy = (matrixh:-1:1)';
h = max(round(hitsx),1);      % x (col)
k = matrixh+1 - round(hitsy); % y (row)
r = round(rand(1,length(hitsx))*2*(pixelRadius-1)+1);

% create hits
for i=1:length(hitsx)
    % set constants for gaussian
    sigma = r(i)/3.75;
    a = 1/(sqrt(2*pi)*sigma);
    b = 2*sigma^2;
    cutoff = 0.03*a;
    
    rows = max(k(i)-r(i),1):min(k(i)+r(i),matrixh);
    cols = max(h(i)-r(i),1):min(h(i)+r(i),matrixw);
    cosmSection = a .* exp(-((xx(cols)-hitsx(i)).^2 + (yy(rows)-hitsy(i)).^2) / b);
    cosmSection(cosmSection<=cutoff) = 0;
    % normalize and scale by FWCim
    cosmSection = cosmSection/max(cosmSection(:)) * FWCim;
    
    frame(rows, cols) = frame(rows, cols) + cosmSection;
end

props.h = h;
props.k = k;
props.r = r;

outMatrix = frame;
end
