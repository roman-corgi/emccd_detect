%% Basic Detector Simulator -- LOCAM example
close all;
clear;
clc;

%% Input flux map in units of electrons per pixel per second
% the user is expected to provide the input flux map in detector-sized pixels
Nwid = 40; 
NpixRow = Nwid;
NpixCol = Nwid;
photonInputFluxMap = 4000* peaks(Nwid).^2; % assume this is the incident photon flux map in ph/pix/sec

%% Usual detector parameters 
readNoise   = 120;   % e-/pix/frame  -- this is the amplifier noise in the downselected EMCCD
darkCurrent = 0.005; % e-/pix/s
CIC         = 0.02;  % e-/pix/frame
frameTime   = 0.001; % seconds (i.e. a frame rate of 1 kHz per requirements for LOCAM)
FWC_serial  = 90000; % e- full well capacity of the serial and gain registers - per requirements for LOCAM
FWC_image   = 50000; % e- full well capacity of the image area - per requirements for LOCAM
QE          = 0.85;  % at 550 nm, per e2v spec -- actual camera might do better
CRrate = 0; % set cosmic ray rate to zero until we improve the CR simulation; 5 hits/cm^2/s  is typical for well shielded regions in L2

%% Example call to the detector simulator  
EMgain  = 2;     % EM gain is normally much higher but set low here to see the read noise effect
t_exp   = 0.020; % seconds -- this is the total exposure time -- this should  be changed to what is needed (for tiptilt t_exp = frameTime = 0.001)
nFrames = ceil( t_exp / frameTime ); % how many frames we desire

tic;
frameStack = zeros(NpixRow, NpixCol, nFrames);
for iFrame = 1:nFrames
    frameStack(:,:,iFrame) = EMCCDdetect(photonInputFluxMap, readNoise, darkCurrent, CIC, CRrate, frameTime, EMgain, FWC_image, FWC_serial, QE); %#ok<*SAGROW>
end
tt = toc;
fprintf('basic:  simulating %u (%u x %u) images took %4.1f seconds (%3.2f sec/image).\n', iFrame, Nwid, Nwid, tt, tt/iFrame);

% show a single image -- note that all these images are done with bias not set, so bias is 0 by default
% that is equivalent to saying the bias subtraction has already been performed in the image processing
% in a bias-subtracted image it is expected that (because of read noise) there would be negative counts
figure;
imagesc(frameStack(:,:,1)); axis square; colorbar; colormap gray;
title(['single bias-subtracted frame (in e-) with gain = ',num2str(EMgain)]);

% total integrated image
coAddedImage = sum(frameStack, 3);
figure;
imagesc(coAddedImage); axis square; colorbar; colormap gray;
title(['co-added bias-subtracted image (in e-) in t_exp = ',num2str(t_exp),' sec']);

%% Auto tile the plots
iMon = 1; figureDims = 550*[1.0,1.0]; 
scrSize = get(0, 'MonitorPositions'); nr = round(scrSize(iMon,4)/figureDims(1)); nc = round(scrSize(iMon,3)/figureDims(2));
autoArrangeFigures(nr, nc, iMon);
