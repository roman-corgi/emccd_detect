% EMCCD Detector Simulation.
%
% S Miller and B Nemati - UAH - 21-Feb-2020
clear; close all; clc; format compact;
addpath('../');
addpath('../util');
jMon = 2; fsz = 450*[1,1.3];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');


% Input fluxmap
npix_across = 20;
flux = 0.07;  % photns/pix/s
fluxmap = flux * ones(npix_across);

% Simulation inputs
frametime = 1.;  % Frame time (s)
em_gain = 6000.;  % CCD EM gain (e-/photon)
full_well_image  = 60000.;  % Image area full well capacity (e-)
full_well_serial = 90000.;  % Serial (gain) register full well capacity (e-)
dark_current = 0.00028;  % Dark  current rate (e-/pix/s)
cic =  0.02;  % Clock induced charge (e-/pix/frame)
read_noise = 100.;  % Read noise (e-/pix/frame)
bias = 0.;  % Bias offset (e-)
qe = 1.;  % Quantum effiency
cr_rate = 0.;  % Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6;  % Distance between pixel centers (m)

zero_frame = zeros(size(fluxmap));
npts = 55;
pc_thresh = linspace(200, 1600, npts);
for i = 1:length(pc_thresh)
    % Threshold and photon count
    eps_thr(i) = exp(-pc_thresh(i) / em_gain);
    
    % Dark frame
    dark_frame = emccd_detect(zero_frame, frametime, em_gain, full_well_image,...
                              full_well_serial, dark_current, cic, read_noise,...
                              bias, qe, cr_rate, pixel_pitch, true);
    % Photon-count the dark frame
    dark_pc = zero_frame;
    dark_pc(dark_frame > pc_thresh(i)) = 1;
    % The raw photon-counted frame needs to be corrected for inefficiencies 
    % from thresholding and coincidence losses
    % Observed mean rate after photon counting 
    nobs_dk(i) = nnz(dark_pc) / npix_across^2;
    lambda_dk = -log(1-(nobs_dk(i)/eps_thr(i)));
    rtrue_dk = lambda_dk / frametime;


    % Bright frame
    bright_frame = emccd_detect(fluxmap, frametime, em_gain, full_well_image,...
                                full_well_serial, dark_current, cic, read_noise,...
                                bias, qe, cr_rate, pixel_pitch, true);
    % Photon-count the dark frame
    bright_pc = zero_frame;
    bright_pc(bright_frame > pc_thresh(i)) = 1;
    % The raw photon-counted frame needs to be corrected for inefficiencies 
    % from thresholding and coincidence losses
    % Observed mean rate after photon counting 
    nobs_br(i) = nnz(bright_pc) / npix_across^2;
    lambda_br = -log(1 - (nobs_br(i)/eps_thr(i)));
    rtrue_br = lambda_br / frametime;

    % Photo-electron rate
    r_phe(i) = rtrue_br - rtrue_dk;
end

%%
% Threshold efficincy for n=1 and n=2 EM probablity distributions
sigma_thr = pc_thresh / read_noise;
eps_th1 = @(x,g) exp(-x/g);
eps_th2 = @(x,g) (1+x/g).*exp(-x/g);
eps_th3 = @(x,g) (1+(x/g)+0.5*(x/g).^2).*exp(-x/g);
pdfEM   = @(x,g,n) x.^(n-1).*exp(-x/g)./(g^n*factorial(n-1));
pp1 = poisspdf(1,r_phe);
pp2 = poisspdf(2,r_phe);
pp3 = poisspdf(3,r_phe);
eth1 = eps_th1(sigma_thr*read_noise, em_gain);
eth2 = eps_th2(sigma_thr*read_noise, em_gain);
eth3 = eps_th3(sigma_thr*read_noise, em_gain);
overcountEst2 = (pp1.*eth1 + pp2.*eth2)             ./ ( (pp1+pp2)    .*eth1 );
overcountEst3 = (pp1.*eth1 + pp2.*eth2 + pp3.*eth3) ./ ( (pp1+pp2+pp3).*eth1 );


figure;
plot(sigma_thr, nobs_br/frametime, sigma_thr, r_phe,  sigma_thr, flux*ones(1, npts));
grid;
legend('Observed', 'Corrected', 'Actual');
xlabel('threshold factor');
ylabel('rates, e/pix/s');
title(['RN=',num2str(read_noise),' emG=',num2str(em_gain),' FWCs=',num2str(full_well_serial/1000),'k']);

figure;
plot(sigma_thr, eps_thr);
grid;
xlabel('threshold factor');
ylabel('threshold effeciency');
title('Assuming all pixels are 1 or 0 real ph-e''s');


figure;
plot(sigma_thr, overcountEst2);
grid;
xlabel('threshold factor');
ylabel('PC over-count factor');


figure;
plot(sigma_thr, nobs_br/frametime,'.-', sigma_thr, r_phe,'.-', sigma_thr, flux*ones(1, npts),...
     sigma_thr, r_phe./overcountEst2,'.-',  sigma_thr, r_phe./overcountEst3,'.-');
grid;
legend('Raw Phot Cnt', 'thr, CL corr', 'Actual', '+ovrcnt corr', '+n3 corr');
xlabel('threshold factor');
ylabel('rates, e/pix/s');
title(['RN=',num2str(read_noise),' emG=',num2str(em_gain),' FWCs=',num2str(full_well_serial/1000),'k']);

actualc = flux*ones(1, npts);

figure;
plot(sigma_thr, r_phe./actualc,'.-', sigma_thr, r_phe./overcountEst2./actualc,'.-',...
     sigma_thr, r_phe./overcountEst3./actualc,'.-', sigma_thr, ones(1, npts));
grid;
legend('thr, CL corr', '+ovrcnt corr', '+n3 corr');
xlabel('threshold factor');
ylabel('rate/actual');
title(['RN=',num2str(read_noise),' emG=',num2str(em_gain),' FWCs=',num2str(full_well_serial/1000),'k']);

autoArrangeFigures(nr, nc, iMon); 
