clear;  format compact; close all; clc;
restoredefaultpath;
addpath('./util');
jMon = 1; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');


avgCount = .5 %#ok<*NOPTS>
Npix = 2^10
EMgain = 1000

fprintf('Poisson distribution:\n');
fprintf('for avg count = %3.2f:\n', avgCount);
pdf = poisspdf([0, 1, 2, 3], avgCount);
fprintf('%u:\t%6.5f\n%u:\t%6.5f\n%u:\t%6.5f\n%u:\t%6.5f\n',0,pdf(1),1,pdf(2),2,pdf(3),3,pdf(4));
% generate a repeatable Poisson random matrix
rng(1);
imgArea = poissrnd( avgCount * ones(Npix), Npix, Npix);



tic
postGain = randemgain3( imgArea, EMgain );
toc


figure
imagesc(imgArea); colorbar; axis square;
title('Image Area Counts Before Gain');
figure
imagesc(postGain); colorbar; axis square;
title('Output After Gain')

autoArrangeFigures(nr, nc, iMon); return;
%-----------------STOP--------------------
