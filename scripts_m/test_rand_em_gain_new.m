% Test rand_em_gain_new
clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');

OnesMtx = ones(100);
EMgain = 1000;

% Check for different Nin values
n_array = 1:4;
NinFigs = true;
for N = n_array
    NinMtx = OnesMtx * N;
    Nout = rand_em_gain_new(NinMtx, EMgain);

    if NinFigs
        figure;
        histbn(Nout);
        title(sprintf('Nin : %d', N));
    end
end

% Check photon counting
out = rand_em_gain_new(OnesMtx, EMgain);
thresh_array = 200:200:1000;
for thresh = thresh_array
    pc_out = zeros(size(out));
    pc_out(out > thresh) = 1;

    figure;
    imagesc(pc_out); colormap('gray');
    title(sprintf('Thresh : %d   Npix : %d', thresh, sum(pc_out(:))));

    figure;
    histbn(out);
    xline(thresh);
end

autoArrangeFigures(4,4)