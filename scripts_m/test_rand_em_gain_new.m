% Test rand_em_gain_new
clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');

OnesMtx = ones(100);
EMgain = 1000;

% Check photon counting
N = 1;
NinMtx = OnesMtx * N;

out_old = rand_em_gain_w(NinMtx, EMgain);
out_new = rand_em_gain_new(NinMtx, EMgain);

thresh_array = 500:500:2000;
for thresh = thresh_array
    pc_out = zeros(size(out_new));
    pc_out(out_new > thresh) = 1;

    figure;
    imagesc(pc_out); colormap('gray');
    title(sprintf('Thresh : %d   Npix : %d', thresh, sum(pc_out(:))));

    figure;
    histbn(out_new);
    xline(thresh, 'r');
end

autoArrangeFigures(3, 4, 2)