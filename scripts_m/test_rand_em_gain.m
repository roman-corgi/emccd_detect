% Test rand_em_gain
clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');
jMon = 1; 
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
fsz = 400*[1.2,1];
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

EMgain = 1000;

% Verify that rand_em_gaincreates exactly the same array as
% rand_em_gain_old_w for n_in greater than 2
OnesMtx = ones(10);
Narray = 2:3;
for N = Narray
    NinMtx = OnesMtx * N;
    avg_rate = sum(NinMtx(:)) / numel(NinMtx);
    
    [old, new] = both_em_gain(NinMtx, EMgain);
    fprintf('N: %d   Avg Rate: %.3f   Arrays Equal: %d\n', N, avg_rate, isequal(old, new));
end
fprintf('\n');

% Check means
Narray = 1:4;
fprintf('%s %10s %10s\n', 'N', 'old_mean', 'new_mean');
for N = Narray
    NinMtx = OnesMtx * N;
    
    [old, new] = both_em_gain(NinMtx, EMgain);
    old_vals = old(old > 0);
    new_vals = new(new > 0);
    
    old_mean = mean(old_vals(:));
    new_mean = mean(new_vals(:));
    fprintf('%d %10.3f %10.3f\n', N, old_mean, new_mean);
    
    figure;
    bb = 20;
    h = histogram(old_vals,'binWidth', bb, 'DisplayStyle', 'stairs', 'FaceColor', 'none'); hold on;
    histogram(new_vals, 'binWidth', bb, 'DisplayStyle', 'stairs', 'FaceColor', 'none');
    title([sprintf('N = %d\n', N), sprintf('Old Mean: %.f   New Mean: %.f', old_mean, new_mean)]);
    legend('old', 'new');grid;
end
fprintf('\n')

% Check threshold efficiency
Narray = 1:2;
thresh_array = 5:5:25;  % Percentage of EMgain
for N = Narray
    fprintf('N: %d\n', N);
    fprintf(' %7s %7s %10s %10s\n', 'thr/g(%)', 'e', 'old e', 'new e'); 
    NinMtx = OnesMtx * N;
    i = 1;
    for percent = thresh_array
        thresh = percent/100 * EMgain;
        e_pc(i) = exp(-thresh/EMgain);

        [old, new] = both_em_gain(NinMtx, EMgain);

        pc_old = zeros(size(old));
        pc_old(old > thresh) = 1;
        e_pc_old(i) = calc_efficiency(NinMtx, pc_old);

        pc_new = zeros(size(new));
        pc_new(new > thresh) = 1;
        e_pc_new(i) = calc_efficiency(NinMtx, pc_new);
        
        fprintf('%6d %10.3f %10.3f %10.3f\n', percent, e_pc(i), e_pc_old(i), e_pc_new(i));
        i = i + 1;
    end
    figure;
    plot(thresh_array, e_pc); hold on;
    plot(thresh_array, e_pc_old);
    plot(thresh_array, e_pc_new);
    title([sprintf('N = %d\n', N), sprintf('PC Efficiencey')]);
    xlabel('t/g (%)');
    ylabel('Efficiency');
    legend('e', 'old e', 'new e');grid;
end

autoArrangeFigures(nr, nc, iMon); return;
%-----------------STOP--------------------


function [old, new] = both_em_gain(NinMtx, EMgain)
% Call old and new versions of rand_em_gain with same random number seed
seed = 1;

rng(seed);
old = rand_em_gain_old_w(NinMtx, EMgain);
rng(seed);
new = rand_em_gain(NinMtx, EMgain);
end


function e_pc = calc_efficiency(NinMtx, pcMtx)
% Calculate the actual photon counting efficiency of an array
n_ones = length(find(NinMtx>=1));  % improve later
e_pc = sum(pcMtx(:)) / n_ones;
end

