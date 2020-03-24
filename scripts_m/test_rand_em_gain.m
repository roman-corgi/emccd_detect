% Test rand_em_gain
clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');
jMon = 1; 
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
fsz = 400*[1.2,1];
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

eps_th1 = @(x,g) exp(-x/g);
eps_th2 = @(x,g) (1+x/g).*exp(-x/g);
eps_th3 = @(x,g) (1+(x/g)+0.5*(x/g)^2).*exp(-x/g);


EMgain = 1000;

% Verify that rand_em_gain creates exactly the same array as
% rand_em_gain_old_w for n_in greater than 2
OnesMtx = ones(10);
Narray = 2:3;
fprintf('Sanity Check\n');
fprintf('------------\n');
for N = Narray
    NinMtx = OnesMtx * N;
    
    [old, new] = both_em_gain(NinMtx, EMgain);
    fprintf('N: %d  Arrays Equal: %d\n', N, isequal(old, new));
end
fprintf('\n');

% Check means
Narray = 1:4;
fprintf('\nCheck Means\n');
fprintf('-----------\n');
fprintf('%s   %s   %s\n', 'N', 'old_mean', 'new_mean');
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
thresh_array = 5:5:35;  % Percentage of EMgain
fprintf('\nCheck Threshold Efficiencies\n');
fprintf('----------------------------');
for N = Narray
    fprintf('\nN: %d       threshold efficiencies, %% \n', N);
    fprintf(' %7s  %8s %9s  %9s\n', 'thr/g(%)', 'expected', 'old gen', 'new gen'); 
    NinMtx = OnesMtx * N;
    i = 1;
    for percent = thresh_array
        thresh = percent/100 * EMgain;
        switch N
            case 1
                e_pc(i) = eps_th1(thresh, EMgain); %#ok<*SAGROW>
            case 2
                e_pc(i) = eps_th2(thresh, EMgain);
            case 3
                e_pc(i) = eps_th3(thresh, EMgain);
            otherwise
                error('not handled');
        end

        [old, new] = both_em_gain(NinMtx, EMgain);

        pc_old = zeros(size(old));
        pc_old(old > thresh) = 1;
        e_pc_old(i) = calc_efficiency(NinMtx, pc_old);

        pc_new = zeros(size(new));
        pc_new(new > thresh) = 1;
        e_pc_new(i) = calc_efficiency(NinMtx, pc_new);
        
        fprintf('%6d %10.2f %10.2f %10.2f\n', percent, e_pc(i)*100, e_pc_old(i)*100, e_pc_new(i)*100);
        i = i + 1;
    end
    figure;
    plot(thresh_array, e_pc); hold on;
    plot(thresh_array, e_pc_old);
    plot(thresh_array, e_pc_new);
    title([sprintf('N = %d\n', N), sprintf('PC Efficiencey')]);
    xlabel('threshold / gain  ,  %');
    ylabel('Efficiency');
    legend('expected', 'old gen', 'new gen');grid;
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

