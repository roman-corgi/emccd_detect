% Analog gain test
clear; close all; clc; format compact;
addpath('../');
addpath('../util');
jMon = 2; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');


npts = 20;
emgains = logspace(1, 4, npts);
fprintf('EM gain random generator tests\n\n1) Analog Mode Test:\n');
figure;
for n_in = 1:3
    for iem = 1:npts
        emgain = emgains(iem);
        
        ntry = 10000;
        rand_em_no = zeros(1, ntry);
        for irand = 1:ntry
            rand_em_no(irand) = rand_em_gain(n_in, emgain);
        end
        analog_out(n_in, iem) = mean(rand_em_no)/emgain/n_in;
    end
    meanout(n_in) = mean(analog_out(n_in, :));
    fprintf('For  n = %u,  <x> = %4.3f\n',n_in,  meanout(n_in));
    ph = semilogx(emgains, analog_out(n_in,:)','.-'); hold on;
    set(ph(1), 'displayname',['Nin=',num2str(n_in),'  ',num2str(meanout(n_in),'%4.3f')])
end
grid;
legend;
xlabel('EM Gain')
ylabel('Mean analog counts / EM gain')


fprintf('\n\n2) Photon Counting Mode Test:\n');
% Check photon counting
read_noise = 100;
n_in = 1;
nthr_pts = 5;
pc_thresh = linspace(200, 900, nthr_pts);
nrnd = 1000;
emgain = 6000;
npix  = 100;
zero_frame = zeros(npix);
for i = 1:nthr_pts
    % Threshold and photon count
    nthr(i) = pc_thresh(i) / read_noise;
    rand_em_no = zero_frame;
    for irnd = 1: npix^2
        rand_em_no(irnd) =  rand_em_gain(n_in, emgain);
    end
    analogfr = reshape(rand_em_no, npix, npix);
    bright_ = zero_frame;
    bright_PC(analogfr > pc_thresh(i)) = 1;
 
    n_obs(i) = nnz(bright_PC) / npix^2;
    
    eps_thr(i) = exp(-pc_thresh(i)/emgain);
    
    fprintf('For n_thr = %3.2f n_obs = %3.2f, eps_thr = %3.2f, n / eps = %3.3f\n',...
            nthr(i), n_obs(i), eps_thr(i), n_obs(i)/eps_thr(i))
    figure;
    imagesc(analogfr);
    axis square;
    colorbar; 
end

autoArrangeFigures;
