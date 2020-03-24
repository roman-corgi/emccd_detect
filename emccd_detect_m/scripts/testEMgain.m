clear; close all; clc; format compact;
addpath('../emccd_detect_m');
addpath('../emccd_detect_m/util');
jMon = 2; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

 
nem = 20;
emgains = logspace(1,4,nem);
fprintf('EM gain random generator tests\n\n1) Analog Mode Test:\n');
figure
for nin = 1:3
    for iem = 1: nem
        emgain = emgains(iem);
        
        Ntry = 10000;
        randEMno = zeros(1, Ntry);
        for irand=1:Ntry
            randEMno(irand) =  rand_em_gain(nin, emgain);
        end
        
        analogOut(nin,iem) = mean(randEMno)/emgain/nin;
    end
    meanout(nin) = mean(analogOut(nin, :));
    fprintf('For  n = %u,  <x> = %4.3f\n',nin,  meanout(nin));
    ph = semilogx(emgains, analogOut(nin,:)','.-'); hold on;
    set(ph(1), 'displayname',['Nin=',num2str(nin),'  ',num2str(meanout(nin),'%4.3f')])
end
grid;
legend;
xlabel('EM Gain')
ylabel('Mean analog counts / EM gain')


fprintf('\n\n2) Photon Counting Mode Test:\n');
% check photon counting
read_noise = 100;
nin = 1;
nthr_pts = 5;
pc_thresh = linspace(200, 900, nthr_pts);
nrnd = 1000;
emgain = 6000;
Npix  = 100;
zeroFrame = zeros(Npix);
for ithr = 1:nthr_pts
    
    % Threshold and photon count
    nthr(ithr) = pc_thresh(ithr) / read_noise ;    %#ok<*SAGROW>
    randEMno = zeroFrame;
    for irnd = 1: Npix^2
        randEMno(irnd) =  rand_em_gain(nin, emgain);
    end
    analogfr = reshape(randEMno, Npix, Npix);
    bright_PC = zeroFrame;
    bright_PC(analogfr > pc_thresh(ithr)) = 1;
 
    n_obs(ithr) = nnz(bright_PC) / Npix^2;
    
    eps_thr(ithr) = exp(-pc_thresh(ithr)/emgain);
    
    fprintf('For n_thr = %3.2f n_obs = %3.2f, eps_thr = %3.2f, n / eps = %3.3f\n',nthr(ithr), n_obs(ithr), eps_thr(ithr), n_obs(ithr)/eps_thr(ithr))
    figure, imagesc(analogfr); axis square; colorbar; 
    
 
        

end



autoArrangeFigures(nr, nc, iMon); return;


return

% For  n = 1,  <x> = 0.905
% For  n = 2,  <x> = 0.954
% For  n = 3,  <x> = 0.949
% For  n = 4,  <x> = 0.950
% For  n = 5,  <x> = 0.951
% For  n = 6,  <x> = 0.949
% For  n = 7,  <x> = 0.949
% For  n = 8,  <x> = 0.949
% For  n = 9,  <x> = 0.949
% For  n = 10,  <x> = 0.949