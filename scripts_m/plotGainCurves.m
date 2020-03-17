clear;  format compact; close all; clc;
restoredefaultpath;
addpath('util');
jMon = 1; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');


gain = 100

nvec = 1:12;
xvec = 0:10:2000;
for in = 1:length(nvec)
    n = nvec(in);
    emgn(in, :)=EMgainpdf(xvec, n,  gain);
%     pois(in, :)=poisspdf(xvec,   
    
end

figure
plot(xvec/gain, emgn)
ylabel('probability density')
xlabel('output counts/gain')
legend('1','2','3','4','5','6','7','8','9','10','11')