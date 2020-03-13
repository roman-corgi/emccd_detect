clear; close all; clc; format compact;
jMon = 2; fsz = 500*[1,1.4];
scrSize = get(0, 'MonitorPositions'); [nMon,~]=size(scrSize); iMon = min(jMon, nMon);
nr = round(scrSize(iMon,4)/fsz(1)); nc = round(scrSize(iMon,3)/fsz(2)); clear('jMon', 'nMon','fsz');

% 
% EMgain = 1000
% Nin = 1
% for irnd = 1, 10000
%     out(irnd) = randemgain2( Nin, EMgain );
% end
% 
% 
% 
% return

nem = 30;
emgains = logspace(1,4,nem);

for nin = 1:10
    for iem = 1: nem
        emgain = emgains(iem);
        
        Ntry = 10000;
        rem = zeros(1, Ntry);
        for irand=1:Ntry
            rem(irand) =  rand_em_gain(nin, emgain);
        end
        
        out(nin,iem) = mean(rem)/emgain/nin;
    end
    meanout(nin) = mean(out(nin, :));
    fprintf('For  n = %u,  <x> = %4.3f\n',nin,  meanout(nin));
end

out

% out = rand_em_gain( Nin, EMgain )
figure
semilogx(emgains, out','.-')
grid

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