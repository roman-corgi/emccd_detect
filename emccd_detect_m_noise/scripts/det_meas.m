
hv  = [22 30:39];
emG = [ 1 1.495 1.797 2.31 3.264 5 8.581 16.99 43.6 152.5 731.4];
fwc = [ 100000, 50000, nan, nan, nan, 50000, nan, 45000, nan, 40000,40000];
shotSlp = [0.5 0.51 nan nan nan 0.55 nan 0.59 nan 0.68 0.72];

hv0 = 22:41; hv; 
emG0 = 1:5000;

%fitG = 10.^(2.^(hv0/2.5)/2.^(hv0(1)/2.5)/39);			
fitG = 10.^(2.^(hv0/2.5)/2.^(22/2.5)/39);			

%hv as function of g:
%hv = -(210*log(2) + 5*log(log(10))...
%    - 5*log(76451918253118239*log(g)))/(2*log(2));

fitFWC  = 6e4*1.5.^(-hv0/2)/1.5.^(-hv0(1)/2)+40000;
fitFWCg = (6917529027641081856000*(3/2).^(337164395852765312/6243314768165359 ...
- (11258999068426240*log(76451918253118240.*log(emG0)))./6243314768165359))/...
    1332894850849759 + 40000;


fitShot = 0.5 +2.^(hv0/2.3)/2.^(hv0(1)/2.3)/750;
fitShotg = (4398046511104*2.^((225179981368524800*...
log(76451918253118240.*log(emG0)))./143596239667803257 - ...
6743287917055306240/143596239667803257))/2498839895518397625 + 1/2;
hv1= hv; hv1(isnan(fwc))=[];
fwc1= fwc; fwc1(isnan(fwc))=[];
fitFWC2 = interp1(hv1, fwc1, hv0,'linear');

fgNum =30;
figure(fgNum), clf,semilogy(hv0, fitG,'ro-', 'linewidth', 1.5)
hold on,semilogy(hv, emG,'b*-', 'linewidth', 1.5), grid on
xlabel('HV'), ylabel('EM Gain'), title('EM Gain vs HV')
legend('fitted','mears''d', 'location', 'north')
print('-dpng', 'hv_vs_em_gain')

figure(fgNum+1), clf,plot(hv0, fitFWC,'ro-', 'linewidth', 1.5)
hold on, plot(hv, fwc,'b*-','linewidth', 1.5), grid on
hold on, plot(hv0, fitFWC2,'gx--','linewidth', 1.5), grid on
legend('fitted (formula)','mears''d', 'fitted2 (interp)','location', 'north')
xlabel('HV'), ylabel('FWC em'), title('FWC vs HV')
print('-dpng', 'hv_vs_fwc_em')

figure(fgNum+2), clf,plot(hv0, fitShot,'ro-', 'linewidth', 1.5)
hold on, plot(hv, shotSlp,'b*-','linewidth', 1.5), grid on
xlabel('HV'), ylabel('Shot slope'),title('Shot Slope vs HV')
legend('fitted','mears''d', 'location', 'north')
print('-dpng', 'hv_vs_shot_slope')

figure(fgNum+3), clf,plot(emG0, fitShotg)
hold on, plot(emG,shotSlp,'b*-','linewidth',1.5),grid on
print('-dpng','shot_slope_vs_em_gain')

figure(fgNum+4), clf,plot(emG0, fitFWCg)
hold on, plot(emG,fwc,'b*-','linewidth',1.5),grid on
print('-dpng','FWC_vs_em_gain')