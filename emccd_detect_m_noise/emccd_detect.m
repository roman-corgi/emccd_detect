function out = emccd_detect(fluxmap, frametime, em_gain, full_well_image,...
                            full_well_serial, dark_current, cic, read_noise,...
                            bias, qe, cr_rate, pixel_pitch, shot_noise_on) 
%Create an EMCCD-detected image for a given fluxmap.
% 
%   Parameters
%   ----------
%   fluxmap : array_like, float
%       Input fluxmap (photons/pix/s).
%   frametime : float
%       Frame time (s).
%   em_gain : float
%       CCD em_gain (e-/photon).
%   full_well_image : float
%       Image area full well capacity (e-).
%   full_well_serial : float
%       Serial (gain) register full well capacity (e-).
%   dark_current: float
%       Dark current rate (e-/pix/s).
%   cic : float
%       Clock induced charge (e-/pix/frame).
%   read_noise : float
%       Read noise (e-/pix/frame).
%   bias : float
%       Bias offset (e-).
%   qe : float
%       Quantum efficiency.
%   cr_rate : float
%       Cosmic ray rate (hits/cm^2/s).
%   pixel_pitch : float
%       Distance between pixel centers (m).
%   shot_noise_on : bool, optional
%       Apply shot noise. Defaults to True.
% 
%   Returns
%   -------
%   serial_frame : array_like, float
%       Detector output (e-).
% 
%   Notes
%   -----
%   Read noise is the amplifier read noise and not the effective read noise
%   after the application of EM gain.
% 
%   B Nemati and S Miller - UAH - 30-Mar-2020

image_frame = image_area(fluxmap, frametime, full_well_image, dark_current,...
                         cic, qe, cr_rate, pixel_pitch, shot_noise_on);

serial_frame = serial_register(image_frame, em_gain, full_well_serial,...
                               read_noise, bias);

out = serial_frame;
end

function image_frame = image_area(fluxmap, frametime, full_well_image,...
                                  dark_current, cic, qe, cr_rate, pixel_pitch,...
                                  shot_noise_on)
%Simulate detector image area.
% 
%   Parameters
%   ----------
%   fluxmap : array_like, float
%       Input fluxmap (photons/pix/s).
%   frametime : float
%       Frame time (s).
%   full_well_image : float
%       Image area full well capacity (e-).
%   dark_current: float
%       Dark current rate (e-/pix/s).
%   cic : float
%       Clock induced charge (e-/pix/frame).
%   qe : float
%       Quantum efficiency.
%   cr_rate : float
%       Cosmic ray rate (hits/cm^2/s).
%   pixel_pitch : float
%       Distance between pixel centers (m).
%   shot_noise_on : bool, optional
%       Apply shot noise. Defaults to True.
% 
%   Returns
%   -------
%   image_frame : array_like
%       Image area frame (e-).

% Mean photo-electrons after inegrating over frametime
mean_phe_map = fluxmap * frametime * qe;

% Mean expected rate after integrating over frametime
mean_dark = dark_current * frametime;
mean_noise = mean_dark + cic;

% Actualize electrons at the pixels
if shot_noise_on
    image_frame = poissrnd(mean_phe_map + mean_noise);
else
    image_frame = mean_phe_map + poissrnd(mean_noise, size(mean_phe_map));
end

% Simulate cosmic hits on image area
image_frame = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch,...
                          full_well_image);

% Cap at full well capacity of image area
image_frame(image_frame > full_well_image) = full_well_image;
end

function serial_frame = serial_register(image_frame, em_gain, full_well_serial,...
                                        read_noise, bias)
%Simulate detector serial (gain) register.
% 
%   Parameters
%   ----------
%   image_frame : array_like
%       Image area frame (e-).
%   em_gain : float
%       CCD em_gain (e-/photon).
%   full_well_serial : float
%       Serial (gain) register full well capacity (e-).
%   read_noise : float
%       Read noise (e-/pix/frame).
%   bias : float
%       Bias offset (e-).
% 
%   Returns
%   -------
%   serial_frame : array_like
%       Serial register frame (e-).

% Flatten image area row by row to simulate readout to serial register
serial_frame = reshape(image_frame.', 1, []);

% Apply EM gain
gainFrame = rand_em_gain(serial_frame, em_gain);

%fitted function for shot noise slope
%fitSlope= @(hv) 0.5 +2.^(hv/2.3)/2.^(22/2.3)/750;
%fitG =@(hv0) 10.^(2.^(hv0/2.5)/2.^(22/2.5)/39);
%invert the function to get gain(hv):
%hv = -(210*log(2) + 5*log(log(10))...
%    - 5*log(76451918253118239*log(g)))/(2*log(2));
fitSlope = @(g) (4398046511104*2.^((225179981368524800*...
log(76451918253118240.*log(g)))./143596239667803257 - ...
6743287917055306240/143596239667803257))/2498839895518397625 + 1/2;

%enhancement factor needed to account for shot noise slope different from
%.5
enh = @(x,slope) x.^(slope-.5) ;

%deviation from the mean, where the mean is n*gain
deviations = gainFrame - serial_frame*em_gain;
enhfactor = enh(gainFrame, fitSlope(em_gain));
serial_frame_slope = deviations.*enhfactor+serial_frame*em_gain;

%account for non-linearity 
serial_frame = serial_frame_slope+serial_frame_slope.*percent_NL(serial_frame_slope,em_gain)./100;
%serial_frame = gainFrame + gainFrame.*percent_NL(serial_frame,em_gain)./100;
%serial_frame = gainFrame;



%FWC as function of HV
%fitFWC  = 6e4*1.5.^(-hv0/2)/1.5.^(-hv0(1)/2)+40000;
%FWC as function of g
fitFWCg = @(g) (6917529027641081856000*(3/2).^(337164395852765312/6243314768165359 ...
- (11258999068426240*log(76451918253118240.*log(g)))./6243314768165359))/...
    1332894850849759 + 40000;

% Cap at full well capacity of gain register
serial_frame(serial_frame > fitFWCg(em_gain)) = fitFWCg(em_gain);
%serial_frame(serial_frame > full_well_serial) = full_well_serial;

% Apply fixed pattern, read noise, and bias
serial_frame = serial_frame + make_fixed_pattern(serial_frame);
serial_frame = serial_frame + make_read_noise(serial_frame, read_noise) + bias;

%throw away data but ensure no negative #s of electrons:
%serial_frame(serial_frame < 0) = 0;

% Reshape for viewing
serial_frame = reshape(serial_frame, size(image_frame, 2), size(image_frame, 1)).';
end

function out = make_fixed_pattern(serial_frame)
%Simulate EMCCD fixed pattern.
out = zeros(size(serial_frame));  % This will be modeled later
end

function out = make_read_noise(serial_frame, read_noise)
%Simulate EMCCD read noise.
out = read_noise * randn(size(serial_frame));
end

function out = percent_NL(signal,em_gain)
%hv as function of g:
%hv = -(210*log(2) + 5*log(log(10))...
%    - 5*log(76451918253118239*log(g)))/(2*log(2));
sig = 500:100:98000;
%imported data from Excel spreadsheet from Nathan Bush
percent_nl_30 = table2array(readtable('residual_non_linearity.xlsx','range','C5:C980'));
percent_nl_39 = table2array(readtable('residual_non_linearity.xlsx','range','L5:L980'));
Nl_30 = interp1(sig,percent_nl_30,signal,'pchip');
Nl_39 = interp1(sig,percent_nl_39,signal,'pchip');
m = (Nl_39 - Nl_30)/(39-30);
b = Nl_39 - m*39;
hv = -(210*log(2) + 5*log(log(10))- 5*log(76451918253118239.*log(em_gain)))./(2*log(2));
if em_gain < 1.0608  %corresponding to hv<22; hv 22 to 30 identical
    out = Nl_30;
end
if (em_gain <= 719.7664) && ( em_gain >= 1.0608) %between hv 30 and 39
    out = m*hv+b;
end
if em_gain > 719.7664  %hv above 39: use hv 39 
    out = Nl_39;
end
end