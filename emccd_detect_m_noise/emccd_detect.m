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
%here are the new variates:
serial_frame = deviations.*enhfactor+serial_frame*em_gain;

%throw away data but ensure no negative #s of electrons:
%serial_frame(serial_frame < 0) = 0;

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
