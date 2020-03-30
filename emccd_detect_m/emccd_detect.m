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

% Mean electrons after inegrating over frametime
mean_e_map = fluxmap * frametime * qe;

% Mean shot noise after integrating over frametime
mean_dark = dark_current * frametime;
mean_shot = mean_dark + cic;

% Actualize electrons at the pixels
if shot_noise_on
    image_frame = poissrnd(mean_e_map + mean_shot);
else
    image_frame = mean_e_map + poissrnd(mean_shot, size(mean_e_map));
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
serial_frame = rand_em_gain(serial_frame, em_gain);
% Cap at full well capacity of gain register
serial_frame(serial_frame > full_well_serial) = full_well_serial;

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
