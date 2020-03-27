function out = emccd_detect(fluxmap, frametime, em_gain, full_well_image,...
                            full_well_serial, dark_current, cic, read_noise,...
                            bias, qe, cr_rate, pixel_pitch, shot_noise_on) 
%EMCCD_DETECT Create an EMCCD-detected image for a given flux map.
%
% Notes:
% The flux map must be in units of photons/pix/s. Read noise is in electrons
% and is the amplifier read noise and not the effective read noise after the
% application of EM gain. Dark current must be supplied in units of e-/pix/s,
% and CIC is the clock induced charge in units of e-/pix/frame.
%
% B Nemati and S Miller - UAH - 18-Jan-2019
image_frame = image_area(fluxmap, frametime, full_well_image, dark_current,...
                         cic, qe, cr_rate, pixel_pitch, shot_noise_on);

serial_frame = serial_register(image_frame, em_gain, full_well_serial,...
                               read_noise, bias);

out = serial_frame;
end

function image_frame = image_area(fluxmap, frametime, full_well_image,...
                                  dark_current, cic, qe, cr_rate, pixel_pitch,...
                                  shot_noise_on)
% Simulate detector image area.

% Mean electrons after inegrating over exptime
mean_e = fluxmap * frametime * qe;

% Mean shot noise after integrating over exptime
mean_dark = dark_current * frametime;
mean_shot = mean_dark + cic;

% Electrons actualized at the pixels
if shot_noise_on
    image_frame = poissrnd(mean_e + mean_shot);
else
    shot_noise_map = poissrnd(mean_shot, size(mean_e));
    image_frame = shot_noise_map + mean_e;
end

% Simulate cosmic hits on image area
image_frame = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch,...
                          full_well_image);

% Cap electrons at full well capacity of imaging area
image_frame(image_frame > full_well_image) = full_well_image;
end

function serial_frame = serial_register(image_frame, em_gain, full_well_serial,...
                                        read_noise, bias)
% Simulate detector serial (gain) register.

% Flatten image area row by row to simulate readout to serial register
serial_frame = reshape(image_frame.', 1, []);

% Apply EM gain
serial_frame = rand_em_gain(serial_frame, em_gain);

% Cap at full well capacity of gain register
serial_frame(serial_frame > full_well_serial) = full_well_serial;

% Apply fixed pattern
serial_frame = serial_frame + make_fixed_pattern(serial_frame);

% Apply read noise and bias
serial_frame = serial_frame + make_read_noise(serial_frame, read_noise) + bias;

% Reshape for viewing
serial_frame = reshape(serial_frame, size(image_frame, 2), size(image_frame, 1)).';
end

function out = make_fixed_pattern(serial_frame)
% Simulate EMCCD fixed pattern.
out = zeros(size(serial_frame));  % This will be modeled later
end

function out = make_read_noise(serial_frame, read_noise)
% Simulate EMCCD read noise.
out = read_noise * randn(size(serial_frame));
end
