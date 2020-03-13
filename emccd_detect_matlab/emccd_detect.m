function sim_im = emccd_detect(fluxmap, exptime, em_gain, full_well_image,...
                               full_well_serial, dark_current, cic, read_noise,...
                               bias, qe, cr_rate, pixel_pitch, shot_noise_on) 
%EMCCD_DETECT Create an EMCCD-detected image for a given flux map.
%
% NOTES
% The flux map must be in units of photons/pix/s. Read noise is in electrons
% and is the amplifier read noise and not the effective read noise after the
% application of EM gain. Dark current must be supplied in units of e-/pix/s,
% and CIC is the clock induced charge in units of e-/pix/frame.
%
% B Nemati and S Miller - UAH - 18-Jan-2019
fixed_pattern = zeros(size(fluxmap));  % This will be modeled later

% Mean electrons after inegrating over exptime
mean_e = fluxmap * exptime * qe;

% Mean shot noise after integrating over exptime
mean_dark = dark_current * exptime;
shot_noise = mean_dark + cic;

% Electrons actualized at the pixels
if shot_noise_on
    image_frame = poissrnd(mean_e + shot_noise);
else
    image_frame = poissrnd(shot_noise, size(mean_e));
    image_frame = image_frame + mean_e;
end

% if cr_rate ~= 0
%     % Cosmic hits on image area
%     [image_frame, props] = cosmic_hits(image_frame, pars, exptime);
% end

% Cap electrons at full well capacity of imaging area
image_frame(image_frame > full_well_image) = full_well_image;

% Go through EM register
serial_frame = zeros(size(image_frame));
indnz = find(image_frame);

for i = 1:length(indnz)
    ie = indnz(i);
    serial_frame(ie) = rand_em_gain(image_frame(ie), em_gain);
end

% if cr_rate ~= 0
%     % Tails from cosmic hits
%     em_frame = cosmic_tails(em_frame, pars, props);
% end

% Cap at full well capacity of gain register
serial_frame(serial_frame > full_well_serial) = full_well_serial;

% Apply fixed pattern
image_frame = image_frame + fixed_pattern;

% Read noise
read_noise_map = read_noise * randn(size(image_frame));

serial_frame = serial_frame + read_noise_map + bias;
sim_im = serial_frame;
return
