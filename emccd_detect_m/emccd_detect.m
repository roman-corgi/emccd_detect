function sim_im = emccd_detect(fluxmap, frametime, em_gain, full_well_image,...
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

% Mean electrons after inegrating over exptime
mean_e = fluxmap * frametime * qe;

% Mean shot noise after integrating over exptime
mean_dark = dark_current * frametime;
shot_noise = mean_dark + cic;

% Electrons actualized at the pixels
if shot_noise_on
    image_frame = poissrnd(mean_e + shot_noise);
else
    image_frame = poissrnd(shot_noise, size(mean_e));
    image_frame = image_frame + mean_e;
end

image_frame = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch,...
                          full_well_image);


% Cap electrons at full well capacity of imaging area
image_frame(image_frame > full_well_image) = full_well_image;

% Go through EM register
post_gain_frame = rand_em_gain(image_frame, em_gain);

% if cr_rate ~= 0
%     % Tails from cosmic hits
%     em_frame = cosmic_tails(em_frame, pars, props);
% end

% Cap at full well capacity of gain register
post_gain_frame(post_gain_frame > full_well_serial) = full_well_serial;

% Apply fixed pattern
fixed_pattern = zeros(size(fluxmap));  % This will be modeled later
image_frame = image_frame + fixed_pattern;

% Read noise
read_noise_map = read_noise * randn(size(image_frame));

post_gain_frame = post_gain_frame + read_noise_map + bias;
sim_im = post_gain_frame;
return
