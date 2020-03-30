function out = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch, max_val)
%Generate cosmic hits.
% 
%   Parameters
%   ----------
%   image_frame : array_like
%       Image area frame (e-).
%   cr_rate : float
%       Cosmic ray rate (hits/cm^2/s).
%   frametime : float
%       Frame time (s).
%   pixel_pitch : float
%       Distance between pixel centers (m).
%   max_val : float
%       Maximum value of cosmic hit (e-).
% 
%   Returns
%   -------
%   image_frame : array_like
%       Image area frame with cosmics added (e-).
% 
%   S Miller - UAH - 16-Jan-2019

% Find number of hits/frame
[nr, nc] = size(image_frame);
framesize = (nr*pixel_pitch * nc*pixel_pitch) / 10^-4;  % cm^2
hits_per_second = cr_rate * framesize;
hits_per_frame = round(hits_per_second * frametime);

% Generate hit locations
% Describe each hit as a gaussian centered at (hit_row, hit_col) and having
% an radius of hit_rad chosen between cr_min_radius and cr_max_radius
cr_min_radius = 0;
cr_max_radius = 2;
hit_row = rand(1, hits_per_frame) * nr;
hit_col = rand(1, hits_per_frame) * nc;
hit_rad = rand(1, hits_per_frame) * (cr_max_radius - cr_min_radius) + cr_min_radius;

% Create hits
for i = 1:hits_per_frame
    % Get pixels where cosmic lands
    min_row = max(floor(hit_row(i) - hit_rad(i)), 1);
    max_row = min(ceil(hit_row(i) + hit_rad(i)), nr);
    min_col = max(floor(hit_col(i) - hit_rad(i)), 1);
    max_col = min(ceil(hit_col(i) + hit_rad(i)), nc);
    [cols, rows] = meshgrid(min_col:max_col, min_row:max_row);

    % Create gaussian
    sigma = 0.5;
    a = 1 / (sqrt(2*pi) * sigma);
    b = 2 * sigma^2;
    cosm_section = a .* exp(-((rows-hit_row(i)).^2 + (cols-hit_col(i)).^2) / b);

    % Scale by maximum value
    cosm_section = cosm_section / max(cosm_section(:)) * max_val;

    % Add cosmic to frame
    image_frame(min_row:max_row, min_col:max_col) = image_frame(min_row:max_row, min_col:max_col) + cosm_section;
end

out = image_frame;
end
