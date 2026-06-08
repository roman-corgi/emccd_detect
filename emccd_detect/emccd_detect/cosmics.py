# -*- coding: utf-8 -*-
"""Generate cosmic hits."""

import numpy as np
from scipy.stats import landau
from scipy.special import erf
from scipy.signal import fftconvolve

def rot_gauss_spot(x, y, A, x0, y0, sx, sy, theta):
    return  A*np.e**(-(np.cos(theta)*(x-x0)-np.sin(theta)*(y-y0))**2/(2*sx**2) - 
                     (np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0))**2/(2*sy**2))

def cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch, zff=8e-6, 
                loc=1590, scale=550, oversample_factor=10):
    """Generate cosmic hits.

    This function does not return the values of the cosmics; instead it returns
    the electron map which occurs as a result of the photelectric effect when
    the cosmics strike the detector. This allows the user to ignore the
    physical properties of the cosmics and focus only on their effect on the
    detector. 

    Parameters
    ----------
    image_frame : array_like
        Image area frame (e-).
    cr_rate : float
        Cosmic ray rate (hits/cm^2/s).
    frametime : float
        Frame time (s).
    pixel_pitch : float
        Distance between pixel centers (m).
    zff : float
        Free-field thickness of CCD (m). This is the perpendicular distance 
        that the cosmic ray travels before hitting the detector, which affects
        the size of the cosmic ray head.  Default is 8e-6m (for Roman CGI EMCCD).
    loc : float
        Location parameter for Landau distribution of total electrons delivered to sensor by cosmic ray. 
        Default is 1590e- (expected for Roman CGI EMCCD).  Together with scale 
        below gives a rough mean of 2360e- and most probable value (MPV) of 
        1360e-, which are the values expected for Roman CGI EMCCD at L2.  
    scale : float
        Scale parameter for Landau distribution of total electrons delivered to sensor by cosmic ray. 
        Default is 550e- (for Roman CGI EMCCD).
    oversample_factor : int
        Factor of oversampling of cosmic Gaussian over which to bin-sum to get 
        pixel values.  Default is 10.

    Returns
    -------
    image_frame : array_like
        Image area frame with cosmics added (e-).

    """
    if cr_rate > 0:
        # Find number of hits/frame
        nr, nc = image_frame.shape
        framesize = (nr*pixel_pitch * nc*pixel_pitch) / 10**-4  # cm^2
        hits_per_second = cr_rate * framesize
        hits_per_frame = int(round(hits_per_second * frametime))

        # Generate hit locations
        hit_row = np.random.uniform(low=0, high=nr-1, size=hits_per_frame)
        hit_col = np.random.uniform(low=0, high=nc-1, size=hits_per_frame)
        # Describe each hit as a Gaussian centered at (hit_row, hit_col) with a
        # Gaussian distribution of comsic head size because of random walk through field-free section.
        cr_sigma = zff/pixel_pitch # in pixels 
        hit_sigma = np.abs(np.random.normal(loc=0, scale=cr_sigma, size=hits_per_frame))
        hit_rad = 2*hit_sigma # have to truncate somewhere, and this should include most of the significant values of the Gaussian.
        # angle of incidence on detector, theta
        theta = np.random.uniform(low=0, high=np.pi/2, size=hits_per_frame)
        # azimuthal orientation of cosmic ray relative to positive x (col) axis
        phi = np.random.uniform(low=0, high=np.pi, size=hits_per_frame)
        stretched_sigma = hit_sigma/np.cos(theta) # in pixels, radius projected onto detector
        sigma_x = np.abs(stretched_sigma*np.cos(phi)) 
        sigma_y = np.abs(stretched_sigma*np.sin(phi)) 
        # number of electrons delivered to sensor follows Landau distribution 
        total_e = landau.rvs(loc=loc, scale=scale, size=hits_per_frame)
        # amplitude in Gaussian (regardless of its orientation)
        # If cosmic ray near edge of image area, so be it.  Energy amount still deposited to the 
        # detector (into shielded area), but a smaller amount delivered to pixels, which is what happens in for loop 
        # below.
        amplitudes = total_e/(2*np.pi*sigma_x*sigma_y) 

        # Create hits
        for i in range(hits_per_frame):
            # Get pixels where cosmic lands
            min_row = max(np.floor(hit_row[i] - hit_rad[i]).astype(int), 0)
            max_row = min(np.ceil(hit_row[i] + hit_rad[i]).astype(int), nr-1)
            min_col = max(np.floor(hit_col[i] - hit_rad[i]).astype(int), 0)
            max_col = min(np.ceil(hit_col[i] + hit_rad[i]).astype(int), nc-1)
            cols, rows = np.meshgrid(np.arange(min_col, max_col+1),
                                    np.arange(min_row, max_row+1))
            # oversampled cols, rows
            o_cols, o_rows = np.meshgrid(np.arange(min_col, max_col+1, 1/oversample_factor),
                                    np.arange(min_row, max_row+1, 1/oversample_factor)) 
            # Create elliptic Gaussian for cosmic ray head
            cosm_section = rot_gauss_spot(o_cols, o_rows, A=amplitudes[i]/(oversample_factor**2), 
                                          x0=hit_col[i], y0=hit_row[i], sx=sigma_x[i], sy=sigma_y[i], theta=phi[i])

            # Downsample: Sum oversampled pixels to form physical CCD pixel values
            num_rows = max_row - min_row + 1
            num_cols = max_col - min_col + 1
            cosm_section_downsampled = cosm_section.reshape(
                num_rows, oversample_factor,
                num_cols, oversample_factor
            ).sum(axis=(1, 3))

            # Add cosmic to frame
            image_frame[min_row:max_row+1, min_col:max_col+1] += cosm_section_downsampled

    return image_frame


def sat_tails(serial_frame, full_well_serial):
    """Simulate tails created by serial register saturation.

    This is most prevalent in cosmic hits.

    Parameters
    ----------
    serial_frame : array_like
        Serial register frame (e-).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).

    """
    overflow = 0.
    overflow_i = 0.
    for i, pix in enumerate(serial_frame):
        serial_frame[i] += _set_tail_val(overflow, overflow_i, i)

        if serial_frame[i] > full_well_serial:
            overflow = serial_frame[i] - full_well_serial
            overflow_i = i

    return serial_frame


def _set_tail_val(overflow, overflow_i, i):
    # Some of excess above FWC is captured by traps in gain register, some lost to substrate
    relative_i = i+1 - overflow_i
    # Fraction of excess above FWC spread across downstream pixels is Sum(1/2**n) 
    # for n=1 to some potentially high number, so the sum never exceeds 1.
    # Perhaps more realistic is Sum(1/3**n) or some other series, but this sum 
    # is good b/c it gives worst case scenario basically, with the most charge
    # retained and distributed.  But we do cut off at some point for surface traps 
    # (when tail_val < 1000). 
    try: 
        tail_val = overflow * 1 / (2**(relative_i-1))
    except:
        tail_val = 0 # relative_i is too large in this case, resulting in overflow error, so tail value is negligible and set to 0.
    if tail_val < 1000:
        tail_val = 0

    return tail_val


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cosmic_frame = cosmic_hits(np.zeros((1024, 1024)), cr_rate=5, frametime=1, pixel_pitch=13e-6)
    full_well_serial = 90000

    row = np.ones(100)
    row[2] = full_well_serial * 2

    tail_row = sat_tails(row, full_well_serial)

    plt.figure()
    plt.plot(tail_row)
    plt.show()