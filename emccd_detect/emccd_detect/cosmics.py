# -*- coding: utf-8 -*-
"""Generate cosmic hits."""

import numpy as np
from scipy.stats import landau
from scipy.special import erf, digamma
from scipy.signal import fftconvolve

# NOTE:  We do not simulate cosmic hits in the non-imaging area directly (only effects from spillover).  
# We could implement that, but it adds very little of value for what is desired from these simulations (namely, the image area), though 
# it could potentially affect the bias read off from prescan if lots of cosmics present there.  Would have to know serial clocking time interval as an input.

# interaction probability:  P_int = 1 - e^(-thickness/lambda_I); lambda_I = 9.6cm for tungsten, so 10mm of thickness gives Pint ~ 0.1.  So do binomial 
# with p=0.1.  Then for multiplicity of secondaries, use n_sec = Poisson(mu) with mu=1-5 (enough to bring average e- due to cosmics down to 1300e-?).
# Deposited e- charge: N_e = LogNormal() or Poisson (i.e., so skew for tail) with mean equal to half of the Landau loc used for the primary rays.

# cosmic rays coming in from front-side of back-illuminated (or front-illuminated) CCD:  
# same effect basically for incidence from either side b/c free-field region effect about the same.  

def rot_gauss_spot(x, y, A, x0, y0, sx, sy, theta):
    return  A*np.e**(-(np.cos(theta)*(x-x0)-np.sin(theta)*(y-y0))**2/(2*sx**2) - 
                     (np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0))**2/(2*sy**2))

def cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch, zff=8e-6, 
                loc=1590, scale=550, oversample_factor=10, sh_thickness=0.01, 
                mfp = 0.096, secondary_mu = 1, sec_mean_e_dep=None):
    """Generate cosmic hits.

    This function does not return the values of the cosmics; instead it returns
    the electron map which occurs as a result of the photelectric effect when
    the cosmics strike the detector. This allows the user to ignore the
    physical properties of the cosmics and focus only on their effect on the
    detector. 

    This function in general assumes primary cosmic rays (direct hits to EMCCD) 
    as well as secondary events (rays that are slow down enough in shielded 
    housing to interact and create secondary events via scattered photons or 
    neutrons).  Secondary rays are not simulated if sh_thickness is set to 0.

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
    sh_thickness : float
        Thickness of shielding (in meters) around EMCCD.  If no sheilding, use 0 (in which case the parameters below 
        are irrelevant).
        Defaults to 10mm, the thickness of the tungsten-copper alloy (HD17) used for Roman.
    mfp : float
        Mean free path (in meters) of cosmic rays in shielded housing around EMCCD.  Defaults to 9.6cm for the
        tungsten used for Roman.
    secondary_mu : float
        Poisson mean number of secondary particles created from a single galactic cosmic ray.  Defaults to 1.  
    sec_mean_e_dep : float
        Poisson mean number of electrons deposited for secondary cosmic rays.  If None, half of loc is used.


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
        # some of these might have been stopped by the shielding, made secondaries
        # probability of interaction in shielding:
        p_int = 1-np.e**(-sh_thickness/mfp) # about 0.1 for Roman
        secondary = np.random.binomial(n=1, p=p_int, size=hits_per_frame)
        num_secondary_cr = np.where(secondary==0)[0].size
        sec_numbers = np.random.poisson(lam=secondary_mu, size=num_secondary_cr)
        num_sec = np.sum(sec_numbers)
        total_non_sec = hits_per_frame - num_secondary_cr
        total_number = total_non_sec + num_sec
        hit_row = np.random.uniform(low=0, high=nr-1, size=total_number)
        hit_col = np.random.uniform(low=0, high=nc-1, size=total_number)

        # Describe each hit as a Gaussian centered at (hit_row, hit_col) with a
        # Gaussian distribution of comsic head size because of random walk through field-free section.
        cr_sigma = zff/pixel_pitch # in pixels 
        hit_sigma = np.abs(np.random.normal(loc=0, scale=cr_sigma, size=total_number))
        # angle of incidence on detector, theta
        theta = np.random.uniform(low=0, high=np.pi/2, size=total_number)
        # azimuthal orientation of cosmic ray relative to positive x (col) axis
        phi = np.random.uniform(low=0, high=np.pi, size=total_number)
        stretched_sigma = hit_sigma/np.cos(theta) # in pixels, radius projected onto detector
        sigma_x = np.abs(stretched_sigma*np.cos(phi)) 
        sigma_y = np.abs(stretched_sigma*np.sin(phi)) 
        # number of electrons delivered to sensor follows Landau distribution for primary events
        total_prim_e = landau.rvs(loc=loc, scale=scale, size=total_non_sec)
        # assume roughly half the energy of peak location of the Landau distribution for the mean 
        # and sigma of the distribution for the secondaries.  Poisson used here for its tail, typical for these events.
        if sec_mean_e_dep is None:
            sec_mean_e_dep = loc/2
        total_sec_e = np.random.poisson(sec_mean_e_dep, size=num_sec)
        
        # Plot histogram of total_sec_e
        if False:
            plt.figure()
            plt.hist(total_sec_e, bins=50, edgecolor='black')
            plt.xlabel('Electrons (e-)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Secondary Cosmic Ray Electrons')
            plt.show()
        
        # amplitude in Gaussian (regardless of its orientation)
        # If cosmic ray near edge of image area, so be it.  Energy amount still deposited to the 
        # detector (into shielded area), but a smaller amount delivered to pixels, which is what happens in for loop 
        # below.
        total_e = np.append(total_prim_e, total_sec_e)
        amplitudes = total_e/(2*np.pi*sigma_x*sigma_y)
        # treating x and y separately as 1D Gaussians separately, 
        # what factor multiplied by simga is needed so that value 
        # at this radius is 0.1 (basically 0, where Gaussian dies out).
        # But this takes a long time for simulation.  Instead, factor of 3 is fairly sufficient.
        #sigma_factor = np.sqrt(np.abs(np.log(amplitudes/0.1)/(2*np.min(np.stack([sigma_x,sigma_y]), axis=0)**2)))
        sigma_factor = 3 
        # have to truncate somewhere, and this should include most of the 
        # significant values of the Gaussian. 
        hit_rad = sigma_factor*hit_sigma 

        # Create hits
        for i in range(total_number):
            # Get pixels where cosmic lands
            min_row = max(np.floor(hit_row[i] - hit_rad[i]).astype(int), 0)
            max_row = min(np.ceil(hit_row[i] + hit_rad[i]).astype(int), nr-1)
            min_col = max(np.floor(hit_col[i] - hit_rad[i]).astype(int), 0)
            max_col = min(np.ceil(hit_col[i] + hit_rad[i]).astype(int), nc-1)

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


def sat_tails(serial_frame, full_well_serial, tail_length):
    """Simulate tails created by serial register saturation.

    This is most prevalent in cosmic hits.

    Parameters
    ----------
    serial_frame : array_like
        Serial register frame (e-).
    full_well_serial : float
        Serial (gain) register full well capacity (e-).
    tail_length : int
        Desired length of tail of cosmic ray.
    
    From https://doi.org/10.1117/1.JATIS.9.1.016003 :
    tail length of about 40 expected from surface traps in gain register for 
    high gains (> 1000).  For smaller gains have roughly proportionally smaller
    tail lengths (e.g., gain of 1000/40 = 25 may have an average tail length of
    40/40 = 1).  However, tails are always cut off whenever the spillover level 
    becomes a fractional amount of charge (0.1 e-), so high tail values for low
    gain wouldn't make too much difference in most normal cases, but better to 
    use a smaller tail length for smaller gains. 

    Note that if this is a sub-frame, wrap-around for spillover will happen, 
    whereas for a full frame, the wrap-around goes into the prescan and 
    overscan regions.

    """
    serial_frame = serial_frame.astype(float) 
    # analytic solution for scalar needed to get tail_length terms adding to 1 for 1/(2a) + 1/(3a) + ... 
    scalar = H(tail_length) - 1
    overflow = 0.
    overflow_i = 0.
    just_overflowed = False
    for i, pix in enumerate(serial_frame):
        # serial_frame[i] += _set_tail_val(overflow, overflow_i, i)
        serial_frame[i] += overflow 
        serial_frame[i-1] -= overflow # overflow is 0 when i=0

        if serial_frame[i] > full_well_serial:
            overflow = serial_frame[i] - full_well_serial
            overflow_i = i
            just_overflowed = True
        else: # tail from charge traps
            overflow = 0 
            if just_overflowed:
                tail_vals = np.array([])
                val = 2 # initiate, something bigger than 0.1.  (We clip fractional e- values during readout.)
                n = 1
                j = i # column to draw from and spread out over j+1 onward
                spread_val = np.sum(serial_frame[j:j+1])
                while val >= 0.1 and np.sum(tail_vals) < spread_val:
                    n += 1
                    if scalar <= 0: # happens when tail_length <= 1
                        val = spread_val
                    else:
                        val = spread_val / (scalar*n) 
                    tail_vals = np.append(tail_vals, val)
                if j+len(tail_vals) > len(serial_frame):
                    end_ind = len(serial_frame)
                else:
                    end_ind = j+len(tail_vals)
                # if this truncates tail_vals at the end of the last row, so be it. Just means that excess 
                # charge is either cleared out or shows up in the next frame (depending on the type of EMCCD modeled).
                serial_frame[j : end_ind] += tail_vals[:end_ind - (j)] 
                serial_frame[j:j+1] -= spread_val
            just_overflowed = False

    return serial_frame


def _set_tail_val_old(overflow, overflow_i, i):
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


def H(s):
    """ Harmonic number, used for solving Sum[1/(a*n), {n,2,tail_length}] for the scalar a.  
    If s is complex the result becomes complex. """
    return digamma(s + 1) + np.euler_gamma

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    t= time.time()
    cosmic_frame = cosmic_hits(np.zeros((1024, 1024)), cr_rate=5, frametime=100, pixel_pitch=13e-6, sh_thickness=0.01, zff=8e-6)
    print("It took ", time.time() - t, 'seconds')
    full_well_serial = 90000

    row = np.ones(100)
    row[2] = full_well_serial * 2

    tail_row = sat_tails(row, full_well_serial, 40)

    plt.figure()
    plt.plot(tail_row)
    plt.show()