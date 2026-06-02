# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect
try:
    import arcticpy as ap
except:
    pass
from emccd_detect.partial_CIC_MLE import EM_gain_fit_conv, EM_gain_fit_conv_rn, EM_gain_fit_LPG_W, EM_gain_fit_W


def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
    """Plot a scaled colormap."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect,
                   origin='lower')

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == '__main__':
    # Set up some inputs here
    here = os.path.abspath(os.path.dirname(__file__))
    from cal.util.gsw_process import Process
    from cal.util.read_metadata import Metadata as MetadataWrapper
    import cal

    libfile = cal.__path__[0]
    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(libfile, 'util', 'metadata.yaml')
    # no actual nonlinearity correction assumed here
    nonlin_path = Path(libfile, 'util', 'testdata', 'ut_nonlin_array_ones.txt')
    meta = MetadataWrapper(meta_path)
    image_rows, image_cols, r0c0 = meta._unpack_geom('image')
    
    np.random.seed(123)

    # full-well capacities
    fwc_em_e = 105000 #e-, for EM gain register
    fwc_pp_e = 90000 #e-, per pixel (before EM gain register)
    frametime = 1

    def read_in_files(directory, eperdn, bias_offset, gain, prescan=False):
        '''This function mainly subtracts the bias, divides by gain, and 
        converts from DN to e-.  See Process class methods for details 
        on the processing done for these specific EMCCD frames that are used
        for demonstration in this script. 
        '''
        proc = Process(bad_pix=np.zeros((meta.frame_rows,meta.frame_cols)), eperdn=eperdn,
                                fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                                bias_offset=bias_offset, em_gain=gain,
                                exptime=frametime, nonlin_path=nonlin_path,
                                meta_path=meta_path)
        framelist = []
        for file in os.listdir(directory):
            if file.endswith('fits'):
                f = os.path.join(directory, file)
                d = fits.getdata(f)
                _, _, _, _, f0, b0, _ = proc.L1_to_L2a(d)
                # could just skip L2a_to_L2b(), but I like having a hook for
                # combining b0 with a proc_dark.bad_pix that isn't all zeros
                f1, b1, _ = proc.L2a_to_L2b(f0, b0)
                # to undo the division by gain in L2a_to_L2b()
                ff = f1*gain
                # assign NaN values to any pixels marked as bad (i.e., due to cosmic rays)
                ff = np.ma.masked_array(ff, mask=b1.astype(bool))
                f1 = ff.astype(float).filled(np.nan)
                # assign NaN values to last row, which does not have any physical readout
                f1[-1] = np.nan

                if prescan:
                    f1 = proc.meta.slice_section(f1, 'prescan')
                else:
                    f1 = proc.meta.slice_section(f1, 'image')
                framelist.append(f1)
        st = np.stack(framelist)
        return st

    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml') 
    
    emccd = EMCCDDetect(
        em_gain=5000.,
        full_well_image=90000.,  # e-
        full_well_serial=105000.,  # e-
        dark_current=0.001,  # e-/pix/s
        cic=0.016,  # e-/pix/frame
        read_noise=110., # e-/pix/frame
        bias=1500.,  # e-
        qe=0.9,
        cr_rate=0, #5 # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=8.2,
        nbits=14,
        numel_gain_register=604,
        meta_path=meta_path,
        row_read_time=223.5e-6, # in seconds
        threshold=2.001e4, #1e8, #XXX default value 
        gain_CIC_Q=0, #0.001,
        #gain_CIC_specs={200:.01,204:.01,300:.01,400:.001}
    )


    # dark frame
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    # Specify frametime
    frametime = 1 # s
    sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
    directory = os.path.join(here, 'gain_stats_comparison')
    if not os.path.exists(directory):
        os.mkdir(directory)
    fits.writeto(os.path.join(directory, 'sim_dark.fits'), sim_full_frame, overwrite=True)

    frames = read_in_files(directory, eperdn=8.2, bias_offset=0, gain=5000)
    pass

    res1, chisquare_value1, pvalue1  = EM_gain_fit_conv(frames,.008, 5000,6000,110,0,cut=-800, tol=1e-10)
    print(res1, chisquare_value1, pvalue1)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([2.76514043e-02, 4.97152306e+03]), 6676631.432429293, 39.129370193111484, 1.0)
    # row index 500 only:
    # (True, array([1.67833954e-02, 4.99999998e+03]), 6444.748315364548, 0.7054126328932218, 1.0)

    # No partial CIC, lambda, g, sigma_rn, and mu (Fits 2 and 6 from paper)
    res2, chisquare_value2, pvalue2 = EM_gain_fit_conv_rn(frames,.008,5000,6000,110,0,lthresh=0,cut=-800, tol=1e-10)
    print(res2, chisquare_value2, pvalue2)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([1.90885768e-02, 4.86106525e+03, 1.15316038e+02, 5.31535511e+00]), 6666469.708461857, 6.25709059243572, 1.0)
    # row index 500 only:
    # (True, array([ 1.65201362e-02,  4.99996516e+03,  1.16149480e+02, -1.55274235e+01]), 6431.654075122935, 0.47416855117423884, 1.0)

    # With partial CIC, lambda, Q, and g (Fits 3 and 7 from paper)
    res3, chisquare_value3, pvalue3 = EM_gain_fit_LPG_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)
    print(res3, chisquare_value3, pvalue3)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([2.41292160e-02, 1.05875819e-03, 5.01924628e+03]), 6662394.908985481, 109.48038278397956, 1.0)
    # row index 500 only:
    # (True, array([1.67817595e-02, 0.00000000e+00, 4.99999999e+03]), 6444.7483025533, 0.7054144891891521, 1.0)

    # With partial CIC, lambda, Q, g, sigma_rn, and mu (Fits 4 and 8 from paper)
    res4, chisquare_value4, pvalue4 = EM_gain_fit_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)
    print(res4, chisquare_value4, pvalue4)
    # shown below: