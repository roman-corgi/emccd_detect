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
    # Get fluxmap
    fits_path = Path(here, 'data', 'sci_fluxmap.fits')
    fluxmap = fits.getdata(fits_path).astype(float)  # (photons/pix/s)
    # Put fluxmap in 1024x1024 image section
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    full_fluxmap[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap
    # Specify frametime
    frametime = 100  # s
    nonlin_sample = Path(here, 'emccd_detect', 'util', 'nonlin_sample.csv')


    # For the simplest possible use of EMCCDDetect, use its defaults
    emccd = EMCCDDetect()

    # If you are using Python<=3.9, you can also apply CTI to the frame using the 
    # pure-Python, older version of arcticpy that was included with this emccd_detect package
    # in its own separate folder.  
    # If you have Python>3.9, this older version will not work. If you have the 
    # current vesion of articpy (which is a wrapper for C++ code), you can have 
    # any version of Python.  See (<https://github.com/jkeger/arctic>) for installation of the 
    # newest version of arcticpy.

    # Below is how you could apply CTI.
    # For the old version of arcticpy, see (<https://github.com/jkeger/arcticpy/tree/row_wise/arcticpy>) for
    # details on the optional inputs to add_cti() so that you can specify
    # something meaningful for the EMCCD you have in mind. For the newer version, 
    # see (<https://github.com/jkeger/arctic>).
    # (using "try" so that this script still runs in the case that arcticpy
    # is not viable.  In that case, running this method update_cti()
    # will not work.)
    try:
        emccd.update_cti()
    except:
        pass
    # Simulate only the fluxmap
    sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)
    # Simulate the full frame (surround the full fluxmap with prescan, etc.)
    sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
    # to turn off CTI application to future frames made with the same class
    # instance (If arcticpy not viable, trying to run unset_cti() will not
    # work):
    try:
        emccd.unset_cti()
    except:
        pass

    np.random.seed(123)
    # For more control, each of the following parameters can be specified.
    # Custom metadata path, if the user wants to use a different metadata file
    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml')
    # Note that the defaults for full_well_serial and eperdn are specified in
    # the metadata file.  Nonlinearity during readout can be applied.  See 
    # nonlinearity.py for details.
    emccd = EMCCDDetect(
        em_gain=5000.,
        full_well_image=78000.,  # e-
        full_well_serial=105000.,  # e-
        dark_current=0.00031,  # e-/pix/s
        cic=0.016,  # e-/pix/frame
        read_noise=110.,  # e-/pix/frame
        bias=1500.,  # e-
        qe=0.9,
        cr_rate=0 ,#5.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=8.2,
        nbits=14,
        numel_gain_register=604,
        meta_path=meta_path,
        nonlin_path=nonlin_sample
    )

    # To retain the same output for multiple runs using the same class 
    # instance, one can specify the same seed before each instance of creating 
    # a frame
    # Simulate only the fluxmap
    np.random.seed(123)
    sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)
    # Simulate the full frame (surround the full fluxmap with prescan, etc.)
    np.random.seed(123)
    sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)


    # The class also has some convenience functions to help with inspecting the
    # simulated frame
    # Get a gain divided, bias subtracted frame in units of e-
    frame_e = emccd.get_e_frame(sim_full_frame)
    # Return just the 1024x1024 region of a full frame
    image = emccd.slice_fluxmap(sim_full_frame)
    # Return the prescan region of a full frame
    prescan = emccd.slice_prescan(sim_full_frame)


    # For legacy purposes, the class can also be called from a function wrapper
    sim_old_style = emccd_detect(fluxmap, frametime, em_gain=5000.)

    ########### example with arcitcpy-specific inputs 
    # There are 2 inputs for update_cti() which are specific to how emccd_detect implements
    # arcticpy:  serial=True turns on serial CTI, and parallel=True turns on parallel CTI. 
    # Both are True by default.
    fluxmap2 = np.zeros((70,70)) # toy fluxmap
    fluxmap2[30:40,30:40] = 200
    sim_sub_frame = emccd.sim_sub_frame(fluxmap2, frametime)
    imagesc(sim_sub_frame, 'Output Sub Frame Before CTI')
    try: 
        emccd.update_cti(parallel_traps=[ap.TrapInstantCapture(density=1,release_timescale=1)], serial=False)
    except:
        pass
    sim_sub_frame = emccd.sim_sub_frame(fluxmap2, frametime)
    imagesc(sim_sub_frame, 'Output Sub Frame After CTI')

    # Plot images
    imagesc(full_fluxmap, 'Input Fluxmap')
    imagesc(sim_full_frame, 'Output Full Frame')
    plt.show()
