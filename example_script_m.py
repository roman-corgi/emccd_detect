# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect


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

# Set up some inputs here
here = os.path.abspath(os.path.dirname(__file__))
# Get fluxmap
fits_path_py = Path(here, 'data', 'sci_fluxmap.fits')
#fits_path_py = Path(here, 'data', 'ref_frame.fits')

def read_func(fits_path=fits_path_py,frametime=100, em_gain=5000.,
        full_well_image=60000.,  # e-
        #full_well_serial=100000.,  # e-
        status=1,
        dark_current=0.0028,  # e-/pix/s
        cic=0.02,  # e-/pix/frame
        read_noise=100.,  # e-/pix/frame
        bias=10000.,  # e-
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=1, #7.,
        nbits=64,#14,
        numel_gain_register=604,
        choice='legacy'
):

    fluxmap = fits.getdata(fits_path).astype(float)  # (photons/pix/s)
    # Put fluxmap in 1024x1024 image section
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    full_fluxmap[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap



    # For the simplest possible use of EMCCDDetect, use its defaults
#    emccd = EMCCDDetect()
    # Simulate only the fluxmap
#    sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)
    # Simulate the full frame (surround the full fluxmap with prescan, etc.)
#   sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)


    # For more control, each of the following parameters can be specified
    # Custom metadata path, if the user wants to use a different metadata file
    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml')
    # Note that the defaults for full_well_serial and eperdn are specified in
    # the metadata file
    if choice=='latest':
        emccd = EMCCDDetect(
            em_gain=em_gain,
            full_well_image=full_well_image,  # e-
            #full_well_serial=100000.,  # e-
            status=status,
            dark_current=dark_current,  # e-/pix/s
            cic=cic,  # e-/pix/frame
            read_noise=read_noise,  # e-/pix/frame
            bias=bias,  # e-
            qe=qe,
            cr_rate=cr_rate,  # hits/cm^2/s
            pixel_pitch=pixel_pitch,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=numel_gain_register,
            meta_path=meta_path
        )
        # Simulate only the fluxmap
        sim_sub_frame = emccd.sim_sub_frame(fluxmap, frametime)
        #np. set_printoptions(threshold=np. inf)
        #print(np.array(sim_sub_frame))
        return sim_sub_frame
        # Simulate the full frame (surround the full fluxmap with prescan, etc.)
        #sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)


    # The class also has some convenience functions to help with inspecting the
    # simulated frame
    # Get a gain divided, bias subtracted frame in units of e-
#    frame_e = emccd.get_e_frame(sim_full_frame)
    # Return just the 1024x1024 region of a full frame
#    image = emccd.slice_fluxmap(sim_full_frame)
    # Return the prescan region of a full frame
#    prescan = emccd.slice_prescan(sim_full_frame)

    if choice=='legacy':
        # For legacy purposes, the class can also be called from a functon wrapper
        sim_old_style = emccd_detect(fluxmap=fluxmap,
            frametime=frametime,
            em_gain=em_gain,
            full_well_image=full_well_image,
            #full_well_serial=90000.,
            status=status,
            dark_current=dark_current,
            cic=cic,
            read_noise=read_noise,
            bias=bias,
            qe=qe,
            cr_rate=cr_rate,
            pixel_pitch=pixel_pitch,
            shot_noise_on=None)
        #np. set_printoptions(threshold=np. inf)
        #print(np.array(sim_old_style))
        return sim_old_style

    # Plot images
    #imagesc(full_fluxmap, 'Input Fluxmap')
    #imagesc(sim_sub_frame, 'Output Sub Frame')
    #imagesc(sim_full_frame, 'Output Full Frame')
    #plt.show()
#print(read_func())
#imagesc(read_func(choice='latest'), 'Latest')
#imagesc(read_func(choice='legacy'),'Legacy')
#plt.show()
if __name__ == '__main__':

    imagesc(read_func(choice='latest'), 'Latest')
    imagesc(read_func(choice='legacy'),'Legacy')
    plt.show()
    #args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    #globals()[args[1]](*args[2:])
    #imagesc(np.array([[1,0],[1,1]]))
    #plt.show()