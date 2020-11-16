# -*- coding: utf-8 -*-
"""Script for profiling emccd_detect."""
from __future__ import absolute_import, division, print_function

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from emccd_detect.emccd_detect import EMCCDDetect

here = os.path.abspath(os.path.dirname(__file__))

# Input fluxmap
fits_name = 'sci_frame.fits'
fits_path = Path(here, 'data', fits_name)
fluxmap = fits.getdata(fits_path).astype(float)  # Input fluxmap (photons/pix/s)

# Put fluxmap in 1024x1024 image section
image = np.zeros((1024, 1024)).astype(float)
image[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap

# Simulation inputs
frametime = 100.  # Frame time (s)
em_gain = 5000.  # CCD EM gain (e-/photon)
full_well_image = 60000.  # Image area full well capacity (e-)
full_well_serial = 100000.  # Serial (gain) register full well capacity (e-)
dark_current = 0.0028  # Dark current rate (e-/pix/s)
cic = 0.01  # Clock induced charge (e-/pix/frame)
read_noise = 100.  # Read noise (e-/pix/frame)
bias = 0.  # Bias offset (e-)
qe = 0.9  # Quantum efficiency
cr_rate = 0.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
pixel_pitch = 13e-6  # Distance between pixel centers (m)
shot_noise_on = True  # Apply shot noise

# Initialize class
emccd = EMCCDDetect(
    meta_path=Path(here, 'data', 'metadata.yaml'),
    em_gain=em_gain,
    full_well_image=full_well_image,
    full_well_serial=full_well_serial,
    dark_current=dark_current,
    cic=cic,
    read_noise=read_noise,
    bias=bias,
    qe=qe,
    cr_rate=cr_rate,
    pixel_pitch=pixel_pitch,
    shot_noise_on=shot_noise_on
)

config = Config()
config.trace_filter = GlobbingFilter(
    include=[
        'emccd_detect.*'
    ]
)

# Make a dot file
output_file = Path(here, 'callgraph_emccd_detect')
graphviz = GraphvizOutput(output_type='dot',
                          output_file=str(output_file) + '.dot')
with PyCallGraph(output=graphviz, config=config):
    emccd.sim_sub_frame(fluxmap, frametime)
