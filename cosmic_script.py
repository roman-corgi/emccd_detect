"""Script for testing cosmic functions."""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

from emccd_detect.cosmic_hits import cosmic_hits
from emccd_detect.util.imagesc import imagesc

framesize = 100
image_frame = np.zeros((framesize, framesize))
cr_rate = 5.  # Cosmic ray rate (5 for L2) (hits/cm^2/s)
frametime = 100.  # Frame time (s)
pixel_pitch = 13e-6  # Distance between pixel centers (m)
full_well_image = 60000.  # Image area full well capacity (e-)

hits_frame = cosmic_hits(image_frame, cr_rate, frametime, pixel_pitch,
                         full_well_image)

imagesc(hits_frame, 'Cosmic Hits')
plt.show()
