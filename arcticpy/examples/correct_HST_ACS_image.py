""" 
Correct CTI in an image from the Hubble Space Telescope (HST) Advanced Camera 
for Surveys (ACS) instrument.

It takes a while to correct a full image, so a small patch far from the readout 
register (where CTI has the most effect) is used for this example.
"""

import arcticpy as ac
import os
from autoconf import conf
import matplotlib.pyplot as plt
import numpy as np

# Path to this file
path = os.path.dirname(os.path.realpath(__file__))

# Set up some configuration options for the automatic fits dataset loading
conf.instance = conf.Config(config_path=f"{path}/config")

# Load the HST ACS dataset
path += "/acs"
name = "jc0a01h8q_raw"
frame = ac.acs.FrameACS.from_fits(file_path=f"{path}/{name}.fits", quadrant_letter="A")

# Extract an example patch of a few rows and columns, offset far from readout
row_start, row_end, column_start, column_end = -70, -40, -205, -190
row_offset = len(frame) + row_start
frame = frame[row_start:row_end, column_start:column_end]
frame.mask = frame.mask[row_start:row_end, column_start:column_end]

# Plot the initial image
plt.figure()
im = plt.imshow(X=frame[1:], aspect="equal", vmin=2300, vmax=2800)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/{name}_input.png", dpi=400)
plt.close()
print(f"Saved {path}/{name}_input.png")

# Set CCD, ROE, and trap parameters for HST ACS at this Julian date
traps, ccd, roe = ac.model_for_HST_ACS(
    date=2400000.5 + frame.exposure_info.modified_julian_date
)

# Remove CTI
image_cti_removed = ac.remove_cti(
    image=frame,
    iterations=3,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
    parallel_offset=row_offset,
    parallel_express=2,
)

# Plot the corrected image
plt.figure()
im = plt.imshow(X=image_cti_removed[1:], aspect="equal", vmin=2300, vmax=2800)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/{name}_corrected.png", dpi=400)
plt.close()
print(f"Saved {path}/{name}_corrected.png")
