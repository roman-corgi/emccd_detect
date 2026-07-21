import numpy as np
import pytest
from pathlib import Path
from emccd_detect.emccd_detect import EMCCDDetect
from astropy.io import fits

def test_hot_pix():
    '''
    This test makes sure that the hot pixel simulated at [33,33] in the 1024x1024 image area, which is 
    row 13+33 = 46 and column 1088+33 = 1121.
    is identifiable.
    '''
    hot_pixel = Path(__file__).resolve().parents[1] / 'emccd_detect' / 'emccd_detect' / 'util' / 'hot_pixel_sample.fits'  # contains all ones except for 150 at [33,33]
    detector = EMCCDDetect(
        em_gain=1.,
        full_well_image=78000.,  # e-
        full_well_serial=105000.,  # e-
        dark_current=0.31,  # e-/pix/s; Suppose we are at a higher temperature
        cic=0.016,  # e-/pix/frame
        read_noise=110.,  # e-/pix/frame
        bias=1500.,  # e- 
        qe=0.9,
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=8.2,
        nbits=14,
        numel_gain_register=604,
        meta_path=None,
        nonlin_path=None, 
        flat_path=None,
        hot_pixel_path=hot_pixel
    )

    # NOTE too high of a dark current multiplication factor in the map leads to smearing when EM gain is high, as expected.  
    # In practice, though, high gain is not usually paired with high exposure times.  Factor is 150 for this bad pixel, whereas it may be 5 to 15 realistically?  
    # More frames averaged together would be needed to detect it.
    out_array1 = detector.sim_full_frame(np.zeros((1024,1024)), 100) # 100 seconds, long exposure time to accentuate the bad pixel
    max_row, max_col = np.where(out_array1==out_array1.max())
    assert (max_row, max_col) == (46, 1121)



if __name__ == '__main__':
    test_hot_pix()