import numpy as np
import pytest
from pathlib import Path
from emccd_detect.emccd_detect import EMCCDDetect
from astropy.io import fits

def test_roman_case1():
    '''
    This test makes sure that the FPN is not random given the same settings
    "Also tests that when other noise sources are turned on, randomized noise is present despite the same settings which would give identical FPN."
    '''

    detector = EMCCDDetect(fpn_path = 'roman', cic=0, read_noise = 0, dark_current=1)

    out_array1 = detector.sim_full_frame(np.zeros((1024,1024)), 1)
    prescan_result1 = detector.meta.slice_section(out_array1, 'prescan')
    image_result1 = detector.meta.slice_section(out_array1, 'image')

    out_array2 = detector.sim_full_frame(np.zeros((1024,1024)), 1)
    prescan_result2 = detector.meta.slice_section(out_array2, 'prescan')
    image_result2 = detector.meta.slice_section(out_array2, 'image')

    assert np.array_equal(prescan_result1, prescan_result2 )
    assert not np.array_equal(image_result1, image_result2)

def test_None_case1():
    '''
    This test does the same thing as the one above expect for the None case
    '''

    detector = EMCCDDetect(fpn_path = None, cic=0, read_noise = 0, dark_current=1)

    out_array1 = detector.sim_full_frame(np.zeros((1024,1024)), 1)
    prescan_result1 = detector.meta.slice_section(out_array1, 'prescan')
    image_result1 = detector.meta.slice_section(out_array1, 'image')

    out_array2 = detector.sim_full_frame(np.zeros((1024,1024)), 1)
    prescan_result2 = detector.meta.slice_section(out_array2, 'prescan')
    image_result2 = detector.meta.slice_section(out_array2, 'image')

    assert np.array_equal(prescan_result1, prescan_result2 )
    assert not np.array_equal(image_result1, image_result2)

def test_None_case2():
    '''
    This test makes sure that when bias_sigma_row and bias_sigma_col are 0, the resulting array is the same number everwhere which should be the integer of bias/eperdn
    '''
    detector = EMCCDDetect(fpn_path = None, cic=0, read_noise = 0, dark_current=0.0, bias_sigma_row = 0, bias_sigma_col = 0, bias = 100)

    out_array1 = detector.sim_full_frame(np.zeros((1024,1024)), 1e-9)
    prescan_result1 = detector.meta.slice_section(out_array1, 'prescan')
    image_result1 = detector.meta.slice_section(out_array1, 'image')
    twelve_array = np.ones_like(image_result1) + 11
    assert np.array_equal(image_result1, twelve_array)


if __name__ == '__main__':
    test_roman_case1()
    test_None_case1()
    test_None_case2()