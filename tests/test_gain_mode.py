import numpy as np
import pytest
from pathlib import Path
from emccd_detect.emccd_detect import EMCCDDetect
from astropy.io import fits
import time

def test_low_gain():
    '''
    Test that the 'auto' mode behaves as expected for low gains.
    '''
    np.random.seed(234)
    detector1 = EMCCDDetect(em_gain=10, fast_gain_mode = 'auto')
    t = time.time()
    out_array1 = detector1.sim_full_frame(np.zeros((1024,1024)), 1)
    print('First one, low gain: ', time.time() - t)

    np.random.seed(234)
    detector2 = EMCCDDetect(em_gain=10, fast_gain_mode = False, gain_CIC_Q=0)
    t = time.time()
    out_array2 = detector2.sim_full_frame(np.zeros((1024,1024)), 1)
    print('Second one, low gain: ', time.time() - t)
    assert np.array_equal(out_array1, out_array2)

def test_high_gain():
    '''
    Test that the 'auto' mode behaves as expected for high gains.
    '''
    np.random.seed(345)
    detector1 = EMCCDDetect(em_gain=1000, fast_gain_mode = 'auto')
    t = time.time()
    out_array1 = detector1.sim_full_frame(np.zeros((1024,1024)), 1)
    print('First one, high gain: ', time.time() - t)

    detector2 = EMCCDDetect(em_gain=1000, fast_gain_mode = True, gain_CIC_Q='roman')
    detector2.n1_fast = False #this is set internally; I set it here to get same array (i.e., so that n=1 case uses binomial instead of Gamma)
    np.random.seed(345)
    t = time.time()
    out_array2 = detector2.sim_full_frame(np.zeros((1024,1024)), 1)
    print('Second one, high gain: ', time.time() - t)
    assert np.array_equal(out_array1, out_array2)


if __name__ == '__main__':
    test_low_gain()
    test_high_gain()
    print('Tests passed')