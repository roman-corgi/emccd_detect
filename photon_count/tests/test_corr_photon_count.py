"""Unit tests for corr_photon_count module."""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from proc_cgi_frame.corr_photon_count import (calc_lam_approx,
                                              corr_photon_count, get_count_rate,
                                              lam_newton_fit, _calc_dfunc,
                                              _calc_func)
from proc_cgi_frame.corr_photon_count import CorrPhotonCountException


# Analytical results from Mathematica
nobs_mat = np.array([50])
nfr_mat = 100
t_mat = 1000
g_mat = 5000
lam_mat = np.array([0.943408566833047])

func_mat = np.array([3.918879853876746])
dfunc_mat = np.array([37.62258292251463])

# From doing the following in Mathematica: lam_mat - func_mat/dfunc_mat
lam_est_mat1 = np.array([0.839245600086509])

# Recalculate func_mat and dfunc_mat using lam_est_mat1 as the lam_mat input
func_mat1 = np.array([-0.178674777567025])
dfunc_mat1 = np.array([41.10079379928366])

# Calculate lam_est_mat1 - func_mat1/dfunc_mat1
lam_est_mat2 = lam_est_mat1 - func_mat1/dfunc_mat1


class TestGetCountRate(unittest.TestCase):
    """Unit tests for get_count_rate function."""

    def setUp(self):
        self.tol = 1e-12

        self.thresh = 1000.
        self.em_gain = 5000.
        self.niter = 2

        self.frames = np.zeros((3, 5, 5)).astype(np.float64)

        # Analytical results from Mathematica (for 2 iterations through
        # Newton's method)
        self.count1_frames3 = 0.4934795083952269
        self.count2_frames3 = 1.342380368984298

    def test_analytical(self):
        """Verify function matches analytical results."""
        frames = self.frames.copy()  # nframes should be 3 from this
        # This should be counted (2 counts)
        frames[0, 0, 0] = self.thresh + 1
        frames[2, 0, 0] = self.thresh + 3  # Shouldn't matter how high above thresh
        # This should be counted (1 count)
        frames[0, 3, 0] = self.thresh + 1
        # These should not be counted (below or at threshold)
        frames[1, 0, 1] = self.thresh
        frames[2, 0, 1] = self.thresh
        frames[0, 2, 0] = self.thresh - 1

        rate_expected = np.zeros_like(frames[0])
        rate_expected[0, 0] = self.count2_frames3
        rate_expected[3, 0] = self.count1_frames3

        rate = get_count_rate(frames, self.thresh, self.em_gain, self.niter)

        self.assertTrue(np.max(np.abs(rate - rate_expected)) < self.tol)

    def test_negative_is_zero(self):
        """Verify function returns zeros for any negative input values."""
        frames = self.frames.copy()
        frames[0, 0, 0] = -1

        rate_expected = np.zeros_like(frames[0])

        rate = get_count_rate(frames, self.thresh, self.em_gain, self.niter)

        self.assertTrue(np.max(np.abs(rate - rate_expected)) < self.tol)

    def test_shape_match(self):
        """Verify output array matches the shape of the input arrays."""
        nside = 5
        mside = 6
        frames = np.ones((3, nside, mside)).astype(np.float64)

        rate = get_count_rate(frames, self.thresh, self.em_gain, self.niter)

        self.assertEqual(rate.shape, (nside, mside))

    def test_defualt_niter2(self):
        """Verify function defaults to niter=2."""
        frames = self.frames.copy()
        # This should be counted (1 count)
        frames[0, 3, 0] = self.thresh + 1

        rate_expected = np.zeros_like(frames[0])
        rate_expected[3, 0] = self.count1_frames3

        rate = get_count_rate(frames, self.thresh, self.em_gain)

        self.assertTrue(np.max(np.abs(rate - rate_expected)) < self.tol)

    def test_exception_not_array(self):
        """Verify that exception is thrown if input is not an array."""
        frames = 1
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(frames, self.thresh, self.em_gain)

    def test_exception_thresh_negative(self):
        """Verify that exception is thrown if thresh is negative."""
        thresh = -1
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(self.frames, thresh, self.em_gain, self.niter)

    def test_exception_em_gain_less_than_zero(self):
        """Verify that exception is thrown if em_gain is less than or equal to
        0.

        """
        em_gain = 0
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(self.frames, self.thresh, em_gain, self.niter)
        em_gain = -1
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(self.frames, self.thresh, em_gain, self.niter)

    def test_niter_greater_than_1(self):
        """Verify that exception is thrown if niter is not an integer greater
        than 1.

        """
        niter = 2.1
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(self.frames, self.thresh, self.em_gain, niter)
        niter = 0
        with self.assertRaises(CorrPhotonCountException):
            get_count_rate(self.frames, self.thresh, self.em_gain, niter)


class TestCorrPhotonCount(unittest.TestCase):
    """Unit tests for corr_photon_count function."""

    def setUp(self):
        self.tol = 1e-12

    def test_1_iter(self):
        """Verify function returns expected value for 1 iteration."""
        lam = corr_photon_count(nobs_mat, nfr_mat, t_mat, g_mat, niter=1)

        self.assertTrue(np.max(np.abs(lam - lam_est_mat1)) < self.tol)

    def test_2_iter(self):
        """Verify function returns expected value for 2 iterations."""
        lam = corr_photon_count(nobs_mat, nfr_mat, t_mat, g_mat, niter=2)

        self.assertTrue(np.max(np.abs(lam - lam_est_mat2)) < self.tol)

    def test_defualt_niter2(self):
        """Verify function defaults to niter=2."""
        lam = corr_photon_count(nobs_mat, nfr_mat, t_mat, g_mat)

        self.assertTrue(np.max(np.abs(lam - lam_est_mat2)) < self.tol)


class TestCalcLamApprox(unittest.TestCase):
    """Unit tests for calc_lam_approx function."""

    def setUp(self):
        self.nfr = np.ceil(2*np.exp(1))
        self.t = 1
        self.g = 1

    def test_less_than1(self):
        """Verify function returns expected value when nobs/nfr * np.exp(t/g)
        is less than 1.

        """
        nobs = np.array([1])
        less_than1 = (nobs/self.nfr) * np.exp(self.t/self.g)
        lam_check = -np.log(1 - less_than1)

        lam = calc_lam_approx(nobs, self.nfr, self.t, self.g)

        self.assertEqual(lam.tolist(), lam_check.tolist())

    def test_equal_to1(self):
        """Verify function returns expected value when nobs/nfr * np.exp(t/g)
        is equal to 1.

        """
        nobs = np.array([self.nfr])
        t = np.log(1)
        equal_to1 = nobs/self.nfr
        lam_check = equal_to1

        lam = calc_lam_approx(nobs, self.nfr, t, self.g)

        self.assertEqual(lam.tolist(), lam_check.tolist())

    def test_greater_than1(self):
        """Verify function returns expected value when nobs/nfr * np.exp(t/g)
        is greater than 1.

        """
        nobs = np.array([self.nfr])
        greater_than1 = nobs/self.nfr
        lam_check = greater_than1

        lam = calc_lam_approx(nobs, self.nfr, self.t, self.g)

        self.assertEqual(lam.tolist(), lam_check.tolist())

    def test_mixed(self):
        """Verify function returns expected value when nobs/nfr * np.exp(t/g)
        is less than 1 for some values and greater than or equal to 1 for
        others.

        """
        nobs = np.array([[1, self.nfr], [self.nfr, 1]])

        less_than1 = (1/self.nfr) * np.exp(self.t/self.g)
        greater_than1 = self.nfr/self.nfr
        lam_check_less = -np.log(1 - less_than1)
        lam_check_greater = greater_than1

        lam_check_array = np.array([[lam_check_less, lam_check_greater],
                                    [lam_check_greater, lam_check_less]])

        lam = calc_lam_approx(nobs, self.nfr, self.t, self.g)

        self.assertEqual(lam.tolist(), lam_check_array.tolist())


class TestLamNewtonFit(unittest.TestCase):
    """Unit tests for lam_newton_fit function."""

    def setUp(self):
        self.tol = 1e12

    def test_1_iter(self):
        """Verify function matches analytical results from Mathematica for 1
        iteration.

        """
        lam_est = lam_newton_fit(nobs_mat, nfr_mat, t_mat, g_mat, lam_mat,
                                 niter=1)

        self.assertTrue(np.max(np.abs(lam_est - lam_est_mat1)) < self.tol)

    def test_2_iter(self):
        """Verify function matches analytical results from Mathematica for 2
        iterations.

        """
        lam_est = lam_newton_fit(nobs_mat, nfr_mat, t_mat, g_mat, lam_mat,
                                 niter=2)

        self.assertTrue(np.max(np.abs(lam_est - lam_est_mat2)) < self.tol)

    def test_zero(self):
        """Verify function returns a zero for every zero input."""
        nobs_array = np.zeros((2, 2))
        nobs_array[0, 0] = nobs_mat  # Every value except this one should be zero

        lam_array = np.zeros_like(nobs_array)
        lam_array[0, 0] = lam_mat

        array_expected = np.zeros_like(nobs_array)
        array_expected[0, 0] = lam_est_mat2

        lam_est = lam_newton_fit(nobs_array, nfr_mat, t_mat, g_mat, lam_array,
                                 niter=2)

        self.assertTrue(np.max(np.abs(lam_est - array_expected)) < self.tol)


class Test_CalcFunc(unittest.TestCase):
    """Unit tests for _calc_func function."""

    def test_func_analytical(self):
        """Verify function matches analytical results from Mathematica."""
        tol = 1e12

        # Calculate for several different inputs
        func = _calc_func(nobs_mat, nfr_mat, t_mat, g_mat, lam_mat)
        func1 = _calc_func(nobs_mat, nfr_mat, t_mat, g_mat, lam_est_mat1)

        self.assertTrue(np.max(np.abs(func - func_mat)) < tol)
        self.assertTrue(np.max(np.abs(func1 - func_mat1)) < tol)


class Test_CalcDfunc(unittest.TestCase):
    """Unit tests for _calc_dfunc function."""

    def test_dfunc_analytical(self):
        """Verify function matches analytical results from Mathematica."""
        tol = 1e12

        # Calculate for several different inputs
        dfunc = _calc_dfunc(nfr_mat, t_mat, g_mat, lam_mat)
        dfunc1 = _calc_dfunc(nfr_mat, t_mat, g_mat, lam_est_mat1)

        self.assertTrue(np.max(np.abs(dfunc - dfunc_mat)) < tol)
        self.assertTrue(np.max(np.abs(dfunc1 - dfunc_mat1)) < tol)
