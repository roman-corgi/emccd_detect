import numpy as np
from scipy import integrate, optimize
from copy import deepcopy
from arcticpy import util


class Trap(object):
    def __init__(
        self,
        density=0.13,
        release_timescale=0.25,
        capture_timescale=0,
        surface=False,
    ):
        """The parameters for a single trap species.

        Controls the density of traps and the timescales/probabilities of
        capture and release, along with utilities for the watermarking tracking
        of trap states and the calculation of capture and release.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.

        release_timescale : float
            The release timescale of the trap, in the same units as the time
            spent in each pixel or phase (Clocker sequence).

        capture_timescale : float
            The capture timescale of the trap. Default 0 for instant capture.

        surface : bool
            #
            # RJM: RESERVED FOR SURFACE TRAPS
            #

        Attributes
        ----------
        capture_rate, emission_rate : float
            The capture and emission rates (Lindegren (1998) section 3.2).
        """

        self.density = float(density)
        self.release_timescale = release_timescale
        self.capture_timescale = capture_timescale
        self.surface = surface

        # Rates
        self.emission_rate = 1 / self.release_timescale

        if self.capture_timescale == 0:
            self.capture_rate = np.inf
        else:
            self.capture_rate = 1 / self.capture_timescale

    def distribution_within_pixel(self, fractional_volume=0):
        if self.surface:
            #
            # RJM: RESERVED FOR SURFACE TRAPS OR SPECIES WITH NONUNIFORM DENSITY WITHIN A PIXEL
            #
            pass
        return None

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """Calculate the fraction of filled traps after a certain time_elapsed.

        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same
            units as the trap timescales.

        Returns
        -------
        fill_fraction : float
            The fraction of filled traps.
        """
        return np.exp(-time_elapsed / self.release_timescale)

    def time_elapsed_from_fill_fraction(self, fill_fraction):
        """Calculate the total time elapsed from the fraction of filled traps.

        Parameters
        ----------
        fill_fraction : float
            The fraction of filled traps.

        Returns
        -------
        time_elapsed : float
            The time elapsed, in the same units as the trap timescales.
        """
        return -self.release_timescale * np.log(fill_fraction)

    def electrons_released_from_electrons_and_dwell_time(self, electrons, dwell_time=1):
        """Calculate the number of released electrons from the trap.

        Parameters
        ----------
        electrons : float
            The initial number of trapped electrons.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the
            trap timescales.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """
        return electrons * (1 - self.fill_fraction_from_time_elapsed(dwell_time))

    def electrons_released_from_time_elapsed_and_dwell_time(
        self, time_elapsed, dwell_time=1
    ):
        """Calculate the number of released electrons from the trap.

        Parameters
        ----------
        time_elapsed : float
            The time elapsed since capture.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the
            trap release_timescale.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """
        return np.exp(-time_elapsed / self.release_timescale) * (
            1 - np.exp(-dwell_time / self.release_timescale)
        )

    @property
    def delta_ellipticity(self):
        """Calculate the effect on a galaxy's ellipticity of the CTI caused by
            a trap species.

        See Israel et al. (2014).
        """

        a = 0.05333
        d_a = 0.03357
        d_p = 1.628
        d_w = 0.2951
        g_a = 0.09901
        g_p = 0.4553
        g_w = 0.4132

        return self.density * (
            a
            + d_a * (np.arctan((np.log(self.release_timescale) - d_p) / d_w))
            + (
                g_a
                * np.exp(
                    -((np.log(self.release_timescale) - g_p) ** 2.0) / (2 * g_w ** 2.0)
                )
            )
        )

    @classmethod
    def poisson_trap(cls, trap, shape, seed=0):
        """
        For a set of traps with a given set of densities (which are in traps
        per pixel), compute a new set of trap densities by drawing new values
        for from a Poisson distribution.

        This requires us to first convert each trap density to the total number
        of traps in the column.

        This is used to model the random distribution of traps on a CCD, which
        changes the number of traps in each column.

        Parameters
        ----------
        trap : Trap
            A trap species object.

        shape : (int, int)
            The shape of the image, so that the correct number of trap densities
            are computed.

        seed : int
            The seed of the Poisson random number generator.
        """
        np.random.seed(seed)
        total_traps = tuple(map(lambda sp: sp.density * shape[0], trap))
        poisson_densities = [
            np.random.poisson(total_traps) / shape[0] for _ in range(shape[1])
        ]
        poisson_trap = []
        for densities in poisson_densities:
            for i, trap_i in enumerate(trap):
                poisson_trap.append(
                    Trap(
                        density=densities[i], release_timescale=trap_i.release_timescale
                    )
                )

        return poisson_trap


class TrapInstantCapture(Trap):
    """ For the old C++ style release-then-instant-capture algorithm. """

    def __init__(self, density=0.13, release_timescale=0.25, surface=False):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.

        release_timescale : float
            The release timescale of the trap, in the same units as the time
            spent in each pixel or phase (Clocker sequence).

        surface : bool
            #
            # RJM: RESERVED FOR SURFACE TRAPS
            #
        """
        super().__init__(
            density=density,
            release_timescale=release_timescale,
            capture_timescale=0,
            surface=surface,
        )


class TrapLifetimeContinuumAbstract(TrapInstantCapture):
    """Base class for a continuum distribution of release lifetimes.

    Must be used with TrapManagerTrackTime.

    Primarily intended to be inherited by a class that sets a particular
    distribution, e.g. TrapLogNormalLifetimeContinuum.
    """

    def __init__(
        self,
        density,
        distribution_of_traps_with_lifetime,
        release_timescale_mu=None,
        release_timescale_sigma=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.

        distribution_of_traps_with_lifetime : func
            The distribution of traps as a function of release_timescale, mu
            lifetime, and lifetime sigma, such that its integral from 0 to
            infinity = 1. e.g. a log-normal probability density function.

        release_timescale_mu : float
            The mu (e.g. mean or median depending on the distribution)
            release timescale of the traps.

        release_timescale_sigma : float
            The sigma of release lifetimes of the traps.
        """
        super(TrapLifetimeContinuumAbstract, self).__init__(
            density=density, release_timescale=release_timescale_mu
        )

        self.distribution_of_traps_with_lifetime = distribution_of_traps_with_lifetime
        self.release_timescale_mu = release_timescale_mu
        self.release_timescale_sigma = release_timescale_sigma

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """Calculate the fraction of filled traps after a certain time_elapsed.

        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same
            units as the trap timescales.

        Returns
        -------
        fill_fraction : float
            The fraction of filled traps.
        """

        def integrand(release_timescale, time_elapsed, mu, sigma):
            return self.distribution_of_traps_with_lifetime(
                release_timescale, mu, sigma
            ) * np.exp(-time_elapsed / release_timescale)

        fill_fraction = integrate.quad(
            integrand,
            0,
            np.inf,
            args=(
                time_elapsed,
                self.release_timescale_mu,
                self.release_timescale_sigma,
            ),
        )[0]

        return fill_fraction

    def time_elapsed_from_fill_fraction(self, fill_fraction):
        """Calculate the total time elapsed from the fraction of filled traps.

        Parameters
        ----------
        fill_fraction : float
            The fraction of filled traps.

        Returns
        -------
        time_elapsed : float
            The time elapsed, in the same units as the trap timescales.
        """
        # Crudely iterate to find the time that gives the required fill fraction
        def find_time(time_elapsed):
            return self.fill_fraction_from_time_elapsed(time_elapsed) - fill_fraction

        time_elapsed = optimize.fsolve(find_time, 0.1 * self.release_timescale_mu)[0]

        # Check solution (seems to be slightly unreliable for very small sigmas)
        assert abs(find_time(time_elapsed)) < 1e-7

        return time_elapsed

    def electrons_released_from_time_elapsed_and_dwell_time(
        self, time_elapsed, dwell_time=1
    ):
        """Calculate the number of released electrons from the trap.

        Parameters
        ----------
        time_elapsed : float
            The time elapsed since capture.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the
            trap timescales.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """

        def integrand(release_timescale, time_elapsed, dwell_time, mu, sigma):
            return (
                self.distribution_of_traps_with_lifetime(release_timescale, mu, sigma)
                * np.exp(-time_elapsed / release_timescale)
                * (1 - np.exp(-dwell_time / release_timescale))
            )

        return integrate.quad(
            integrand,
            0,
            np.inf,
            args=(
                time_elapsed,
                dwell_time,
                self.release_timescale_mu,
                self.release_timescale_sigma,
            ),
        )[0]


class TrapLogNormalLifetimeContinuum(TrapLifetimeContinuumAbstract):
    """ For a log-normal continuum distribution of release lifetimes. """

    @staticmethod
    def log_normal_distribution(x, median, sigma):
        """Return the log-normal probability density.

        Parameters
        ----------
        x : float
            The input value.

        median : float
            The median of the distribution.

        sigma : float
            The sigma of the distribution.

        Returns
        --------
        """
        return np.exp(-((np.log(x) - np.log(median)) ** 2) / (2 * sigma ** 2)) / (
            x * sigma * np.sqrt(2 * np.pi)
        )

    def __init__(
        self,
        density,
        release_timescale_mu=None,
        release_timescale_sigma=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ---------
        density : float
            The density of the trap species in a pixel.

        release_timescale_mu : float
            The median release timescale of the traps.

        release_timescale_sigma : float
            The sigma of release lifetimes of the traps.
        """

        super(TrapLogNormalLifetimeContinuum, self).__init__(
            density=density,
            distribution_of_traps_with_lifetime=self.log_normal_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
