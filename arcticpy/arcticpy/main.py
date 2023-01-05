""" 
ArCTIC python
=============

AlgoRithm for Charge Transfer Inefficiency (CTI) Correction
-----------------------------------------------------------

Add or remove image trails due to charge transfer inefficiency in CCD detectors
by modelling the trapping, releasing, and moving of charge along pixels.

See the README.md for general information, and see the docstrings and comments 
throughout the code for further details.

https://github.com/jkeger/arcticpy

Jacob Kegerreis: jacob.kegerreis@durham.ac.uk  
Richard Massey: r.j.massey@durham.ac.uk  
James Nightingale  
"""

import numpy as np
from copy import deepcopy

from autoarray.structures import frames

from arcticpy.roe import ROE, ROETrapPumping
from arcticpy.ccd import CCD, CCDPhase
from arcticpy.trap_managers import AllTrapManager
from arcticpy.traps import TrapInstantCapture
from arcticpy import util


def _clock_charge_in_one_direction(
    image,
    roe,
    ccd,
    traps,
    express,
    offset,
    window_row_range,
    window_column_range,
    time_window_range,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons
    along their independent columns.

    Parameters
    ----------
    image : [[float]]
        The input array of pixel values, assumed to be in units of electrons.

        The first dimension is the "row" index, the second is the "column"
        index. By default (for parallel clocking), charge is transferred "up"
        from row n to row 0 along each independent column. i.e. the readout
        register is above row 0. (For serial clocking, the image is rotated
        beforehand, outside of this function, see add_cti().)

        e.g. (with arbitrary trap parameters)
        Initial image with one bright pixel in the first three columns:
            [[0.0,     0.0,     0.0,     0.0  ],
             [200.0,   0.0,     0.0,     0.0  ],
             [0.0,     200.0,   0.0,     0.0  ],
             [0.0,     0.0,     200.0,   0.0  ],
             [0.0,     0.0,     0.0,     0.0  ],
             [0.0,     0.0,     0.0,     0.0  ]]
        Final image with CTI trails behind each bright pixel:
            [[0.0,     0.0,     0.0,     0.0  ],
             [196.0,   0.0,     0.0,     0.0  ],
             [3.0,     194.1,   0.0,     0.0  ],
             [2.0,     3.9,     192.1,   0.0  ],
             [1.3,     2.5,     4.8,     0.0  ],
             [0.8,     1.5,     2.9,     0.0  ]]

    roe : ROE
        An object describing the timing and direction(s) in which electrons are
        moved during readout.

    ccd : CCD
        An object describing the way in which a cloud of electrons fills the CCD
        volume.

    traps : [Trap] or [[Trap]]
        A list of one or more trap objects. To use different types of traps that
        will require different watermark levels, pass a 2D list of lists, i.e. a
        list containing lists of one or more traps for each type.

    express : int
        The number of times the pixel-to-pixel transfers are computed,
        determining the balance between accuracy (high values) and speed
        (low values) (Massey et al. 2014, section 2.1.5).
            n_pix   (slower, accurate) Compute every pixel-to-pixel
                    transfer. The default 0 = alias for n_pix.
            k       Recompute on k occasions the effect of each transfer.
                    After a few transfers (and e.g. eroded leading edges),
                    the incremental effect of subsequent transfers can change.
            1       (faster, approximate) Compute the effect of each
                    transfer only once.
        Runtime scales approximately as O(express^0.5). ###WIP

    offset : int (>= 0)
        The number of (e.g. prescan) pixels separating the supplied image from
        the readout register.

    window_row_range : range
        The subset of row pixels to model, to save time when only a specific
        region of the image is of interest. Defaults to range(0, n_pixels) for
        the full image.

    window_column_range : range
        The subset of column pixels to model, to save time when only a specific
        region of the image is of interest. Defaults to range(0, n_columns) for
        the full image.

    time_window_range : range
        The subset of transfers to implement. Defaults to range(0, n_pixels) for
        the full image. e.g. range(0, n_pixels/3) to do only the first third of
        the pixel-to-pixel transfers.

        The entire readout is still modelled, but only the results from this
        subset of transfers are implemented in the final image.

    Returns
    -------
    image : [[float]]
        The output array of pixel values.
    """

    # Generate the arrays over each step for: the number of of times that the
    # effect of each pixel-to-pixel transfer can be multiplied for the express
    # algorithm; and whether the traps must be monitored (usually whenever
    # express matrix > 0, unless using a time window)
    (
        express_matrix,
        monitor_traps_matrix,
    ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
        pixels=window_row_range,
        express=express,
        offset=offset,
        time_window_range=time_window_range,
    )
    # ; and whether the trap occupancy states must be saved for the next express
    # pass rather than being reset (usually at the end of each express pass)
    save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
        express_matrix=express_matrix
    )

    n_express_pass, n_rows_to_process = express_matrix.shape

    # Decide in advance which steps need to be evaluated and which can be skipped
    phases_with_traps = [
        i for i, frac in enumerate(ccd.fraction_of_traps_per_phase) if frac > 0
    ]
    steps_with_nonzero_dwell_time = [
        i for i, time in enumerate(roe.dwell_times) if time > 0
    ]

    # Set up the set of trap managers to monitor the occupancy of all trap species
    if isinstance(roe, ROETrapPumping):
        # For trap pumping there is only one pixel and row to process but
        # multiple transfers back and forth without clearing the watermarks
        # Note, this allows for many more watermarks than are actually needed
        # in standard trap-pumping clock sequences
        max_n_transfers = n_express_pass * len(steps_with_nonzero_dwell_time)
    else:
        max_n_transfers = n_rows_to_process * len(steps_with_nonzero_dwell_time)
    trap_managers = AllTrapManager(
        traps=traps,
        n_columns=len(window_column_range),
        max_n_transfers=max_n_transfers,
        ccd=ccd,
    )

    # Temporarily expand image, if charge released from traps ever migrates to
    # a different charge packet, at any time during the clocking sequence
    n_rows_zero_padding = max(roe.pixels_accessed_during_clocking) - min(
        roe.pixels_accessed_during_clocking
    )
    zero_padding = np.zeros((n_rows_zero_padding, image.shape[1]), dtype=image.dtype)
    image = np.concatenate((image, zero_padding), axis=0)

    # Monitor the traps in every pixel, or just one (express=1) or a few
    # (express=a few) then replicate their effect
    for express_index in range(n_express_pass):
        # Restore the trap occupancy levels (to empty, or to a saved state
        # from a previous express pass)
        trap_managers.restore()

        # Each pixel
        for row_index in range(len(window_row_range)):
            express_multiplier = express_matrix[express_index, row_index]
            # Skip this step if not needed to be evaluated (may need to
            # monitor the traps and update their occupancies even if
            # express_mulitplier is 0, e.g. for a time window)
            if not monitor_traps_matrix[express_index, row_index]:
                continue

            for clocking_step in steps_with_nonzero_dwell_time:

                for phase in phases_with_traps:
                    # Information about the potentials in this phase
                    roe_phase = roe.clock_sequence[clocking_step][phase]

                    # Select the relevant pixel (and phase) for the initial charge
                    row_index_read = (
                        window_row_range[row_index]
                        + roe_phase.capture_from_which_pixels
                    )[0]  ## why an array not an int without the [0]?

                    # Initial charge (0 if this phase's potential is not high)
                    n_free_electrons = image[row_index_read] * roe_phase.is_high

                    # Allow electrons to be released from and captured by traps
                    n_electrons_released_and_captured = 0
                    for trap_manager in trap_managers[phase]:
                        n_electrons_released_and_captured += (
                            trap_manager.n_electrons_released_and_captured(
                                n_free_electrons=n_free_electrons,
                                dwell_time=roe.dwell_times[clocking_step],
                                ccd_filling_function=ccd.well_filling_function(
                                    phase=phase
                                ),
                                express_multiplier=express_multiplier,
                            )
                        )

                    # Skip updating the image if only monitoring the traps
                    if express_multiplier == 0:
                        continue

                    # Select the relevant pixel (and phase(s)) for the returned charge
                    row_index_write = (
                        window_row_range[row_index] + roe_phase.release_to_which_pixels
                    )

                    # Return the electrons back to the relevant charge cloud, or
                    # a fraction if they are being returned to multiple phases
                    image[row_index_write] += (
                        n_electrons_released_and_captured
                        * roe_phase.release_fraction_to_pixel
                        * express_multiplier
                    )

                    # Make sure image counts don't go negative, as could
                    # otherwise happen with a too-large express_multiplier
                    for row_index_single in row_index_write:
                        image[row_index_single][image[row_index_single] < 0] = 0

            # Save the trap occupancy states for the next express pass
            if save_trap_states_matrix[express_index, row_index]:
                trap_managers.save()

    # Unexpand the image to its original dimensions
    if n_rows_zero_padding > 0:
        image = image[0:-n_rows_zero_padding, :]

    return image


def add_cti(
    image,
    parallel_ccd=None,
    parallel_roe=None,
    parallel_traps=None,
    parallel_express=0,
    parallel_offset=0,
    parallel_window_range=None,
    serial_ccd=None,
    serial_roe=None,
    serial_traps=None,
    serial_express=0,
    serial_offset=0,
    serial_window_range=None,
    time_window_range=None,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons
    along their independent columns, for parallel and/or serial clocking.

    Parameters
    ----------
    image : [[float]] or frames.Frame
        The input array of pixel values, assumed to be in units of electrons.

        The first dimension is the "row" index, the second is the "column"
        index. By default (for parallel clocking), charge is transfered "up"
        from row n to row 0 along each independent column. i.e. the readout
        register is above row 0. (For serial clocking, the image is rotated
        before modelling, such that charge moves from column n to column 0.)

        e.g. (with arbitrary trap parameters)
        Initial image with one bright pixel in the first three columns:
            [[0.0,     0.0,     0.0,     0.0  ],
             [200.0,   0.0,     0.0,     0.0  ],
             [0.0,     200.0,   0.0,     0.0  ],
             [0.0,     0.0,     200.0,   0.0  ],
             [0.0,     0.0,     0.0,     0.0  ],
             [0.0,     0.0,     0.0,     0.0  ]]
        Image with parallel CTI trails:
            [[0.0,     0.0,     0.0,     0.0  ],
             [196.0,   0.0,     0.0,     0.0  ],
             [3.0,     194.1,   0.0,     0.0  ],
             [2.0,     3.9,     192.1,   0.0  ],
             [1.3,     2.5,     4.8,     0.0  ],
             [0.8,     1.5,     2.9,     0.0  ]]
        Final image with parallel and serial CTI trails:
            [[0.0,     0.0,     0.0,     0.0  ],
             [194.1,   1.9,     1.5,     0.9  ],
             [2.9,     190.3,   2.9,     1.9  ],
             [1.9,     3.8,     186.5,   3.7  ],
             [1.2,     2.4,     4.7,     0.9  ],
             [0.7,     1.4,     2.8,     0.6  ]]

    parallel_express : int
        The number of times the transfers are computed, determining the
        balance between accuracy (high values) and speed (low values), for
        parallel clocking (Massey et al. 2014, section 2.1.5).

    parallel_roe : ROE
        The object describing the clocking read-out electronics for parallel
        clocking.

    parallel_ccd : CCD
        The object describing the CCD volume for parallel clocking. For
        multi-phase clocking optionally use a list of different CCD volumes
        for each phase, in the same size list as parallel_roe.dwell_times.

    parallel_traps : [Trap] or [[Trap]]
        A list of one or more trap objects for parallel clocking. To use
        different types of traps that will require different watermark
        levels, pass a 2D list of lists, i.e. a list containing lists of
        one or more traps for each type.

    parallel_offset : int (>= 0)
        The number of (e.g. prescan) pixels separating the supplied image from
        the readout register. i.e. Treat the input image as a sub-image that is
        offset this number of pixels from readout, increasing the number of
        pixel-to-pixel transfers.

    parallel_window_range : range
        For speed, calculate only the effect on this subset of pixels. Defaults
        to range(0, n_pixels) for the full image.

        Note that, because of edge effects, the range should be started several
        pixels before the actual region of interest.

        For a single pixel (e.g. for trap pumping), can enter just the single
        integer index of the pumping traps to monitor, which will be converted
        to range(index, index + 1).

    serial_* : *
        The same as the parallel_* objects described above but for serial
        clocking instead.

    time_window_range : range
        The subset of transfers to implement. Defaults to range(0, n_pixels) for
        the full image. e.g. range(0, n_pixels/3) to do only the first third of
        the pixel-to-pixel transfers.

        The entire readout is still modelled, but only the results from this
        subset of transfers are implemented in the final image.

        This could be used to e.g. add cosmic rays during readout of simulated
        images. Successive calls to complete the readout should start at
        the same value that the previous one ended, e.g. range(0, 1000) then
        range(1000, 2000). Be careful not to divide the readout too finely, as
        there is only as much temporal resolution as there are rows (not rows *
        phases) in the image. Also, for each time that readout is split between
        successive calls to this function, the output in one row of pixels
        will change slightly (unless express=0) because trap occupancy is
        not stored between calls.

    Returns
    -------
    image : [[float]] or frames.Frame
        The output array of pixel values.
    """
    n_rows_in_image, n_columns_in_image = image.shape

    # Default windows to the full image; convert single-pixel windows to ranges
    if parallel_window_range is None:
        parallel_window_range = range(n_rows_in_image)
    elif isinstance(parallel_window_range, int):
        parallel_window_range = range(parallel_window_range, parallel_window_range + 1)
    if serial_window_range is None:
        serial_window_range = range(n_columns_in_image)
    elif isinstance(serial_window_range, int):
        serial_window_range = range(serial_window_range, serial_window_range + 1)
    if time_window_range is None:
        time_window_range = range(n_rows_in_image + parallel_offset)
        # Set the "columns" window in the rotated image for serial clocking
        serial_window_column_range = parallel_window_range
    else:
        # Intersection of spatial and time windows for serial clocking
        serial_window_column_range = range(
            int(max(parallel_window_range[0], time_window_range[0] - parallel_offset)),
            int(min(parallel_window_range[-1], time_window_range[-1] - parallel_offset))
            + 1,
        )

    # Default ROE: simple, single-phase clocking in imaging mode
    if parallel_roe is None:
        parallel_roe = ROE()
    if serial_roe is None:
        serial_roe = ROE()

    # Don't modify the external array passed to this function
    image_add_cti = deepcopy(image)

    # Parallel clocking
    if parallel_traps is not None:

        # Transfer charge in parallel direction
        image_add_cti = _clock_charge_in_one_direction(
            image=image_add_cti,
            ccd=parallel_ccd,
            roe=parallel_roe,
            traps=parallel_traps,
            express=parallel_express,
            offset=parallel_offset,
            window_row_range=parallel_window_range,
            window_column_range=serial_window_range,
            time_window_range=time_window_range,
        )

    # Serial clocking
    if serial_traps is not None:

        # Switch axes, so clocking happens in other direction
        image_add_cti = image_add_cti.T.copy()

        # Transfer charge in serial direction
        image_add_cti = _clock_charge_in_one_direction(
            image=image_add_cti,
            ccd=serial_ccd,
            roe=serial_roe,
            traps=serial_traps,
            express=serial_express,
            offset=serial_offset,
            window_row_range=serial_window_range,
            window_column_range=serial_window_column_range,
            time_window_range=None,
        )

        # Switch axes back
        image_add_cti = image_add_cti.T

    # TODO : Implement as decorator

    if isinstance(image, frames.Frame):

        return image.__class__(
            array=image_add_cti,
            mask=image.mask,
            original_roe_corner=image.original_roe_corner,
            scans=image.scans,
            exposure_info=image.exposure_info,
        )

    return image_add_cti


def remove_cti(
    image,
    iterations,
    parallel_ccd=None,
    parallel_roe=None,
    parallel_traps=None,
    parallel_express=0,
    parallel_offset=0,
    parallel_window_range=None,
    serial_ccd=None,
    serial_roe=None,
    serial_traps=None,
    serial_express=0,
    serial_offset=0,
    serial_window_range=None,
    time_window_range=None,
):
    """
    Remove CTI trails from an image by first modelling the addition of CTI.

    See add_cti()'s documentation for the forward modelling. This function
    iteratively models the addition of more CTI trails to the input image to
    then extract the corrected image without the original trails.

    Parameters
    ----------
    All parameters are identical to those of add_cti() as described in its
    documentation, with the exception of:

    iterations : int
        The number of times CTI-adding clocking is run to perform the correction
        via forward modelling.

    Returns
    -------
    image : [[float]] or frames.Frame
        The output array of pixel values with CTI removed.
    """

    # Initialise the iterative estimate of removed CTI; don't modify the external array
    image_remove_cti = deepcopy(image)

    # Estimate the image with removed CTI more precisely each iteration
    for iteration in range(iterations):

        image_add_cti = add_cti(
            image=image_remove_cti,
            parallel_ccd=parallel_ccd,
            parallel_roe=parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=parallel_express,
            parallel_offset=parallel_offset,
            parallel_window_range=parallel_window_range,
            serial_ccd=serial_ccd,
            serial_roe=serial_roe,
            serial_traps=serial_traps,
            serial_express=serial_express,
            serial_offset=serial_offset,
            serial_window_range=serial_window_range,
            time_window_range=time_window_range,
        )

        # Improved estimate of image with CTI trails removed
        image_remove_cti += image - image_add_cti

    # TODO : Implement as decorator

    if isinstance(image, frames.Frame):

        return image.__class__(
            array=image_remove_cti,
            mask=image.mask,
            original_roe_corner=image.original_roe_corner,
            scans=image.scans,
            exposure_info=image.exposure_info,
        )

    return image_remove_cti


def model_for_HST_ACS(date):
    """
    Return arcticpy objects that provide a preset CTI model for the Hubble Space
    Telescope (HST) Advanced Camera for Surveys (ACS).

    The returned objects are ready to be passed to add_cti() or remove_cti()
    for parallel clocking.

    Parameters
    ----------
    date : float
        The Julian date. Should not be before the launch date of 2452334.5.

    Returns
    -------
    traps : [Trap]
        A list of trap objects that set the parameters for each trap species.
        See traps.py.

    ccd : CCD
        The CCD object that describes how electrons fill the volume. See ccd.py.

    roe : ROE
        The ROE object that describes the readout electronics. See roe.py.
    """

    # Key dates when ACS/WFC configuration changed
    launch_date = 2452334.5
    temperature_change_date = 2453920
    sm4_repair_date = 2454968

    assert date >= launch_date, "Julian date must be after launch, i.e. >= 2452334.5"

    # Trap density
    if date < sm4_repair_date:
        trap_initial_density = 0.017845
        trap_growth_rate = 3.5488e-4
    else:
        trap_initial_density = -0.246591 * 1.011
        trap_growth_rate = 0.000558980 * 1.011
    total_trap_density = trap_initial_density + trap_growth_rate * (date - launch_date)
    trap_densities = (
        np.array([1.27, 3.38, 2.85]) / (1.27 + 3.38 + 2.85) * total_trap_density
    )

    # Trap release time
    if date < temperature_change_date:
        operating_temperature = 273.15 - 77
    else:
        operating_temperature = 273.15 - 81
    sm4_temperature = 273.15 - 81  # K
    k = 8.617343e-5  # eV / K
    DeltaE = np.array([0.31, 0.34, 0.44])  # eV
    trap_release_times = (
        np.array([0.74, 7.7, 37])
        * (operating_temperature / (273.15 - 81))
        * np.exp(
            DeltaE
            / (k * sm4_temperature * operating_temperature)
            * (operating_temperature - sm4_temperature)
        )
    )  # pixels

    # Assemble variables to pass to add_cti()
    traps = [
        TrapInstantCapture(
            density=trap_densities[0], release_timescale=trap_release_times[0]
        ),
        TrapInstantCapture(
            density=trap_densities[1], release_timescale=trap_release_times[1]
        ),
        TrapInstantCapture(
            density=trap_densities[2], release_timescale=trap_release_times[2]
        ),
    ]

    ccd = CCD(full_well_depth=84700, well_fill_power=0.478, well_notch_depth=0)

    roe = ROE(dwell_times=[1])

    return traps, ccd, roe
