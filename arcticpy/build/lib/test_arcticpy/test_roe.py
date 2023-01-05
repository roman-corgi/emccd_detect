import pytest
import numpy as np

import arcticpy as ac


class TestExpressMatrix:
    def test__express_matrix_from_pixels(self):

        roe = ac.ROE(empty_traps_for_first_transfers=False, express_matrix_dtype=int)
        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=1
        )

        assert express_matrix == pytest.approx(np.array([np.arange(1, 13)]))

        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=4
        )

        assert express_matrix == pytest.approx(
            np.array(
                [
                    [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                ]
            )
        )

        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=12
        )

        assert express_matrix == pytest.approx(np.triu(np.ones((12, 12))))

    def test__express_matrix__offset(self):

        roe = ac.ROE(empty_traps_for_first_transfers=False, express_matrix_dtype=int)
        pixels = 12

        # Compare with offset added directly
        for offset in [1, 5, 11]:
            for express in [1, 3, 12, 0]:
                (
                    express_matrix,
                    _,
                ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
                    pixels=pixels,
                    express=express,
                    offset=offset,
                )

                (
                    express_matrix_manual_offset,
                    _,
                ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
                    pixels=pixels + offset,
                    express=express,
                )

                assert express_matrix[:] == pytest.approx(
                    express_matrix_manual_offset[:, offset:]
                )

    def test__express_matrix__dtype(self):

        # Unchanged for empty_traps_for_first_transfers = False
        roe = ac.ROE(empty_traps_for_first_transfers=False, express_matrix_dtype=float)
        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=1
        )

        assert express_matrix == pytest.approx(np.array([np.arange(1, 13)]))

        roe = ac.ROE(empty_traps_for_first_transfers=True, express_matrix_dtype=float)

        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=4
        )

        assert express_matrix == pytest.approx(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.75, 1.75, 2.75],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0.5, 1.5, 2.5, 2.75, 2.75, 2.75],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0.25, 1.25, 2.25, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 2, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75],
                ]
            )
        )

        # Unchanged for no express
        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=12
        )

        assert express_matrix == pytest.approx(np.triu(np.ones((12, 12))))

    def test__express_matrix__empty_traps_for_first_transfers(self):

        roe = ac.ROE(empty_traps_for_first_transfers=True, express_matrix_dtype=int)
        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=1
        )

        express_matrix_check = np.rot90(np.diag(np.ones(12)))
        express_matrix_check[-1] += np.arange(12)

        assert express_matrix == pytest.approx(express_matrix_check)

        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=4
        )

        assert express_matrix == pytest.approx(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                ]
            )
        )

        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=12
        )

        assert express_matrix == pytest.approx(np.triu(np.ones((12, 12))))

    def test__express_matrix_always_sums_to_n_transfers(self):
        for pixels in [5, 7, 17]:
            for express in [0, 1, 2, 7]:
                for offset in [0, 1, 13]:
                    for dtype in [int, float]:
                        for empty_traps_for_first_transfers in [True, False]:
                            roe = ac.ROE(
                                empty_traps_for_first_transfers=empty_traps_for_first_transfers,
                                express_matrix_dtype=dtype,
                            )
                            (
                                express_matrix,
                                _,
                            ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
                                pixels=pixels, express=express, offset=offset
                            )
                            assert np.sum(express_matrix, axis=0) == pytest.approx(
                                np.arange(1, pixels + 1) + offset
                            )

    def test__monitor_traps_matrix(self):

        roe = ac.ROE(empty_traps_for_first_transfers=False)
        (
            _,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=1
        )

        assert monitor_traps_matrix == pytest.approx(
            np.array([np.ones(12).astype(bool)])
        )

        (
            _,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=4
        )

        assert monitor_traps_matrix == pytest.approx(
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            )
        )

        (
            _,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=12
        )

        assert monitor_traps_matrix == pytest.approx(
            np.triu(np.ones((12, 12)).astype(bool))
        )

    def test__express_matrix_monitor_and_traps_matrix__time_window(self):
        roe = ac.ROE(express_matrix_dtype=int)
        express = 2
        offset = 2
        pixels = 8
        total_pixels = pixels + offset

        (
            express_matrix_a,
            monitor_traps_matrix_a,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels, express, offset=offset, time_window_range=range(0, 6)
        )
        (
            express_matrix_b,
            monitor_traps_matrix_b,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels, express, offset=offset, time_window_range=range(6, 9)
        )
        (
            express_matrix_c,
            monitor_traps_matrix_c,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels,
            express,
            offset=offset,
            time_window_range=range(9, total_pixels),
        )
        (
            express_matrix_d,
            monitor_traps_matrix_d,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels, express, offset=offset
        )
        total_transfers = (
            np.sum(express_matrix_a, axis=0)
            + np.sum(express_matrix_b, axis=0)
            + np.sum(express_matrix_c, axis=0)
        )

        # All monitor all the transfers
        assert monitor_traps_matrix_a == pytest.approx(monitor_traps_matrix_b)
        assert monitor_traps_matrix_a == pytest.approx(monitor_traps_matrix_c)
        assert monitor_traps_matrix_a == pytest.approx(monitor_traps_matrix_d)

        # Separate time windows add to the full image
        assert total_transfers == pytest.approx(np.arange(1, pixels + 1) + offset)
        assert express_matrix_a + express_matrix_b + express_matrix_c == pytest.approx(
            express_matrix_d
        )

    def test__save_trap_states_matrix(self):

        roe = ac.ROE()
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=1
        )
        save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
            express_matrix=express_matrix
        )

        assert save_trap_states_matrix == pytest.approx(
            (express_matrix * 0).astype(bool)
        )

        roe = ac.ROE(empty_traps_for_first_transfers=False)
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=12, express=4
        )
        save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
            express_matrix=express_matrix
        )

        # Save on the pixel before where the next express pass will begin, so
        # that the trap states are appropriate for continuing
        assert save_trap_states_matrix == pytest.approx(
            np.array(
                [
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            )
        )


class TestClockingSequences:
    def test__release_fractions_sum_to_unity(self):

        for n_phases in [1, 2, 3, 5, 8]:
            roe = ac.ROE([1] * n_phases)
            for step in roe.clock_sequence:
                for phase in step:
                    assert sum(phase.release_fraction_to_pixel) == 1

    def test__readout_sequence_single_phase_single_phase_high(self):

        for force_release_away_from_readout in [True, False]:
            roe = ac.ROE(
                [1], force_release_away_from_readout=force_release_away_from_readout
            )
            assert roe.pixels_accessed_during_clocking == pytest.approx([0])
            assert roe.n_phases == 1
            assert roe.n_steps == 1
            assert roe.clock_sequence[0][0].is_high
            assert roe.clock_sequence[0][0].capture_from_which_pixels == 0
            assert roe.clock_sequence[0][0].release_to_which_pixels == 0

    def test__readout_sequence_two_phase_single_phase_high(self):

        n_phases = 2

        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=False)
        assert roe.pixels_accessed_during_clocking == pytest.approx([-1, 0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert all(
            roe.clock_sequence[0][1].release_to_which_pixels == np.array([-1, 0])
        )
        assert all(roe.clock_sequence[1][0].release_to_which_pixels == np.array([0, 1]))

        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=True)
        assert roe.pixels_accessed_during_clocking == pytest.approx([0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert all(roe.clock_sequence[0][1].release_to_which_pixels == np.array([0, 1]))
        assert all(roe.clock_sequence[1][0].release_to_which_pixels == np.array([0, 1]))

    def test__readout_sequence_three_phase_single_phase_high(self):

        n_phases = 3

        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=False)
        assert roe.pixels_accessed_during_clocking == pytest.approx([-1, 0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert roe.clock_sequence[0][1].release_to_which_pixels == 0
        assert roe.clock_sequence[0][2].release_to_which_pixels == -1
        assert roe.clock_sequence[1][0].release_to_which_pixels == 0
        assert roe.clock_sequence[1][2].release_to_which_pixels == 0
        assert roe.clock_sequence[2][0].release_to_which_pixels == 1
        assert roe.clock_sequence[2][1].release_to_which_pixels == 0

        # Never move electrons ahead of the trap
        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=True)
        assert roe.pixels_accessed_during_clocking == pytest.approx([0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        # Check all high phases
        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert roe.clock_sequence[0][1].release_to_which_pixels == 1
        assert roe.clock_sequence[0][2].release_to_which_pixels == 0
        assert roe.clock_sequence[1][0].release_to_which_pixels == 0
        assert roe.clock_sequence[1][2].release_to_which_pixels == 1
        assert roe.clock_sequence[2][0].release_to_which_pixels == 1
        assert roe.clock_sequence[2][1].release_to_which_pixels == 0

    def test__trappumping_sequence_three_phase_single_phase_high(self):

        n_phases = 3

        roe = ac.ROETrapPumping([1] * 2 * n_phases)
        assert roe.pixels_accessed_during_clocking == pytest.approx([-1, 0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == 2 * n_phases

        for step in range(roe.n_steps):
            phase = ([0, 1, 2, 0, 2, 1])[step]
            assert roe.clock_sequence[step][phase].is_high
        assert roe.clock_sequence[3][0].release_to_which_pixels == 1
        assert roe.clock_sequence[3][1].release_to_which_pixels == 1
        assert roe.clock_sequence[3][2].release_to_which_pixels == 0
        for phase in [0, 1, 2]:
            assert (
                roe.clock_sequence[4][phase].release_to_which_pixels
                == roe.clock_sequence[2][phase].release_to_which_pixels
            )
            assert (
                roe.clock_sequence[4][phase].capture_from_which_pixels
                == roe.clock_sequence[2][phase].capture_from_which_pixels
            )
            assert (
                roe.clock_sequence[5][phase].release_to_which_pixels
                == roe.clock_sequence[1][phase].release_to_which_pixels
            )
            assert (
                roe.clock_sequence[5][phase].capture_from_which_pixels
                == roe.clock_sequence[1][phase].capture_from_which_pixels
            )

    def test__readout_sequence_four_phase_single_phase_high(self):

        n_phases = 4

        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=False)
        assert roe.pixels_accessed_during_clocking == pytest.approx([-1, 0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert roe.clock_sequence[0][1].release_to_which_pixels == 0
        assert all(
            roe.clock_sequence[0][2].release_to_which_pixels == np.array([-1, 0])
        )
        assert roe.clock_sequence[0][3].release_to_which_pixels == -1
        assert roe.clock_sequence[1][0].release_to_which_pixels == 0
        assert roe.clock_sequence[1][2].release_to_which_pixels == 0
        assert all(
            roe.clock_sequence[1][3].release_to_which_pixels == np.array([-1, 0])
        )
        assert all(roe.clock_sequence[2][0].release_to_which_pixels == np.array([0, 1]))
        assert roe.clock_sequence[2][1].release_to_which_pixels == 0
        assert roe.clock_sequence[2][3].release_to_which_pixels == 0
        assert roe.clock_sequence[3][0].release_to_which_pixels == 1
        assert all(roe.clock_sequence[3][1].release_to_which_pixels == np.array([0, 1]))
        assert roe.clock_sequence[3][2].release_to_which_pixels == 0

        roe = ac.ROE([1] * n_phases, force_release_away_from_readout=True)
        assert roe.pixels_accessed_during_clocking == pytest.approx([0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == n_phases

        for step in range(n_phases):
            phase = step
            assert roe.clock_sequence[step][phase].is_high
            assert (
                roe.clock_sequence[step][phase].capture_from_which_pixels == 0
            ), "Step {}, phase {}, capture".format(step, phase)
            assert (
                roe.clock_sequence[step][phase].release_to_which_pixels == 0
            ), "Step {}, phase {}, release".format(step, phase)

        # Check other phases
        assert roe.clock_sequence[0][1].release_to_which_pixels == 1
        assert all(roe.clock_sequence[0][2].release_to_which_pixels == np.array([0, 1]))
        assert roe.clock_sequence[0][3].release_to_which_pixels == 0
        assert roe.clock_sequence[1][0].release_to_which_pixels == 0
        assert roe.clock_sequence[1][2].release_to_which_pixels == 1
        assert all(roe.clock_sequence[1][3].release_to_which_pixels == np.array([0, 1]))
        assert all(roe.clock_sequence[2][0].release_to_which_pixels == np.array([0, 1]))
        assert roe.clock_sequence[2][1].release_to_which_pixels == 0
        assert roe.clock_sequence[2][3].release_to_which_pixels == 1
        assert roe.clock_sequence[3][0].release_to_which_pixels == 1
        assert all(roe.clock_sequence[3][1].release_to_which_pixels == np.array([0, 1]))
        assert roe.clock_sequence[3][2].release_to_which_pixels == 0

    def test__trappumping_sequence_four_phase_single_phase_high(self):

        n_phases = 4
        roe = ac.ROETrapPumping([1] * 2 * n_phases)
        assert roe.pixels_accessed_during_clocking == pytest.approx([-1, 0, 1])
        assert roe.n_phases == n_phases
        assert roe.n_steps == 2 * n_phases

        for step in range(roe.n_steps):
            phase = ([0, 1, 2, 3, 0, 3, 2, 1])[step]
            assert roe.clock_sequence[step][phase].is_high


class TestChargeInjection:
    def test__charge_injection_express_matrix_and_monitor_traps_matrix(self):

        pixels = 12
        n_pixel_transfers = 12
        express = 0
        roe = ac.ROEChargeInjection(n_pixel_transfers=n_pixel_transfers)
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=pixels, express=express
        )
        assert express_matrix == pytest.approx(np.ones((n_pixel_transfers, pixels)))
        assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))

        express = 1
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=pixels, express=express
        )
        assert express_matrix == pytest.approx(np.ones((1, pixels)) * n_pixel_transfers)
        assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))

        n_pixel_transfers = 24
        express = 4
        roe = ac.ROEChargeInjection(n_pixel_transfers=n_pixel_transfers)
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=pixels, express=express
        )
        assert express_matrix == pytest.approx(
            np.ones((express, pixels)) * n_pixel_transfers / express
        )
        assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))

        express = 4
        offset = 6
        roe = ac.ROEChargeInjection(n_pixel_transfers=None)
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=pixels, express=express, offset=offset
        )
        assert express_matrix == pytest.approx(
            np.ones((express, pixels)) * (pixels + offset) / express
        )
        assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))


class TestTrapPumping:
    def test__charge_injection_express_matrix_and_monitor_traps_matrix(self):

        n_pumps = 12
        roe = ac.ROETrapPumping(dwell_times=[1] * 6, n_pumps=n_pumps)

        express = 0
        (
            express_matrix,
            monitor_traps_matrix,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=2, express=express
        )

        assert express_matrix == pytest.approx(np.ones((n_pumps, 1)))
        assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))

        for express in [1, 4, 7]:
            (
                express_matrix,
                monitor_traps_matrix,
            ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
                pixels=5, express=express
            )

            assert express_matrix == pytest.approx(
                np.transpose(
                    [np.append([1], np.ones(express) * (n_pumps - 1) / express)]
                )
            )
            assert monitor_traps_matrix == pytest.approx(np.ones_like(express_matrix))

    def test__save_trap_states_matrix(self):

        n_pumps = 12
        roe = ac.ROETrapPumping(dwell_times=[1] * 6, n_pumps=n_pumps)

        express = 0
        (
            express_matrix,
            _,
        ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
            pixels=2, express=express
        )
        save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
            express_matrix=express_matrix
        )
        assert save_trap_states_matrix == pytest.approx(
            np.transpose([np.append(np.ones(n_pumps - 1), [0])])
        )

        for express in [1, 4, 7]:
            (
                express_matrix,
                _,
            ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
                pixels=2, express=express
            )
            save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
                express_matrix=express_matrix
            )

            assert save_trap_states_matrix == pytest.approx(
                np.transpose([np.append(np.ones(express), [0])])
            )
