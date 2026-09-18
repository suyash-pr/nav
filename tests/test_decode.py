import numpy as np

from nav.session.decode import dequantise_depth, invalid_threshold


def test_invalid_threshold_matches_karter_formula():
    # karter: max(1, MIN_PLAUSIBLE_DEPTH_MM // step_mm), MIN_PLAUSIBLE_DEPTH_MM = 200
    assert invalid_threshold(16) == 12  # 200 // 16
    assert invalid_threshold(8) == 25  # 200 // 8
    assert invalid_threshold(1000) == 1  # max(1, ...) guards a step larger than the guard band


def test_dequantise_depth_guard_band_collapses_to_zero():
    step_mm = 16
    threshold = invalid_threshold(step_mm)  # 12
    units = np.array([0, threshold - 1], dtype=np.uint16)
    depth_mm = dequantise_depth(units, step_mm)
    np.testing.assert_array_equal(depth_mm, [0, 0])


def test_dequantise_depth_mid_range_is_unit_times_step():
    step_mm = 16
    threshold = invalid_threshold(step_mm)
    units = np.array([threshold, threshold + 1, 100], dtype=np.uint16)
    depth_mm = dequantise_depth(units, step_mm)
    np.testing.assert_array_equal(depth_mm, units.astype(np.int64) * step_mm)


def test_dequantise_depth_horizon_5m_step_16mm():
    # arch cameras: horizon_m=5.0, step_mm=16 -> horizon_units = round(5000/16) = 313
    step_mm = 16
    horizon_units = round(5000 / step_mm)
    depth_mm = dequantise_depth(np.array([horizon_units], dtype=np.uint16), step_mm)
    assert depth_mm[0] == horizon_units * step_mm


def test_dequantise_depth_far_sentinel_stays_large_not_wrapped():
    # far_sentinel = units + max(4, units // 8), always > MAX_TEN_BIT-adjacent range but still
    # a plausible-looking large distance, never zero/invalid.
    step_mm = 8
    horizon_units = round(2000 / step_mm)  # blind-spot: horizon_m=2.0
    far_sentinel = horizon_units + max(4, horizon_units // 8)
    depth_mm = dequantise_depth(np.array([far_sentinel], dtype=np.uint16), step_mm)
    assert depth_mm[0] > horizon_units * step_mm
    assert depth_mm[0] != 0


def test_dequantise_depth_rejects_non_uint16():
    import pytest

    with pytest.raises(TypeError):
        dequantise_depth(np.array([1, 2], dtype=np.int32), 16)
