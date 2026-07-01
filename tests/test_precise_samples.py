"""PreciseEphemerisSamples binding parity tests.

Mirror the core sample-vs-SP3 rigor: a source rebuilt from an SP3 product's
extracted samples interpolates and predicts ranges identically (within the
documented round-trip tolerance) to the SP3-parsed source, the from_samples
validation failures raise ``ValueError``, and the ``predict_ranges`` batch
equals the per-request single call bit-for-bit."""

import os

import numpy as np
import pytest
import sidereon

try:
    from _helpers import CORE_FIXTURES

    SP3_PATH = os.path.join(
        CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    )
    _HAVE_SP3 = os.path.exists(SP3_PATH)
except Exception:
    _HAVE_SP3 = False

pytestmark = pytest.mark.skipif(
    not _HAVE_SP3, reason="core SP3 fixture unavailable (set SIDEREON_CORE_FIXTURES)"
)

# A spread of static ECEF receivers, matching the core parity grid.
RECEIVERS = [
    [4027894.0, 307046.0, 4919474.0],
    [1130000.0, -4830000.0, 3994000.0],
    [-2700000.0, -4290000.0, 3855000.0],
]


def _sp3():
    with open(SP3_PATH, "rb") as handle:
        return sidereon.load_sp3(handle.read())


def _query_grid(sp3):
    """Node epochs plus interior midpoints."""
    epochs = sp3.epochs_j2000_seconds
    queries = []
    for a, b in zip(epochs[:-1], epochs[1:]):
        queries.append(float(a))
        queries.append(0.5 * (float(a) + float(b)))
    queries.append(float(epochs[-1]))
    return queries


def _bits_equal(a, b):
    """True when two numpy float64 arrays are byte-for-byte identical."""
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_extracted_samples_round_trip_shape():
    sp3 = _sp3()
    samples = sp3.precise_ephemeris_samples()
    assert len(samples) > 0
    s0 = samples[0]
    assert s0.satellite in sp3.satellites
    assert s0.time_scale == sidereon.TimeScale.GPST
    assert s0.position_ecef_m.shape == (3,)

    source = sidereon.PreciseEphemerisSamples.from_samples(samples)
    assert source.time_scale == sidereon.TimeScale.GPST
    # Every extracted satellite is interpolatable by the rebuilt source.
    assert set(source.satellites) == set(sp3.satellites)


def test_from_samples_matches_sp3_across_grid():
    """The rebuilt source's interpolated states (raw, light-time / Sagnac off)
    and full predicted ranges must match the SP3-parsed source across a grid,
    byte-for-byte for the vast majority and within sub-micron everywhere (the
    documented km->m reconstruction bound on a real product)."""
    sp3 = _sp3()
    source = sidereon.PreciseEphemerisSamples.from_samples(
        sp3.precise_ephemeris_samples()
    )
    sats = list(sp3.satellites)[:8]
    # Interior queries only, so the 11-node interpolation window is always
    # populated for both sources and the batch never aborts on a coverage gap.
    queries = _query_grid(sp3)[12:80]

    compared = 0
    byte_identical = 0
    max_abs_diff_m = 0.0
    for sat in sats:
        for rx in RECEIVERS:
            for opts in (dict(light_time=False, sagnac=False), dict()):
                reqs = [sidereon.RangePredictionRequest(sat, rx, q) for q in queries]
                a = sidereon.predict_ranges(sp3, reqs, **opts)
                b = sidereon.predict_ranges(source, reqs, **opts)
                assert len(a) == len(b) == len(reqs)
                for ra, rb in zip(a, b):
                    # Range and transmit time agree byte-for-byte.
                    if ra.geometric_range_m == rb.geometric_range_m:
                        byte_identical += 1
                    compared += 1
                    max_abs_diff_m = max(
                        max_abs_diff_m,
                        abs(ra.geometric_range_m - rb.geometric_range_m),
                        float(np.max(np.abs(ra.sat_pos_ecef_m - rb.sat_pos_ecef_m))),
                    )
                    assert ra.transmit_time_j2000_s == rb.transmit_time_j2000_s
                    assert ra.sat_clock_s == rb.sat_clock_s

    assert compared > 0
    assert max_abs_diff_m < 1.0e-6, f"max divergence {max_abs_diff_m:e} m"
    assert byte_identical * 100 >= compared * 90, (
        f"expected the vast majority byte-identical, got {byte_identical}/{compared}"
    )


def test_batch_equals_per_request_single_call():
    """The batch is amortization of the call boundary only: entry ``i`` must be
    bit-for-bit identical to a single-request call for request ``i``, on both an
    SP3 source and a rebuilt sample source."""
    sp3 = _sp3()
    source = sidereon.PreciseEphemerisSamples.from_samples(
        sp3.precise_ephemeris_samples()
    )
    sats = list(sp3.satellites)[:6]
    queries = _query_grid(sp3)[12:40]
    reqs = [
        sidereon.RangePredictionRequest(sat, RECEIVERS[i % len(RECEIVERS)], q)
        for i, sat in enumerate(sats)
        for q in queries
    ]

    for src in (sp3, source):
        batch = sidereon.predict_ranges(src, reqs)
        assert len(batch) == len(reqs)
        for req, got in zip(reqs, batch):
            (single,) = sidereon.predict_ranges(src, [req])
            assert got.geometric_range_m == single.geometric_range_m
            assert got.transmit_time_j2000_s == single.transmit_time_j2000_s
            assert got.sat_clock_s == single.sat_clock_s
            assert _bits_equal(got.sat_pos_ecef_m, single.sat_pos_ecef_m)


def _sample(prn, j2000_s, pos=(2.0e7, 1.4e7, 2.1e7), clock_s=1.0e-6, scale=None):
    kwargs = {} if scale is None else {"time_scale": scale}
    return sidereon.PreciseEphemerisSample(
        f"G{prn:02d}", j2000_s, list(pos), clock_s, **kwargs
    )


def test_from_samples_rejects_empty():
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples([])


def test_from_samples_rejects_single_sample_satellite():
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples([_sample(21, 0.0)])


def test_from_samples_rejects_non_monotonic_epochs():
    # Repeated epoch is not strictly increasing.
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples(
            [_sample(21, 900.0), _sample(21, 900.0)]
        )
    # Descending epochs.
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples(
            [_sample(7, 1800.0), _sample(7, 900.0)]
        )


def test_from_samples_rejects_mixed_time_scales():
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples(
            [
                _sample(21, 0.0, scale=sidereon.TimeScale.GPST),
                _sample(21, 900.0, scale=sidereon.TimeScale.UTC),
            ]
        )


def test_from_samples_rejects_non_finite_sample():
    with pytest.raises(ValueError):
        sidereon.PreciseEphemerisSamples.from_samples(
            [_sample(21, 0.0, pos=(np.nan, 2.0e7, 3.0e7)), _sample(21, 900.0)]
        )


def test_out_of_range_query_errors():
    """An out-of-coverage query through the shared predict_ranges hot path is a
    structured ephemeris failure (SolveError), matching the SP3 / observe_batch
    convention rather than being masked."""
    source = sidereon.PreciseEphemerisSamples.from_samples(
        [_sample(21, 0.0), _sample(21, 900.0)]
    )
    req = sidereon.RangePredictionRequest("G21", RECEIVERS[0], 1_000_000.0)
    with pytest.raises(sidereon.SolveError):
        sidereon.predict_ranges(source, [req])
