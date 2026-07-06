"""0.19 track-filter and smoother binding parity."""

import numpy as np
import sidereon


def _arr(values):
    return np.asarray(values, dtype=np.float64)


def _mat(values):
    return np.asarray(values, dtype=np.float64)


def test_track_filter_gated_update_suppresses_position_spike():
    filt = sidereon.TrackFilter(
        sidereon.TrackFilterConfig(
            sidereon.TrackCoordinateFrame.CALLER_DEFINED_CARTESIAN,
            0.0,
            _arr([0.0]),
            _arr([1.0]),
            _mat([[1.0, 0.0], [0.0, 1.0]]),
            0.1,
        )
    )
    history = sidereon.TrackRtsHistoryBuilder.from_filter(filt)

    prediction = filt.predict_recorded(1.0, history)
    predicted_position = prediction.predicted.position_m.copy()
    predicted_covariance = prediction.predicted.covariance.copy()

    spike = _arr([100.0])
    spike_covariance = _mat([[0.01]])
    innovation = filt.position_innovation(spike, spike_covariance)
    assert not innovation.gate(0.95).in_gate

    gated = filt.update_position_gated_recorded(spike, spike_covariance, 0.95, history)
    assert not gated.gate.in_gate
    assert gated.update is None
    assert np.array_equal(gated.state.position_m, predicted_position)
    assert np.array_equal(gated.state.covariance, predicted_covariance)
    assert np.array_equal(filt.state.position_m, predicted_position)

    recorded = history.finish()
    assert len(recorded) == len(recorded.epochs)
    assert np.array_equal(recorded.epochs[-1].predicted.position_m, predicted_position)
    assert np.array_equal(recorded.epochs[-1].updated.position_m, predicted_position)

    smoothed = sidereon.smooth_track_rts(recorded)
    assert isinstance(smoothed, sidereon.SmoothedTrack)
    assert len(smoothed) == len(recorded)
    assert np.array_equal(smoothed.epochs[-1].state.position_m, predicted_position)


def test_track_filter_accepts_fix_covariance_and_smooths_recorded_flow():
    filt = sidereon.TrackFilter.from_position(
        sidereon.TrackCoordinateFrame.ECEF,
        0.0,
        _arr([0.0, 0.0, 0.0]),
        np.eye(3, dtype=np.float64),
        25.0,
        0.05,
    )
    history = sidereon.TrackRtsHistoryBuilder.from_filter(filt)

    filt.predict_recorded(1.0, history)
    update = filt.update_position_recorded(
        _arr([1.0, 0.0, 0.0]), np.eye(3, dtype=np.float64) * 0.25, history
    )
    assert isinstance(update, sidereon.TrackUpdate)
    assert update.updated.frame == sidereon.TrackCoordinateFrame.ECEF
    assert update.innovation.nis >= 0.0
    assert update.kalman_gain.shape == (6, 3)
    assert update.updated.position_m[0] > update.predicted.position_m[0]

    smoothed = sidereon.smooth_track_rts(history.finish())
    assert len(smoothed.epochs) == len(smoothed)
    assert smoothed.epochs[0].rts_gain_to_next.shape == (6, 6)
    assert smoothed.epochs[-1].rts_gain_to_next is None
    assert np.all(np.isfinite(smoothed.epochs[0].state.covariance))


def test_force_components_expose_solid_earth_tide_options():
    components = sidereon.ForceModelComponents(
        two_body_mu_km3_s2=398600.4418,
        spherical_harmonic_max_degree=2,
        spherical_harmonic_max_order=0,
        third_body=True,
        solid_earth_tide=True,
        solid_earth_pole_tide=True,
        relativity=True,
    )

    assert components.spherical_harmonic_max_degree == 2
    assert components.spherical_harmonic_max_order == 0
    assert components.solid_earth_tide
    assert components.solid_earth_pole_tide
    assert sidereon.ForceModelKind.composite(components).label == "composite"
