"""TCA finding, conjunction Pc, and catalog screening delegate to ``..::astro::tca``.

The binding marshals TLE strings, the two-part Julian-date window, finder
tolerances, and Pc options into the core finders and packages the results.
"""

import numpy as np
import pytest
import sidereon

# A committed ISS TLE (also used by the propagation tests).
PRIMARY_L1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
PRIMARY_L2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"

# Search window bracketing the TLE epoch (JD 2458303.31), as (whole, fraction).
WINDOW_START = (2458303.0, 0.0)
WINDOW_END = (2458304.0, 0.0)


def _fix_checksum(line: str) -> str:
    body = line[:68]
    total = 0
    for ch in body:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return body + str(total % 10)


def _secondary():
    # A distinct orbit plane and phase so the pair has real close approaches and a
    # non-zero relative velocity; recompute the line-2 checksum after editing.
    l2 = PRIMARY_L2.replace("295.8524", "100.0000").replace("204.2868", "010.0000")
    return PRIMARY_L1, _fix_checksum(l2)


def test_find_tca_candidates_well_formed():
    sl1, sl2 = _secondary()
    candidates = sidereon.find_tca_candidates(
        PRIMARY_L1,
        PRIMARY_L2,
        sl1,
        sl2,
        WINDOW_START,
        WINDOW_END,
        coarse_step_seconds=60.0,
    )
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    for c in candidates:
        assert c.miss_distance_km >= 0.0
        assert np.isfinite(c.miss_distance_km)
        assert 0.0 <= c.tca_seconds_since_window_start <= 86400.0
        assert c.relative_position_km.shape == (3,)
        assert c.relative_velocity_km_s.shape == (3,)
        assert c.tca_time_jd == pytest.approx(
            c.tca_time_jd_whole + c.tca_time_jd_fraction
        )


def test_find_tca_conjunctions_pc_in_range():
    sl1, sl2 = _secondary()
    conjunctions = sidereon.find_tca_conjunctions(
        PRIMARY_L1,
        PRIMARY_L2,
        sl1,
        sl2,
        WINDOW_START,
        WINDOW_END,
        hard_body_radius_km=0.02,
        method=sidereon.PcMethod.FOSTER_EQUAL_AREA,
    )
    assert len(conjunctions) > 0
    for cj in conjunctions:
        assert 0.0 <= cj.pc <= 1.0
        assert cj.miss_km >= 0.0
        assert cj.candidate.miss_distance_km >= 0.0


def test_screen_tca_candidates_returns_hits():
    sl1, sl2 = _secondary()
    hits = sidereon.screen_tca_candidates(
        PRIMARY_L1,
        PRIMARY_L2,
        [(sl1, sl2)],
        WINDOW_START,
        WINDOW_END,
        miss_distance_threshold_km=1.0e9,
    )
    assert len(hits) > 0
    for hit in hits:
        assert hit.secondary_index == 0
        assert hit.candidate.miss_distance_km <= 1.0e9


def test_screen_tca_conjunctions_returns_hits():
    sl1, sl2 = _secondary()
    hits = sidereon.screen_tca_conjunctions(
        PRIMARY_L1,
        PRIMARY_L2,
        [(sl1, sl2)],
        WINDOW_START,
        WINDOW_END,
        miss_distance_threshold_km=1.0e9,
        hard_body_radius_km=0.02,
    )
    assert len(hits) > 0
    for hit in hits:
        assert hit.secondary_index == 0
        assert 0.0 <= hit.conjunction.pc <= 1.0
