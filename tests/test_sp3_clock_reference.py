import math
import textwrap

import numpy as np
import sidereon


def _sp3(clocks_us):
    body = textwrap.dedent(
        f"""\
        #cP2020  6 25  0  0  0.00000000       1 ORBIT TEST0 FIT  TST
        ## 2111 432000.00000000   900.00000000 59025 0.0000000000000
        +    3   G01G02G03  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
        %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
        %f  1.2500000  1.025000000  0.00000000000  0.000000000000000
        %f  0.0000000  0.000000000  0.00000000000  0.000000000000000
        %i    0    0    0    0      0      0      0      0         0
        %i    0    0    0    0      0      0      0      0         0
        /* TEST SP3-c FIXTURE
        *  2020  6 25  0  0  0.00000000
        PG01  15000.000000 -20000.000000   5000.000000 {clocks_us[0]:13.6f}
        PG02  -1234.567890   2345.678901  -3456.789012 {clocks_us[1]:13.6f}
        PG03   8000.000000  12000.000000 -19000.000000 {clocks_us[2]:13.6f}
        EOF
        """
    )
    return sidereon.load_sp3(body.encode())


def test_sp3_clock_reference_offset_and_align():
    reference = _sp3([100.0, 200.0, 300.0])
    other = _sp3([150.0, 250.0, 350.0])

    offsets = sidereon.sp3_clock_reference_offset(reference, other)

    assert len(offsets) == 1
    assert offsets[0].satellites == 3
    assert math.isfinite(offsets[0].epoch_j2000_seconds)
    assert abs(offsets[0].offset_s - 5.0e-5) < 1.0e-12

    aligned = sidereon.align_sp3_clock_reference(reference, other)

    for sat in ("G01", "G02", "G03"):
        ref_state = reference.state(sat, 0)
        other_state = other.state(sat, 0)
        aligned_state = aligned.state(sat, 0)
        assert abs(aligned_state.clock_s - ref_state.clock_s) < 1.0e-15
        np.testing.assert_allclose(
            aligned_state.position_m, other_state.position_m, rtol=0.0, atol=0.0
        )
