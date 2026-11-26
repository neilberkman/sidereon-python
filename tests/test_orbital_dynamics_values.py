import pytest
import sidereon as s


@pytest.mark.parametrize(
    "enum,members",
    [
        (
            s.AttitudeMode,
            "NOMINAL NOON_TURN MIDNIGHT_TURN SHADOW ORBIT_NORMAL MODIFIED_YAW_STEERING",
        ),
        (s.CosineTurnStartAngle, "COLINEARITY ORBIT_ANGLE"),
        (s.BeidouManufacturer, "CAST SECM"),
        (s.BeidouOrbitClass, "GEO IGSO MEO"),
        (s.IcgemErrors, "NO CALIBRATED FORMAL CALIBRATED_AND_FORMAL"),
        (s.IcgemFormat, "ICGEM1_0 ICGEM2_0"),
        (s.IcgemNormalization, "FULLY_NORMALIZED UNNORMALIZED"),
        (
            s.DeBody,
            "SUN MERCURY VENUS EARTH_MOON EARTH MOON MARS JUPITER "
            "SATURN URANUS NEPTUNE PLUTO SOLAR_SYSTEM",
        ),
        (s.AnalyticTheory, "ECKSTEIN_HECHLER BROUWER_LYDDANE"),
        (s.RtnAxis, "RADIAL ALONG_TRACK CROSS_TRACK"),
        (s.EmpiricalHarmonic, "CONSTANT COSINE SINE"),
        (s.ShadowFunction, "APPARENT_DISC CONICAL"),
        (s.ClockDatum, "EPOCH_SYSTEM_MEDIAN UNADJUSTED"),
        (s.OrbitScoreGapPolicy, "COUNT REFUSE"),
        (s.OrbitScoreReferencePoint, "CENTER_OF_MASS ANTENNA_PHASE_CENTER"),
        (s.OrbitRegime, "MEO IGSO GEO"),
        (s.ScoredProduct, "SOURCE REFERENCE"),
        (s.TriadVelocity, "INERTIAL EARTH_FIXED"),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else "members",
)
def test_orbital_enum_values(enum, members):
    values = [getattr(enum, member) for member in members.split()]
    assert [int(value) for value in values] == list(range(len(values)))
    for number, value in enumerate(values):
        assert value == number
        assert value != -1
        if type(value).__hash__ is not None:
            assert hash(value) == hash(number)
    if enum in {
        s.BeidouManufacturer,
        s.BeidouOrbitClass,
        s.RtnAxis,
        s.EmpiricalHarmonic,
    }:
        for left, right in zip(values, values[1:]):
            assert left < right
            assert left <= right
            assert right > left
            assert right >= left


def test_orbital_error_messages():
    cases = [
        (
            s.GnssAttitudeError,
            lambda: s.dyb_axes([0.0] * 3, [0.0, 1.0, 0.0]),
            "position_m has zero length",
        ),
        (
            s.GnssBlockError,
            lambda: s.GnssBlock.from_igs_name("bad"),
            'unknown GNSS block name "bad"',
        ),
        (
            s.DeError,
            lambda: s.DeEphemeris.from_bytes(b""),
            "Invalid DE ephemeris: incomplete first header record",
        ),
        (
            s.IcgemError,
            lambda: s.IcgemGravityField.parse(""),
            "ICGEM file has no end_of_head line",
        ),
        (
            s.AnalyticPropagationError,
            lambda: s.CircularElements(7000.0, 0.0, 0.0, 0.7, 0.0, 0.0).to_cartesian(
                0.0
            ),
            "invalid mu_km3_s2: must be positive",
        ),
        (
            s.OrbitParameterFitError,
            lambda: s.fit_inertial_orbit_parameters(
                [], "G01", None, s.OrbitParameterFitOptions()
            ),
            "satellite G01 has 0 samples; need at least 2",
        ),
        (
            s.OrbitScoreError,
            lambda: s.rtn_components(
                [0.0] * 3, [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]
            ),
            "invalid orbit scoring velocity_m_s: not finite",
        ),
    ]
    for error, evaluate, message in cases:
        with pytest.raises(error) as caught:
            evaluate()
        assert str(caught.value) == message
