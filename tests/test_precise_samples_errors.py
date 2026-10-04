import numpy as np
import pytest
import sidereon


def precise_sample(satellite, epoch, position=(1.0, 2.0, 3.0), time_scale=None):
    options = {} if time_scale is None else {"time_scale": time_scale}
    return sidereon.PreciseEphemerisSample(satellite, epoch, position, **options)


def assert_sample_error(call, kind, satellite):
    with pytest.raises(sidereon.PreciseSamplesError) as captured:
        call()

    assert captured.value.kind == kind
    assert captured.value.satellite == satellite


@pytest.mark.parametrize(
    "constructor",
    [
        sidereon.PreciseEphemerisSamples.from_samples,
        sidereon.PreciseEphemerisInterpolant.from_samples,
    ],
)
def test_sample_constructors_preserve_all_core_validation_payloads(constructor):
    assert_sample_error(lambda: constructor([]), "empty", None)
    assert_sample_error(
        lambda: constructor([precise_sample("G03", 0.0)]),
        "single_sample_satellite",
        "G03",
    )
    assert_sample_error(
        lambda: constructor([precise_sample("G07", 10.0), precise_sample("G07", 0.0)]),
        "non_monotonic_epochs",
        "G07",
    )
    assert_sample_error(
        lambda: constructor(
            [
                precise_sample("G11", 0.0),
                precise_sample("G11", 1.0, time_scale=sidereon.TimeScale.UTC),
            ]
        ),
        "mixed_time_scales",
        None,
    )
    assert_sample_error(
        lambda: constructor(
            [
                precise_sample("G19", 0.0, position=(np.nan, 2.0, 3.0)),
                precise_sample("G19", 1.0),
            ]
        ),
        "non_finite_sample",
        "G19",
    )
    assert_sample_error(
        lambda: constructor(
            [
                sidereon.PreciseEphemerisSample.from_instant(
                    "G23",
                    sidereon.ClockInstant.from_split_julian_date(
                        sidereon.TimeScale.GPST,
                        float.fromhex("0x1.fffffffffffffp+1023"),
                        0.0,
                    ),
                    [1.0, 2.0, 3.0],
                )
            ]
        ),
        "epoch_not_representable",
        "G23",
    )


@pytest.mark.parametrize(
    "constructor",
    [
        sidereon.PreciseEphemerisSamples.from_samples_with_accuracy,
        sidereon.PreciseEphemerisInterpolant.from_samples_with_accuracy,
    ],
)
def test_accuracy_constructors_preserve_typed_sample_errors(constructor):
    samples = [precise_sample("G02", 0.0), precise_sample("G02", 1.0)]
    unknown = sidereon.Sp3AccuracyValue.unknown()
    mismatched_accuracy = [
        sidereon.PreciseEphemerisAccuracySample.new(
            "G05", epoch, [unknown, unknown, unknown], unknown
        )
        for epoch in (0.0, 1.0)
    ]

    with pytest.raises(sidereon.AccuracySamplesMismatchError) as captured:
        constructor(samples, mismatched_accuracy)

    assert isinstance(captured.value, sidereon.PreciseSamplesError)
    assert captured.value.kind == "accuracy_samples_mismatch"
    assert captured.value.satellite is None
