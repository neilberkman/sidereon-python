import sidereon

FIRST_EPOCH = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n"
FINAL_EPOCH = b"$GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"


def test_finish_with_output_preserves_final_sentence_epochs_and_warning():
    accumulator = sidereon.NmeaAccumulator()
    first = accumulator.push_bytes(FIRST_EPOCH)
    assert len(first.sentences) == 1
    assert first.snapshots == []

    pending = accumulator.push_bytes(FINAL_EPOCH)
    assert pending.sentences == []
    output = accumulator.finish_with_output()

    assert [sentence.kind for sentence in output.sentences] == ["GGA"]
    assert [snapshot.time for snapshot in output.snapshots] == [
        (12, 35, 19, 0, 0),
        (12, 35, 20, 0, 0),
    ]
    assert output.diagnostics.skip_count == 0
    assert output.diagnostics.warning_count == 1
    assert output.diagnostics.warnings[0].at.line == 2
    assert output.diagnostics.warnings[0].kind.label == "missing_metadata"

    repeated = accumulator.finish_with_output()
    assert repeated.sentences == []
    assert repeated.snapshots == []
    assert repeated.diagnostics.is_empty()


def test_finish_with_output_preserves_malformed_final_line_skip():
    accumulator = sidereon.NmeaAccumulator()
    assert accumulator.push_bytes(b"bad").sentences == []

    output = accumulator.finish_with_output()

    assert output.sentences == []
    assert output.snapshots == []
    assert output.diagnostics.skip_count == 1
    assert output.diagnostics.skips[0].at.line == 1
