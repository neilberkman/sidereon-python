import datetime as dt

import pytest
from sidereon import data
from sidereon.distribution import ProductIdentity

UTC = dt.timezone.utc


def test_next_ultra_issue_maps_identity_due_time_and_split_coverage():
    issue = data.next_issue_due(
        "igs_ult", "sp3", dt.datetime(2026, 8, 4, 2, 59, 59, tzinfo=UTC)
    )

    assert isinstance(issue, data.NominalIssue)
    assert isinstance(issue.identity, ProductIdentity)
    assert issue.identity.analysis_center == "igs_ult"
    assert issue.identity.date == dt.date(2026, 8, 3)
    assert issue.identity.issue == "0000"
    assert issue.due_at == dt.datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    assert issue.covers == {
        "observed": {
            "from": dt.datetime(2026, 8, 3, tzinfo=UTC),
            "until": dt.datetime(2026, 8, 4, tzinfo=UTC),
        },
        "predicted": {
            "from": dt.datetime(2026, 8, 4, tzinfo=UTC),
            "until": dt.datetime(2026, 8, 5, tzinfo=UTC),
        },
    }


def test_next_issue_due_normalizes_offsets_and_fractional_seconds():
    local = dt.timezone(dt.timedelta(hours=-7))
    issue = data.next_issue_due(
        "igs_ult",
        "sp3",
        dt.datetime(2026, 8, 3, 20, 0, 0, 1, tzinfo=local),
    )

    assert issue.identity.issue == "0600"
    assert issue.due_at == dt.datetime(2026, 8, 4, 9, 0, tzinfo=UTC)


def test_next_final_issue_advances_over_the_gps_week_rollover():
    issue = data.next_issue_due("igs", "sp3", dt.datetime(2026, 8, 22, tzinfo=UTC))

    assert issue.identity.date == dt.date(2026, 8, 15)
    assert issue.due_at == dt.datetime(2026, 8, 28, 23, 59, 59, tzinfo=UTC)
    assert issue.covers["observed"] == {
        "from": dt.datetime(2026, 8, 15, tzinfo=UTC),
        "until": dt.datetime(2026, 8, 16, tzinfo=UTC),
    }
    assert issue.covers["predicted"] is None


def test_next_issue_due_reports_unsupported_schedule():
    with pytest.raises(data.UnsupportedProduct, match="nominal due-time schedule"):
        data.next_issue_due("wum_nrt", "sp3", dt.datetime(2026, 8, 4, tzinfo=UTC))
