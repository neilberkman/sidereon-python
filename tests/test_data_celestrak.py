"""CelesTrak and NAVCEN catalog and transport contracts."""

import httpx
import pytest
import sidereon
import sidereon.data as data


def test_gp_catalog_urls_and_selectors():
    root = "https://celestrak.org/NORAD/elements/gp.php"
    cases = [
        ("group", "gps-ops", "GROUP=gps-ops"),
        ("catalog_number", 123456, "CATNR=123456"),
        ("international_designator", "2020-025", "INTDES=2020-025"),
        ("name", "COSMOS 2251 DEB", "NAME=COSMOS%202251%20DEB"),
        ("special", data.CelestrakGpSpecial.GPZ, "SPECIAL=GPZ"),
        ("special", data.CelestrakGpSpecial.GPZ_PLUS, "SPECIAL=GPZ-PLUS"),
        ("special", data.CelestrakGpSpecial.DECAYING, "SPECIAL=DECAYING"),
    ]
    for kind, value, expected in cases:
        assert (
            data.celestrak_gp_url(
                data.CelestrakGpQuery(kind, value), data.CelestrakGpFormat.CSV
            )
            == f"{root}?{expected}&FORMAT=CSV"
        )
    for format, wire in [
        (data.CelestrakGpFormat.TLE, "TLE"),
        (data.CelestrakGpFormat.THREE_LINE, "3LE"),
        (data.CelestrakGpFormat.TWO_LINE, "2LE"),
        (data.CelestrakGpFormat.XML, "XML"),
        (data.CelestrakGpFormat.KVN, "KVN"),
        (data.CelestrakGpFormat.JSON, "JSON"),
        (data.CelestrakGpFormat.JSON_PRETTY, "JSON-PRETTY"),
        (data.CelestrakGpFormat.CSV, "CSV"),
    ]:
        assert format.as_str() == wire
        assert (
            data.celestrak_gp_url(
                data.CelestrakGpQuery("catalog_number", 25544), format
            )
            == f"{root}?CATNR=25544&FORMAT={wire}"
        )
    assert data.CelestrakGpSpecial.GPZ.as_str() == "GPZ"
    assert data.CelestrakGpSpecial.GPZ_PLUS.as_str() == "GPZ-PLUS"
    assert data.CelestrakGpSpecial.DECAYING.as_str() == "DECAYING"
    with pytest.raises(data.InvalidCelestrakGpSelector) as exc:
        data.celestrak_gp_url(
            data.CelestrakGpQuery("group", "bad group"), data.CelestrakGpFormat.TLE
        )
    assert exc.value.detail["kind"] == "invalid_celestrak_gp_selector"
    assert exc.value.detail["selector"] == "GROUP"
    assert exc.value.detail["value"] == "bad group"
    assert data.celestrak_gp_source_entry().host == "celestrak.org"
    assert data.celestrak_gp_source_entry() == data.CelestrakGpSourceEntry(
        "https", "celestrak.org", root
    )
    assert data.navcen_source_entry().host == "www.navcen.uscg.gov"
    assert data.navcen_source_entry() == data.NavcenSourceEntry(
        "https", "www.navcen.uscg.gov", "https://www.navcen.uscg.gov/gps-constellation"
    )
    assert (
        data.navcen_constellation_url()
        == "https://www.navcen.uscg.gov/gps-constellation"
    )
    assert "www.navcen.uscg.gov" in data.allowed_hosts()
    assert data.celestrak_group(sidereon.GnssSystem.SBAS) == "sbas"
    assert (
        data.celestrak_constellation_url(
            sidereon.GnssSystem.SBAS, data.CelestrakGpFormat.JSON
        )
        == f"{root}?GROUP=sbas&FORMAT=JSON"
    )


def test_navcen_fetch_cache_and_offline(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, content=b"<html>status</html>", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        assert (
            data.fetch_navcen_constellation_status(cache_dir=str(tmp_path))
            == b"<html>status</html>"
        )
        assert (
            data.fetch_navcen_constellation_status(
                cache_dir=str(tmp_path), offline=True
            )
            == b"<html>status</html>"
        )
    assert requests == ["https://www.navcen.uscg.gov/gps-constellation"]
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_navcen_constellation_status(
            cache_dir=str(tmp_path / "empty"), offline=True
        )


def test_gp_fixture_records_rejections_and_response_errors():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    root = Path(CORE_FIXTURES)
    raw = (root / "omm/25544.tle").read_bytes()
    response = data.parse_celestrak_gp_response(raw, data.CelestrakGpFormat.TLE)
    assert not response.rejected
    record = response.records[0]
    assert record.record_index == 1
    assert record.line_number == 1
    assert record.raw == raw
    assert record.object_name == "ISS (ZARYA)"
    assert record.payload.elements.catalog_number == "25544"
    elements = record.payload.elements
    assert {
        name: getattr(elements, name)
        for name in (
            "catalog_number",
            "classification",
            "international_designator",
            "epoch_year",
            "epoch_day_of_year",
            "mean_motion_dot",
            "mean_motion_double_dot",
            "mean_motion_double_dot_text",
            "bstar",
            "bstar_text",
            "ephemeris_type",
            "elset_number",
            "inclination_deg",
            "raan_deg",
            "eccentricity",
            "arg_perigee_deg",
            "mean_anomaly_deg",
            "mean_motion",
            "rev_number",
        )
    } == {
        "catalog_number": "25544",
        "classification": "U",
        "international_designator": "98067A",
        "epoch_year": 2026,
        "epoch_day_of_year": 168.18949189,
        "mean_motion_dot": 0.00009113,
        "mean_motion_double_dot": 0.0,
        "mean_motion_double_dot_text": " 00000+0",
        "bstar": 0.00017172000000000002,
        "bstar_text": " 17172-3",
        "ephemeris_type": 0,
        "elset_number": 999,
        "inclination_deg": 51.6332,
        "raan_deg": 300.0813,
        "eccentricity": 0.0004737,
        "arg_perigee_deg": 195.1146,
        "mean_anomaly_deg": 164.9702,
        "mean_motion": 15.49273435,
        "rev_number": 57175,
    }
    sgp4 = record.to_element_set()
    assert sgp4.epoch == (2461208.5, 0.18949189)
    assert sgp4.bstar == 0.00017172000000000002
    assert sgp4.mean_motion_dot == 0.00009113
    assert sgp4.mean_motion_double_dot == 0.0
    assert sgp4.eccentricity == 0.0004737
    assert sgp4.argument_of_perigee_deg == 195.1146
    assert sgp4.inclination_deg == 51.6332
    assert sgp4.mean_anomaly_deg == 164.9702
    assert sgp4.mean_motion_rev_per_day == 15.49273435
    assert sgp4.right_ascension_deg == 300.0813
    assert sgp4.catalog_number == 25544
    assert sgp4.omm_epoch_days is None
    assert record.to_omm().object_name == "ISS (ZARYA)"
    assert (
        response.records[0]
        != data.parse_celestrak_gp_response(
            (root / "omm/24876.tle").read_bytes(), data.CelestrakGpFormat.TLE
        ).records[0]
    )
    mixed = data.parse_celestrak_gp_response(
        raw + b"STRAY\n" + raw, data.CelestrakGpFormat.TLE
    )
    assert len(mixed.records) == 2
    assert len(mixed.rejected) == 1
    assert mixed.rejected[0].raw == b"STRAY\n"
    assert mixed.rejected[0].reason.kind == "structural"
    assert mixed.rejected[0].reason.detail["kind"] == "invalid_tle_structure"
    with pytest.raises(data.CelestrakGpNotModified) as exc:
        data.parse_celestrak_gp_response(
            (root / "celestrak/repeat_download.txt").read_bytes(),
            data.CelestrakGpFormat.JSON,
        )
    assert exc.value.query == "GROUP=active"
    assert exc.value.last_successful_download == "2026-03-26 08:10:22 UTC"
    with pytest.raises(data.CelestrakGpDataNotFound):
        data.parse_celestrak_gp_response(
            b"No GP data found", data.CelestrakGpFormat.JSON
        )
    with pytest.raises(data.EmptyCelestrakGpResponse):
        data.parse_celestrak_gp_response(b" \r\n", data.CelestrakGpFormat.JSON)
    with pytest.raises(data.CelestrakGpHtmlResponse):
        data.parse_celestrak_gp_response(
            b"<!DOCTYPE html>", data.CelestrakGpFormat.JSON
        )
    with pytest.raises(data.InvalidCelestrakGpResponse) as invalid:
        data.parse_celestrak_gp_response(b"not json", data.CelestrakGpFormat.JSON)
    assert invalid.value.line == 1
    assert invalid.value.column == 2
    assert invalid.value.reason == (
        "malformed JSON: expected ident at line 1 column 2; "
        'response excerpt: "not json"'
    )


def test_gp_fetch_rate_limit_not_retried(tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    calls = []
    body = (Path(CORE_FIXTURES) / "celestrak/repeat_download.txt").read_bytes()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.CelestrakGpNotModified):
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery("group", "active"),
                cache_dir=str(tmp_path),
                retries=3,
            )
    assert len(calls) == 1


def test_gp_fetch_success_and_constellation(tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    calls = []
    body = (Path(CORE_FIXTURES) / "omm/25544.json").read_bytes()

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)

        def unexpected_group(_system):
            raise AssertionError("constellation URL must come from the core builder")

        monkeypatch.setattr(data, "celestrak_group", unexpected_group)
        response = data.fetch_constellation(
            sidereon.GnssSystem.GPS, cache_dir=str(tmp_path)
        )
        assert response.records[0].object_name == "ISS (ZARYA)"
        assert isinstance(response.records[0].payload, sidereon.Omm)
        assert response.records[0].raw == body
        assert response.records[0].to_element_set().catalog_number == 25544
        cached = data.fetch_constellation(
            sidereon.GnssSystem.GPS, cache_dir=str(tmp_path), offline=True
        )
        assert cached.records[0].raw == body
    assert calls == [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=JSON"
    ]


@pytest.mark.parametrize(
    ("format", "filename"),
    [
        (data.CelestrakGpFormat.TLE, "25544.tle"),
        (data.CelestrakGpFormat.KVN, "25544.kvn"),
        (data.CelestrakGpFormat.XML, "25544.xml"),
        (data.CelestrakGpFormat.JSON, "25544.json"),
        (data.CelestrakGpFormat.JSON_PRETTY, "25544.json"),
    ],
)
def test_gp_formats_preserve_fixture_records(format, filename):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    raw = (Path(CORE_FIXTURES) / "omm" / filename).read_bytes()
    response = data.parse_celestrak_gp_response(raw, format)
    assert len(response.records) == 1
    assert response.rejected == []
    assert response.records[0].raw == raw
    assert response.records[0].object_name == "ISS (ZARYA)"
    assert response.records[0].to_omm().norad_cat_id == 25544


def test_gp_tle_layouts_and_csv():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    raw = (Path(CORE_FIXTURES) / "omm/25544.tle").read_bytes()
    name, line1, line2 = raw.splitlines()
    two = line1 + b"\n" + line2 + b"\n"
    three = b"0 " + name.strip() + b"\n" + two
    record = data.parse_celestrak_gp_response(
        two, data.CelestrakGpFormat.TWO_LINE
    ).records[0]
    assert record.raw == two
    assert record.object_name is None
    record = data.parse_celestrak_gp_response(
        three, data.CelestrakGpFormat.THREE_LINE
    ).records[0]
    assert record.raw == three
    assert record.object_name == "ISS (ZARYA)"
    csv = (
        b"OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,"
        b"RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,NORAD_CAT_ID,BSTAR\n"
        b"ISS,1998-067A,2026-06-17T04:32:52.099296,15.49273435,.0004737,"
        b"51.6332,300.0813,195.1146,164.9702,25544,.00017172\n"
    )
    record = data.parse_celestrak_gp_response(csv, data.CelestrakGpFormat.CSV).records[
        0
    ]
    assert record.object_name == "ISS"
    assert record.to_element_set().catalog_number == 25544
    assert record.raw == csv


def test_gp_stations_fixture_byte_accounting():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    source = (Path(CORE_FIXTURES) / "celestrak/stations.tle").read_bytes()
    response = data.parse_celestrak_gp_response(source, data.CelestrakGpFormat.TLE)
    assert len(response.records) == 31
    assert response.rejected == []
    all_records = sorted(
        response.records + response.rejected, key=lambda item: item.record_index
    )
    assert b"".join(record.raw for record in all_records) == source
    assert response.records[0].object_name == "ISS (ZARYA)"
    assert response.records[0].record_index == 1
    assert response.records[0].line_number == 1
    assert response.records[0].payload.elements.catalog_number == "25544"


def test_gp_rejected_tle_error_details():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    raw = (Path(CORE_FIXTURES) / "omm/25544.tle").read_bytes()
    lines = raw.splitlines(keepends=True)
    line = lines[1]
    altered = line[:68] + (b"0" if line[68:69] != b"0" else b"1") + line[69:]
    broken = lines[0] + altered + lines[2]
    response = data.parse_celestrak_gp_response(broken, data.CelestrakGpFormat.TLE)
    assert response.records == []
    rejected = response.rejected[0]
    assert rejected.raw == broken
    assert rejected.reason.kind == "tle"
    assert rejected.reason.detail["kind"] == "checksum_mismatch"
    assert rejected.reason.detail["line_label"] == "line 1"
    assert rejected.reason.detail["computed"] == 6


def test_gp_tle_checksum_warning_is_retained():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    name, line1, line2 = (
        (Path(CORE_FIXTURES) / "omm/25544.tle").read_bytes().splitlines()
    )
    raw = b"\n".join((name, line1[:68], line2)) + b"\n"
    record = data.parse_celestrak_gp_response(raw, data.CelestrakGpFormat.TLE).records[
        0
    ]
    assert record.raw == raw
    assert record.payload.line1 == line1[:68].decode("ascii")
    warning = record.payload.checksum_warnings[0]
    assert warning.line_label == "line 1"
    assert warning.kind == "missing"
    assert warning.computed == 6
    assert warning.expected is None
    assert warning.found is None


def test_gp_rejected_omm_retains_detail_and_source():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    source = (Path(CORE_FIXTURES) / "omm/25544.json").read_bytes().strip()
    response = data.parse_celestrak_gp_response(
        source[:-1] + b",{}]", data.CelestrakGpFormat.JSON
    )
    assert len(response.records) == 1
    rejected = response.rejected[0]
    assert rejected.record_index == 2
    assert rejected.line_number == 1
    assert rejected.raw == b"{}]"
    assert rejected.reason.kind == "omm"
    assert rejected.reason.detail == {
        "family": "omm",
        "field": "EPOCH",
        "kind": "MissingField",
    }


def test_gp_structural_rejection_variants():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    raw = (Path(CORE_FIXTURES) / "omm/25544.tle").read_bytes()
    line1, line2 = raw.splitlines()[1:]
    two = line1 + b"\n" + line2 + b"\n"
    response = data.parse_celestrak_gp_response(
        b"junk\n" + two, data.CelestrakGpFormat.TWO_LINE
    )
    assert response.rejected[0].raw == b"junk\n"
    assert response.rejected[0].reason.detail == {
        "family": "CelestrakGpStructuralError",
        "kind": "unexpected_content",
        "message": "UnexpectedContent",
    }
    response = data.parse_celestrak_gp_response(
        two + line1 + b"\n", data.CelestrakGpFormat.TWO_LINE
    )
    assert response.rejected[0].raw == line1 + b"\n"
    assert response.rejected[0].reason.detail == {
        "family": "CelestrakGpStructuralError",
        "kind": "incomplete_record",
        "message": "IncompleteRecord",
    }
    csv = (
        b"OBJECT_NAME,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
        b"ARG_OF_PERICENTER,MEAN_ANOMALY,NORAD_CAT_ID\n"
        b"ISS,2026-06-17T04:32:52,15.49273435,.0004737,51.6332,300.0813,"
        b"195.1146,164.9702,25544\n"
    )
    rejected = data.parse_celestrak_gp_response(
        csv, data.CelestrakGpFormat.CSV
    ).rejected[0]
    assert rejected.raw == csv
    assert rejected.reason.detail == {
        "family": "CelestrakGpStructuralError",
        "kind": "invalid_element_set",
        "message": 'InvalidElementSet(InvalidInput { field: "BSTAR", kind: Missing })',
        "cause": {
            "family": "Sgp4Error",
            "kind": "invalid_input",
            "message": "invalid SGP4 input BSTAR: missing",
            "field": "BSTAR",
            "reason": "missing",
        },
    }


@pytest.mark.parametrize(
    ("system", "group"),
    [
        (sidereon.GnssSystem.GPS, "gps-ops"),
        (sidereon.GnssSystem.GLONASS, "glo-ops"),
        (sidereon.GnssSystem.GALILEO, "galileo"),
        (sidereon.GnssSystem.BEIDOU, "beidou"),
        (sidereon.GnssSystem.QZSS, "gnss"),
        (sidereon.GnssSystem.NAVIC, "gnss"),
        (sidereon.GnssSystem.SBAS, "sbas"),
    ],
)
def test_gp_constellation_group_values(system, group):
    assert data.celestrak_group(system) == group
    assert data.celestrak_constellation_url(system, data.CelestrakGpFormat.KVN) == (
        f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=KVN"
    )


@pytest.mark.parametrize(
    ("query", "selector", "value"),
    [
        (data.CelestrakGpQuery.catalog_number(0), "CATNR", "0"),
        (
            data.CelestrakGpQuery.international_designator("2020-02"),
            "INTDES",
            "2020-02",
        ),
        (data.CelestrakGpQuery.name("  "), "NAME", "  "),
        (data.CelestrakGpQuery.group("bad group"), "GROUP", "bad group"),
        (data.CelestrakGpQuery("group", 123), "GROUP", "123"),
        (data.CelestrakGpQuery("name", 123), "NAME", "123"),
        (data.CelestrakGpQuery("international_designator", 123), "INTDES", "123"),
    ],
)
def test_gp_refuses_invalid_selectors_with_detail(query, selector, value):
    with pytest.raises(data.InvalidCelestrakGpSelector) as exc:
        data.celestrak_gp_url(query, data.CelestrakGpFormat.JSON)
    assert exc.value.detail["kind"] == "invalid_celestrak_gp_selector"
    assert exc.value.detail["selector"] == selector
    assert exc.value.detail["value"] == value
    assert exc.value.selector == selector
    assert exc.value.value == value


def test_gp_source_bytes_preserve_unicode_name():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    source = (Path(CORE_FIXTURES) / "omm/25544.tle").read_bytes()
    _name, line1, line2 = source.splitlines()
    raw = "ÉTOILE".encode() + b"\r\n" + line1 + b"\r\n" + line2 + b"\r\n"
    record = data.parse_celestrak_gp_response(raw, data.CelestrakGpFormat.TLE).records[
        0
    ]
    assert record.raw == raw
    assert record.object_name == "ÉTOILE"
    assert record.to_omm().object_name == "ÉTOILE"


@pytest.mark.parametrize("status", [200, 403, 429])
def test_gp_repeat_status_is_not_retried_or_cached(status, tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    body = (Path(CORE_FIXTURES) / "celestrak/repeat_download.txt").read_bytes()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.CelestrakGpNotModified) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("active"),
                cache_dir=str(tmp_path),
                retries=3,
                backoff_s=0,
            )
    assert len(calls) == 1
    assert exc.value.status == status
    assert (
        exc.value.url
        == "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON"
    )
    assert exc.value.detail["status"] == status
    assert exc.value.detail["url"] == exc.value.url
    assert exc.value.cache_path is None
    assert exc.value.detail["cache_path"] is None
    assert list(tmp_path.rglob("*")) == []


def test_gp_http_html_is_not_retried_or_cached(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, content=b"<!DOCTYPE html><title>rate limit</title>", request=request
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.CelestrakGpHtmlResponse) as exc:
            data.fetch_constellation(
                sidereon.GnssSystem.GPS, cache_dir=str(tmp_path), retries=3, backoff_s=0
            )
    assert len(calls) == 1
    assert exc.value.status == 200
    assert (
        exc.value.url
        == "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=JSON"
    )
    assert list(tmp_path.rglob("*")) == []


def test_gp_transport_retries_and_checksum(tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    body = (Path(CORE_FIXTURES) / "omm/25544.json").read_bytes()
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500, content=b"upstream unavailable", request=request)
        return httpx.Response(200, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.ChecksumMismatch) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("stations"),
                cache_dir=str(tmp_path),
                sha256="0" * 64,
                retries=3,
                backoff_s=0,
            )
    assert len(calls) == 3
    assert exc.value.expected == "0" * 64
    assert exc.value.got == (
        "01c4adc5186ac571830103bd741dbcd096190f347024faba2b47492053f731c3"
    )
    assert list(tmp_path.rglob("*")) == []


def test_gp_offline_age_and_verified_cache_on_repeat(tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    root = Path(CORE_FIXTURES)
    body = (root / "omm/25544.json").read_bytes()
    repeat = (root / "celestrak/repeat_download.txt").read_bytes()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, content=body if len(calls) == 1 else repeat, request=request
        )

    query = data.CelestrakGpQuery.group("active")
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_celestrak_gp(query, cache_dir=str(tmp_path), offline=True)
    for age in (-1.0, float("nan"), float("inf")):
        with pytest.raises(data.UnsupportedProduct, match="max_age_s"):
            data.fetch_celestrak_gp(query, cache_dir=str(tmp_path), max_age_s=age)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        data.fetch_celestrak_gp(query, cache_dir=str(tmp_path))
        monkeypatch.setattr(data, "_fresh_enough", lambda path, age: False)
        with pytest.raises(data.CelestrakGpNotModified) as exc:
            data.fetch_celestrak_gp(query, cache_dir=str(tmp_path))
    assert len(calls) == 2
    assert exc.value.cache_path is not None
    assert Path(exc.value.cache_path).read_bytes() == body
    assert exc.value.detail["cache_path"] == exc.value.cache_path
    assert exc.value.status == 200


@pytest.mark.parametrize("value", [-1, 2**32, 10**9, 1.5, "123", True, False])
def test_gp_invalid_catalog_number_is_typed(value):
    with pytest.raises(data.InvalidCelestrakGpSelector) as exc:
        data.celestrak_gp_url(
            data.CelestrakGpQuery.catalog_number(value), data.CelestrakGpFormat.JSON
        )
    assert exc.value.detail["kind"] == "invalid_celestrak_gp_selector"
    assert exc.value.detail["selector"] == "CATNR"
    assert exc.value.detail["value"] == str(value)
    assert exc.value.selector == "CATNR"
    assert exc.value.value == str(value)
    assert exc.value.__cause__ is None


def test_gp_boundary_errors_have_detail():
    from sidereon import _sidereon as native

    calls = [
        (
            lambda: native.data_celestrak_gp_url("unknown", "x", "JSON"),
            "invalid_celestrak_gp_query_kind",
            "QUERY",
            "unknown",
        ),
        (
            lambda: native.data_celestrak_gp_url("special", "unknown", "JSON"),
            "invalid_celestrak_gp_special",
            "SPECIAL",
            "unknown",
        ),
        (
            lambda: native.data_celestrak_gp_format_as_str("unknown"),
            "invalid_celestrak_gp_format",
            "FORMAT",
            "unknown",
        ),
    ]
    for call, kind, selector, value in calls:
        with pytest.raises(ValueError) as exc:
            call()
        assert exc.value.detail == {
            "family": "DataCatalogError",
            "kind": kind,
            "selector": selector,
            "value": value,
            "message": f"invalid CelesTrak GP {selector}: {value}",
        }

    class InvalidFormat:
        value = "unknown"

    for query, format, selector in [
        (data.CelestrakGpQuery("unknown", "x"), data.CelestrakGpFormat.JSON, "QUERY"),
        (
            data.CelestrakGpQuery("special", "unknown"),
            data.CelestrakGpFormat.JSON,
            "SPECIAL",
        ),
        (data.CelestrakGpQuery.group("active"), InvalidFormat(), "FORMAT"),
    ]:
        with pytest.raises(data.InvalidCelestrakGpSelector) as exc:
            data.celestrak_gp_url(query, format)
        assert exc.value.selector == selector
        assert exc.value.detail["selector"] == selector
        assert exc.value.__cause__ is None


def test_gp_non_success_classifies_no_data(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, content=b"No GP data found", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.CelestrakGpDataNotFound) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("active"),
                cache_dir=str(tmp_path),
                retries=3,
                backoff_s=0,
            )
    assert exc.value.status == 404
    assert exc.value.detail["status"] == 404
    assert exc.value.detail["url"] == exc.value.url
    assert len(calls) == 1
    assert list(tmp_path.rglob("*")) == []


def test_gp_retry_then_success_and_default_age(tmp_path, monkeypatch):
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    body = (Path(CORE_FIXTURES) / "omm/25544.json").read_bytes()
    calls = []
    ages = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                503, content=b"<html>unavailable</html>", request=request
            )
        return httpx.Response(200, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        response = data.fetch_celestrak_gp(
            data.CelestrakGpQuery.group("active"),
            cache_dir=str(tmp_path),
            retries=2,
            backoff_s=0,
        )
        assert response.records[0].object_name == "ISS (ZARYA)"
        monkeypatch.setattr(
            data, "_fresh_enough", lambda path, age: ages.append(age) or True
        )
        data.fetch_celestrak_gp(
            data.CelestrakGpQuery.group("active"), cache_dir=str(tmp_path)
        )
    assert len(calls) == 2
    assert ages == [7200.0]


@pytest.mark.parametrize(("status", "body"), [(500, b""), (429, b"<html>busy</html>")])
def test_gp_transient_html_or_empty_exhausts_retries(
    status, body, tmp_path, monkeypatch
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.HttpStatusError) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("active"),
                cache_dir=str(tmp_path),
                retries=3,
                backoff_s=0,
            )
    assert exc.value.status == status
    assert len(calls) == 3
    assert list(tmp_path.rglob("*")) == []


def test_gp_error_without_detail_gets_http_context(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(200, content=b"<html>error</html>", request=request)

    def parse_without_detail(body, format):
        raise data.CelestrakGpHtmlResponse("html")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        monkeypatch.setattr(data, "parse_celestrak_gp_response", parse_without_detail)
        with pytest.raises(data.CelestrakGpHtmlResponse) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("active"), cache_dir=str(tmp_path)
            )
    assert exc.value.detail == {"status": 200, "url": exc.value.url}


def test_gp_unrecognized_http_failure_preserves_status_after_retries(
    tmp_path, monkeypatch
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, content=b"server unavailable", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(data.httpx, "stream", client.stream)
        with pytest.raises(data.HttpStatusError) as exc:
            data.fetch_celestrak_gp(
                data.CelestrakGpQuery.group("active"),
                cache_dir=str(tmp_path),
                retries=3,
                backoff_s=0,
            )
    assert exc.value.status == 500
    assert (
        exc.value.url
        == "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON"
    )
    assert len(calls) == 3
    assert list(tmp_path.rglob("*")) == []


def test_gp_value_equality_and_inequality():
    from pathlib import Path

    from _helpers import CORE_FIXTURES

    fixtures = Path(CORE_FIXTURES) / "omm"
    raw = (fixtures / "25544.tle").read_bytes()
    other_raw = (fixtures / "24876.tle").read_bytes()
    first = data.parse_celestrak_gp_response(raw, data.CelestrakGpFormat.TLE)
    equal = data.parse_celestrak_gp_response(raw, data.CelestrakGpFormat.TLE)
    different = data.parse_celestrak_gp_response(other_raw, data.CelestrakGpFormat.TLE)
    assert first == equal
    assert first != different
    assert first.records[0] == equal.records[0]
    assert first.records[0] != different.records[0]
    assert first.records[0].payload == equal.records[0].payload
    assert first.records[0].payload != different.records[0].payload
    assert first.records[0].to_element_set() == equal.records[0].to_element_set()
    assert first.records[0].to_element_set() != different.records[0].to_element_set()
    malformed = data.parse_celestrak_gp_response(
        raw + b"STRAY\n" + raw, data.CelestrakGpFormat.TLE
    )
    same = data.parse_celestrak_gp_response(
        raw + b"STRAY\n" + raw, data.CelestrakGpFormat.TLE
    )
    incomplete = data.parse_celestrak_gp_response(
        raw + b"0 NAME\n", data.CelestrakGpFormat.TLE
    )
    assert malformed.rejected[0] == same.rejected[0]
    assert malformed.rejected[0] != incomplete.rejected[0]
    assert malformed.rejected[0].reason == same.rejected[0].reason
    assert malformed.rejected[0].reason != incomplete.rejected[0].reason
