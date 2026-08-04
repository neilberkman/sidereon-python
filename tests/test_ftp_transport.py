"""Anonymous-FTP transport for cataloged ``ftp://`` archives.

Parity with the Elixir interface's 0.36.1 transport: the Wuhan ``wum_nrt``
hourly line is FTP-only, and its acquisition must carry the same bounded
semantics as HTTP - byte caps, a 550 mapped to archive absence like a 404,
and directory URLs fetching ``LIST`` text. The fake FTP client below is
duck-typed to the small surface the transport uses, so everything but the
socket is under test; the ``network``-marked test exercises the real WHU
archive.
"""

import datetime as dt
import ftplib
import gzip

import pytest
from sidereon import data, distribution

WUM_DATE = dt.date(2026, 8, 3)
WUM_PATH = "/pub/gps/products/mgex/2430/WUM0MGXNRT_20262150500_02D_05M_ORB.SP3.gz"


def _wum_nrt_sp3_text():
    """Synthesize an archive-shaped WUM NRT product: SP3-d, agency WHU,
    576 epochs (half-open two-day grid at 300 s) from the filename epoch."""
    start = dt.datetime(2026, 8, 3, 5, 0, 0)
    gps_seconds = int((start - dt.datetime(1980, 1, 6)).total_seconds())
    week, tow = divmod(gps_seconds, 7 * 86_400)
    mjd_day = (start.date() - dt.date(1858, 11, 17)).days
    mjd_fraction = (
        start - dt.datetime.combine(start.date(), dt.time())
    ).seconds / 86_400

    lines = [
        f"#dP{start.year:4d} {start.month:2d} {start.day:2d} {start.hour:2d}"
        f" {start.minute:2d} {start.second:11.8f}     576   u+U IGS20 FIT  WHU",
        f"## {week:4d} {float(tow):15.8f} {300.0:14.8f} {mjd_day:5d}"
        f" {mjd_fraction:.13f}",
        "+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "++         5  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* TEST WUM NRT FIXTURE",
        "/*",
        "/*",
        "/*",
    ]
    for index in range(576):
        epoch = start + dt.timedelta(seconds=300 * index)
        seconds = epoch.second + epoch.microsecond / 1_000_000
        lines.append(
            f"*  {epoch.year:4d} {epoch.month:2d} {epoch.day:2d} {epoch.hour:2d}"
            f" {epoch.minute:2d} {seconds:11.8f}"
        )
        lines.append("PG01  15000.000000 -20000.000000   5000.000000    123.456789")
    lines.append("EOF")
    return "\n".join(lines) + "\n"


class _FakeFtp:
    """Duck-typed stand-in for ``ftplib.FTP`` serving a canned tree."""

    tree = {}
    connected_hosts = []

    def __init__(self, host, timeout=None):
        type(self).connected_hosts.append((host, timeout))

    def login(self):
        return "230"

    def voidcmd(self, _command):
        return "200"

    def retrbinary(self, command, sink, blocksize=8192):
        path = command.removeprefix("RETR ")
        payload = self._lookup(path)
        for offset in range(0, len(payload), blocksize):
            sink(payload[offset : offset + blocksize])
        return "226"

    def retrlines(self, command, collect):
        path = command.removeprefix("LIST ")
        payload = self._lookup(path)
        for line in payload.decode().splitlines():
            collect(line)
        return "226"

    def _lookup(self, path):
        try:
            return self.tree[path]
        except KeyError:
            raise ftplib.error_perm(f"550 {path}: No such file or directory") from None

    def quit(self):
        return "221"

    def close(self):
        return None


@pytest.fixture
def fake_ftp(monkeypatch):
    _FakeFtp.tree = {}
    _FakeFtp.connected_hosts = []
    monkeypatch.setattr(distribution, "_ftp_connect", _FakeFtp)
    return _FakeFtp


def test_wum_nrt_acquires_over_ftp_with_exact_provenance(fake_ftp, tmp_path):
    fake_ftp.tree[WUM_PATH] = gzip.compress(_wum_nrt_sp3_text().encode())

    product = data.ops_ultra_sp3("wum_nrt", WUM_DATE, issue="0500")
    request = distribution.request(product, [distribution.Distribution.direct()])
    acquired = distribution.acquire(request, cache_dir=str(tmp_path))

    assert acquired.provenance.resolved_identity.analysis_center == "wum_nrt"
    assert acquired.provenance.resolved_identity.solution_class == "near_real_time"
    assert fake_ftp.connected_hosts[0][0] == "igs.gnsswhu.cn"

    # Cache-first on the second acquisition: no new FTP connection.
    connections = len(fake_ftp.connected_hosts)
    distribution.acquire(request, cache_dir=str(tmp_path))
    assert len(fake_ftp.connected_hosts) == connections


def test_ftp_550_maps_to_archive_absence(fake_ftp, tmp_path):
    product = data.ops_ultra_sp3("wum_nrt", WUM_DATE, issue="0500")
    request = distribution.request(product, [distribution.Distribution.direct()])
    with pytest.raises(distribution.ProductNotPublished):
        distribution.acquire(request, cache_dir=str(tmp_path), retries=1)


def test_ftp_download_respects_the_byte_cap(fake_ftp, tmp_path):
    fake_ftp.tree[WUM_PATH] = b"x" * 4096

    product = data.ops_ultra_sp3("wum_nrt", WUM_DATE, issue="0500")
    request = distribution.request(product, [distribution.Distribution.direct()])
    # Oversize surfaces exactly as it does on the HTTP path: wrapped into the
    # acquisition's typed validation failure.
    with pytest.raises(distribution.ProductValidationFailure, match="byte limit"):
        distribution.acquire(
            request, cache_dir=str(tmp_path), retries=1, max_archive_bytes=1024
        )


def test_ftp_directory_urls_fetch_list_text(fake_ftp):
    listing = (
        b"-r--r--r--    1 0        0         1865742 Aug 04 06:30 "
        b"WUM0MGXNRT_20262150500_02D_05M_ORB.SP3.gz\n"
    )
    fake_ftp.tree["/pub/gps/products/mgex/2430/"] = listing

    download = distribution._download_ftp_once(
        "ftp://igs.gnsswhu.cn/pub/gps/products/mgex/2430/", 30.0, 1 << 20
    )
    objects = data.parse_archive_listing(download.archive.decode())
    newest = data.newest_published_product("wum_nrt", "sp3", objects)
    assert newest.issue == "0500"


@pytest.mark.network
def test_live_whu_listing_over_real_ftp():
    download = distribution._download_ftp_once(
        "ftp://igs.gnsswhu.cn/pub/gps/products/mgex/2430/", 30.0, 1 << 22
    )
    objects = data.parse_archive_listing(download.archive.decode())
    newest = data.newest_published_product("wum_nrt", "sp3", objects)
    assert newest is not None
