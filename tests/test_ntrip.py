import socketserver
import threading

import pytest
import sidereon
import sidereon.ntrip as ntrip

SOURCETABLE_TEXT = (
    "STR;MOUNT;ID;RTCM 3;1004;2;GPS;NET;USA;40.0;-105.0;1;0;gen;none;N;N;"
    "9600;misc\r\nENDSOURCETABLE\r\n"
)


def test_core_ntrip_request_and_sourcetable_types():
    config = sidereon.NtripConfig(
        "caster.example.test",
        2101,
        "MOUNT",
        username="user",
        password="pass",
    )
    request = config.request_bytes()
    assert request.startswith(b"GET /MOUNT HTTP/1.1\r\n")
    assert b"Host: caster.example.test:2101\r\n" in request
    assert b"Authorization: Basic dXNlcjpwYXNz\r\n" in request

    table = sidereon.parse_sourcetable(SOURCETABLE_TEXT)
    stream = table.streams()[0]
    assert stream.mountpoint == "MOUNT"
    assert stream.nmea_required is True
    assert table.to_text().endswith("ENDSOURCETABLE\r\n")


def test_core_ntrip_machine_decodes_rev2_chunked_payload():
    config = sidereon.NtripConfig("caster.example.test", 2101, "MOUNT")
    machine = sidereon.NtripClientMachine(config)
    machine.connection_request()

    events = machine.push(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: gnss/data\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"3\r\nabc\r\n0\r\n\r\n"
    )

    assert [event.kind for event in events] == ["connected", "payload", "stream_ended"]
    assert events[0].handshake.chunked is True
    assert events[1].payload == b"abc"


def test_nmea_parser_and_gga_writer_delegate_to_core():
    sentence = sidereon.write_gga(48.0, 11.0, 500.0, 45296.12)
    parsed = sidereon.parse_nmea_sentence(sentence)
    assert parsed.kind == "GGA"
    assert parsed.gga.lat_deg == 48.0
    assert parsed.gga.lon_deg == 11.0

    accumulator = sidereon.NmeaAccumulator()
    output = accumulator.push_bytes(sentence.encode())
    assert output.sentences[0].kind == "GGA"
    snapshot = accumulator.finish()
    assert snapshot.gga.altitude_msl_m == 500.0


class _Headers:
    def __init__(self, rows):
        self._rows = list(rows)

    def multi_items(self):
        return list(self._rows)


class _Response:
    def __init__(self, status_code, headers, chunks, reason_phrase="OK"):
        self.status_code = status_code
        self.headers = _Headers(headers)
        self.reason_phrase = reason_phrase
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_raw(self):
        yield from self._chunks

    def read(self):
        return b"".join(self._chunks)


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, headers):
        self.calls.append((method, url, headers))
        if not self.responses:
            raise AssertionError("unexpected stream call")
        return self.responses.pop(0)


def test_ntrip_client_httpx_stream_uses_core_classification(monkeypatch):
    fake = _Client(
        [
            _Response(
                200,
                [("Content-Type", "gnss/data")],
                [b"abc", b"def"],
            )
        ]
    )
    monkeypatch.setattr(ntrip.httpx, "Client", lambda **_kwargs: fake)
    client = ntrip.NtripClient(
        "caster.example.test",
        mountpoint="MOUNT",
        reconnect=ntrip.ReconnectPolicy(initial_s=0.0, max_reconnects=0),
    )

    stream = client.stream()
    assert next(stream) == b"abc"
    assert next(stream) == b"def"
    with pytest.raises(ntrip.NtripProtocolError):
        next(stream)
    assert fake.calls[0][1] == "http://caster.example.test:2101/MOUNT"


def test_ntrip_client_sourcetable_fetch(monkeypatch):
    fake = _Client(
        [
            _Response(
                200,
                [("Content-Type", "gnss/sourcetable")],
                [SOURCETABLE_TEXT.encode()],
            )
        ]
    )
    monkeypatch.setattr(ntrip.httpx, "Client", lambda **_kwargs: fake)
    client = ntrip.NtripClient("caster.example.test")

    table = client.sourcetable()

    assert table.streams()[0].mountpoint == "MOUNT"


class _Rev1Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.server.requests.append(self.request.recv(4096))
        self.request.sendall(b"ICY 200 OK\r\n\r\nabc")


def test_ntrip_client_rev1_raw_socket_streams_payload():
    with socketserver.TCPServer(("127.0.0.1", 0), _Rev1Handler) as server:
        server.requests = []
        thread = threading.Thread(target=server.handle_request)
        thread.start()

        client = ntrip.NtripClient(
            "127.0.0.1",
            port=server.server_address[1],
            mountpoint="MOUNT",
            version="rev1",
            reconnect=ntrip.ReconnectPolicy(initial_s=0.0, max_reconnects=0),
        )
        stream = client.stream()
        assert next(stream) == b"abc"
        with pytest.raises(ntrip.NtripProtocolError):
            next(stream)

        thread.join(timeout=2.0)
        assert b"GET /MOUNT HTTP/1.0\r\n" in server.requests[0]
