"""NTRIP client transport for real-time correction streams.

This module owns sockets, HTTP streaming, reconnect timing, and Python
exceptions. NTRIP request construction, response classification, chunk decoding,
sourcetable parsing, and GGA formatting are delegated to the core sans-I/O
machine exposed by :mod:`sidereon._sidereon`.

NTRIP casters are user-supplied endpoints, so this module does not use the
fixed archive-host allowlist from :mod:`sidereon.data`. It connects only to the
configured host and port and never follows redirects. Basic authentication over
plain TCP is allowed because many NTRIP deployments require it, but credentials
are sent in cleartext unless ``tls=True``.
"""

from __future__ import annotations

import datetime as _dt
import socket as _socket
import ssl as _ssl
import time as _time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import httpx

import sidereon as _sidereon
from sidereon._sidereon import (
    GgaPosition,
    NtripClientMachine,
    NtripConfig,
    SsrStreamAssembler,
    classify_http_response,
    parse_sourcetable,
)

__all__ = [
    "NtripError",
    "CasterUnreachable",
    "NtripUnauthorized",
    "DigestNotSupported",
    "MountpointNotFound",
    "NtripHttpError",
    "NtripCasterError",
    "NtripProtocolError",
    "StreamStalled",
    "ReconnectPolicy",
    "GgaFeed",
    "NtripClient",
]


class NtripError(Exception):
    """Base class for NTRIP transport and caster failures."""


class CasterUnreachable(NtripError):
    """DNS, TCP, TLS, or connection-timeout failure."""


class NtripUnauthorized(NtripError):
    """The caster rejected the supplied credentials."""


class DigestNotSupported(NtripError):
    """The caster requires Digest authentication, which is not implemented."""


class MountpointNotFound(NtripError):
    """The caster rejected the mountpoint."""

    def __init__(self, message: str = "mountpoint not found", sourcetable=None) -> None:
        self.sourcetable = sourcetable
        super().__init__(message)


class NtripHttpError(NtripError):
    """HTTP status or unexpected content type returned by a caster."""

    def __init__(
        self,
        status: int,
        reason: str,
        *,
        content_type: Optional[str] = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.content_type = content_type
        detail = f"HTTP {status} {reason}".strip()
        if content_type is not None:
            detail = f"{detail} (content-type {content_type!r})"
        super().__init__(detail)


class NtripCasterError(NtripError):
    """A rev1 caster returned an ``ERROR - ...`` line."""


class NtripProtocolError(NtripError):
    """Malformed handshake, chunk stream, or protocol-inconsistent response."""


class StreamStalled(NtripError):
    """No stream bytes arrived within the configured stall timeout."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(
            f"stream stalled for {seconds:.3f}s; the mountpoint may require a GGA feed"
        )


@dataclass(frozen=True)
class ReconnectPolicy:
    initial_s: float = 1.0
    factor: float = 2.0
    cap_s: float = 60.0
    max_reconnects: Optional[int] = None


_DEFAULT_RECONNECT = ReconnectPolicy()


@dataclass(frozen=True)
class GgaFeed:
    position: Union[GgaPosition, Callable[[], GgaPosition]]
    interval_s: float = 10.0


class _Reconnectable(Exception):
    def __init__(self, error: NtripError) -> None:
        self.error = error
        super().__init__(str(error))


class NtripClient:
    def __init__(
        self,
        host: str,
        port: int = 2101,
        mountpoint: str = "",
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        version: str = "auto",
        tls: bool = False,
        user_agent_product: Optional[str] = None,
        gga: Optional[GgaFeed] = None,
        stall_timeout_s: float = 30.0,
        reconnect: ReconnectPolicy = _DEFAULT_RECONNECT,
    ) -> None:
        if version not in ("auto", "rev1", "rev2"):
            raise ValueError('version must be "auto", "rev1", or "rev2"')
        if stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be positive")
        self.host = host
        self.port = int(port)
        self.mountpoint = mountpoint
        self.username = username
        self.password = password
        self.version = version
        self.tls = tls
        self.user_agent_product = user_agent_product
        self.gga = gga
        self.stall_timeout_s = float(stall_timeout_s)
        self.reconnect = reconnect
        self._closed = False
        self._preferred_version: Optional[str] = None

    def __enter__(self) -> "NtripClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True

    def sourcetable(self):
        """Fetch and parse the caster sourcetable."""
        client = self._with_mountpoint("")
        if client._use_httpx():
            try:
                return client._httpx_sourcetable_once()
            except httpx.RemoteProtocolError:
                if client.version == "auto":
                    client._preferred_version = "rev1"
                    return client._raw_sourcetable_once("rev1")
                raise
        return client._raw_sourcetable_once(client._wire_version())

    def stream(self) -> Iterator[bytes]:
        """Yield payload bytes across reconnects; terminal rejections raise."""
        reconnects = 0
        while not self._closed:
            try:
                if self._use_httpx():
                    yield from self._httpx_stream_once()
                else:
                    yield from self._raw_stream_once(self._wire_version())
                raise _Reconnectable(NtripProtocolError("stream ended"))
            except httpx.RemoteProtocolError as exc:
                if self.version == "auto" and self.gga is None:
                    self._preferred_version = "rev1"
                    continue
                raise NtripProtocolError(str(exc)) from exc
            except (OSError, httpx.ConnectError, httpx.TimeoutException) as exc:
                error = CasterUnreachable(f"caster unreachable: {exc}")
                reconnects = self._maybe_reconnect(reconnects, error)
            except _Reconnectable as exc:
                reconnects = self._maybe_reconnect(reconnects, exc.error)

    def messages(self):
        """Yield decoded RTCM messages from :meth:`stream`.

        Frame-decode errors are skipped. Use :meth:`stream` with
        :class:`sidereon.SsrStreamAssembler` directly when you need decode
        diagnostics.
        """
        assembler = SsrStreamAssembler()
        for chunk in self.stream():
            yield from assembler.push_lossy(chunk)

    def stream_into(self, store, week_of=None):
        """Yield decoded RTCM messages while ingesting SSR messages into ``store``."""
        mapper = week_of or _default_week_of
        for message in self.messages():
            week, tow_s = mapper(_dt.datetime.now(_dt.timezone.utc))
            store.ingest(message, week, tow_s)
            yield message

    def _with_mountpoint(self, mountpoint: str) -> "NtripClient":
        return NtripClient(
            self.host,
            self.port,
            mountpoint,
            username=self.username,
            password=self.password,
            version=self.version,
            tls=self.tls,
            user_agent_product=self.user_agent_product,
            gga=self.gga,
            stall_timeout_s=self.stall_timeout_s,
            reconnect=self.reconnect,
        )

    def _wire_version(self) -> str:
        if self.version == "auto":
            return self._preferred_version or "rev2"
        return self.version

    def _config(self, mountpoint: Optional[str] = None, version: Optional[str] = None):
        interval = self.gga.interval_s if self.gga is not None else None
        return NtripConfig(
            self.host,
            self.port,
            self.mountpoint if mountpoint is None else mountpoint,
            version=version or self._wire_version(),
            username=self.username,
            password=self.password,
            user_agent_product=self.user_agent_product,
            gga_interval_s=interval,
        )

    def _use_httpx(self) -> bool:
        return self.gga is None and self._wire_version() == "rev2"

    def _url(self, path: str) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host}:{self.port}{path}"

    def _httpx_response(self, mountpoint: Optional[str] = None):
        config = self._config(mountpoint=mountpoint, version="rev2")
        path, headers = config.request_headers()
        timeout = httpx.Timeout(
            connect=self.stall_timeout_s,
            read=self.stall_timeout_s,
            write=self.stall_timeout_s,
            pool=self.stall_timeout_s,
        )
        client = httpx.Client(http2=False, follow_redirects=False, timeout=timeout)
        return client, client.stream("GET", self._url(path), headers=headers)

    def _httpx_sourcetable_once(self):
        client, stream = self._httpx_response(mountpoint="")
        with client, stream as response:
            classification = classify_http_response(
                response.status_code,
                response.reason_phrase,
                list(response.headers.multi_items()),
            )
            if classification.kind == "rejection":
                raise _map_rejection(classification.rejection)
            body = response.read()
            if classification.kind != "sourcetable":
                raise NtripProtocolError("caster did not return a sourcetable")
            return parse_sourcetable(body.decode("utf-8", "replace"))

    def _httpx_stream_once(self) -> Iterator[bytes]:
        client, stream = self._httpx_response()
        try:
            with client, stream as response:
                classification = classify_http_response(
                    response.status_code,
                    response.reason_phrase,
                    list(response.headers.multi_items()),
                )
                if classification.kind == "rejection":
                    raise _map_rejection(classification.rejection)
                if classification.kind == "sourcetable":
                    table = parse_sourcetable(
                        response.read().decode("utf-8", "replace")
                    )
                    raise MountpointNotFound(sourcetable=table)
                for chunk in response.iter_raw():
                    if self._closed:
                        return
                    if chunk:
                        yield bytes(chunk)
                raise _Reconnectable(NtripProtocolError("stream ended"))
        except httpx.ReadTimeout as exc:
            raise _Reconnectable(StreamStalled(self.stall_timeout_s)) from exc

    def _socket(self):
        raw = _socket.create_connection(
            (self.host, self.port), timeout=self.stall_timeout_s
        )
        raw.settimeout(min(1.0, self.stall_timeout_s))
        if not self.tls:
            return raw
        context = _ssl.create_default_context()
        tls_socket = context.wrap_socket(raw, server_hostname=self.host)
        tls_socket.settimeout(min(1.0, self.stall_timeout_s))
        return tls_socket

    def _raw_sourcetable_once(self, version: str):
        config = self._config(mountpoint="", version=version)
        machine = NtripClientMachine(config)
        try:
            with self._socket() as sock:
                sock.sendall(machine.connection_request())
                deadline = _time.monotonic() + self.stall_timeout_s
                while _time.monotonic() < deadline:
                    try:
                        data = sock.recv(65536)
                    except _socket.timeout:
                        continue
                    if not data:
                        events = machine.finish()
                    else:
                        events = machine.push(data)
                    for event in events:
                        if event.kind == "sourcetable":
                            return event.sourcetable
                        if event.kind == "rejected":
                            raise _map_rejection(event.rejection)
                        if event.kind == "stream_corrupted":
                            raise NtripProtocolError(event.detail or "stream corrupted")
                    if not data:
                        break
        except OSError as exc:
            raise CasterUnreachable(f"caster unreachable: {exc}") from exc
        raise StreamStalled(self.stall_timeout_s)

    def _raw_stream_once(self, version: str) -> Iterator[bytes]:
        config = self._config(version=version)
        machine = NtripClientMachine(config)
        try:
            with self._socket() as sock:
                sock.sendall(machine.connection_request())
                connected = False
                last_received = _time.monotonic()
                while not self._closed:
                    now = _time.monotonic()
                    if connected:
                        self._send_gga_if_due(machine, sock, now)
                    if now - last_received > self.stall_timeout_s:
                        raise _Reconnectable(StreamStalled(self.stall_timeout_s))
                    try:
                        data = sock.recv(65536)
                    except _socket.timeout:
                        continue
                    if not data:
                        for event in machine.finish():
                            yield from self._handle_raw_stream_event(
                                event, machine, sock
                            )
                        raise _Reconnectable(NtripProtocolError("stream ended"))
                    last_received = _time.monotonic()
                    for event in machine.push(data):
                        if event.kind == "connected":
                            connected = True
                        yield from self._handle_raw_stream_event(event, machine, sock)
        except OSError as exc:
            raise _Reconnectable(
                CasterUnreachable(f"caster unreachable: {exc}")
            ) from exc

    def _handle_raw_stream_event(self, event, machine, sock) -> Iterator[bytes]:
        if event.kind == "connected":
            self._send_gga_if_due(machine, sock, _time.monotonic())
            return
        if event.kind == "payload":
            payload = event.payload
            if payload:
                yield bytes(payload)
            return
        if event.kind == "sourcetable":
            raise MountpointNotFound(sourcetable=event.sourcetable)
        if event.kind == "rejected":
            raise _map_rejection(event.rejection)
        if event.kind == "stream_corrupted":
            raise _Reconnectable(NtripProtocolError(event.detail or "stream corrupted"))
        if event.kind == "stream_ended":
            raise _Reconnectable(NtripProtocolError("stream ended"))

    def _send_gga_if_due(self, machine, sock, now_s: float) -> None:
        if self.gga is None:
            return
        position = (
            self.gga.position() if callable(self.gga.position) else self.gga.position
        )
        message = machine.gga_message(now_s, position, _utc_seconds_of_day())
        if message:
            sock.sendall(message)

    def _maybe_reconnect(self, reconnects: int, error: NtripError) -> int:
        if (
            self.reconnect.max_reconnects is not None
            and reconnects >= self.reconnect.max_reconnects
        ):
            raise error
        delay = min(
            self.reconnect.cap_s,
            self.reconnect.initial_s * (self.reconnect.factor**reconnects),
        )
        if delay > 0:
            _time.sleep(delay)
        return reconnects + 1


def _map_rejection(rejection) -> NtripError:
    kind = rejection.kind
    if kind == "unauthorized":
        return NtripUnauthorized("unauthorized")
    if kind == "digest_required":
        return DigestNotSupported("digest authentication is not supported")
    if kind == "mountpoint_not_found":
        return MountpointNotFound()
    if kind == "caster_error":
        return NtripCasterError(rejection.reason or "caster error")
    if kind == "unexpected_content_type":
        return NtripHttpError(
            200,
            "unexpected content type",
            content_type=rejection.content_type,
        )
    if kind == "http_error":
        return NtripHttpError(rejection.status or 0, rejection.reason or "")
    if kind == "malformed_handshake":
        return NtripProtocolError("malformed NTRIP handshake")
    return NtripProtocolError(f"NTRIP rejection: {kind}")


def _utc_seconds_of_day() -> float:
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.hour * 3600.0 + now.minute * 60.0 + now.second + now.microsecond * 1.0e-6


def _default_week_of(when: _dt.datetime) -> tuple[int, float]:
    utc = when.astimezone(_dt.timezone.utc)
    seconds = utc.second + utc.microsecond * 1.0e-6
    jd = _sidereon.split_julian_date(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        seconds,
    ).jd
    gps_utc = _sidereon.gps_utc_offset_s(jd)
    gps_epoch = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc)
    elapsed_s = (utc - gps_epoch).total_seconds() + gps_utc
    week = int(elapsed_s // 604800.0)
    tow_s = elapsed_s - week * 604800.0
    return week, tow_s
