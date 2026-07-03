"""Type stubs for the NTRIP transport client."""

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import sidereon

__all__: list[str]

class NtripError(Exception): ...
class CasterUnreachable(NtripError): ...
class NtripUnauthorized(NtripError): ...
class DigestNotSupported(NtripError): ...

class MountpointNotFound(NtripError):
    sourcetable: sidereon.Sourcetable | None
    def __init__(
        self, message: str = ..., sourcetable: sidereon.Sourcetable | None = ...
    ) -> None: ...

class NtripHttpError(NtripError):
    status: int
    reason: str
    content_type: str | None
    def __init__(
        self, status: int, reason: str, *, content_type: str | None = ...
    ) -> None: ...

class NtripCasterError(NtripError): ...
class NtripProtocolError(NtripError): ...

class StreamStalled(NtripError):
    seconds: float
    def __init__(self, seconds: float) -> None: ...

@dataclass(frozen=True)
class ReconnectPolicy:
    initial_s: float = ...
    factor: float = ...
    cap_s: float = ...
    max_reconnects: Optional[int] = ...

@dataclass(frozen=True)
class GgaFeed:
    position: Union[sidereon.GgaPosition, Callable[[], sidereon.GgaPosition]]
    interval_s: float = ...

class NtripClient:
    host: str
    port: int
    mountpoint: str
    username: str | None
    password: str | None
    version: str
    tls: bool
    user_agent_product: str | None
    gga: GgaFeed | None
    stall_timeout_s: float
    reconnect: ReconnectPolicy
    def __init__(
        self,
        host: str,
        port: int = ...,
        mountpoint: str = ...,
        *,
        username: str | None = ...,
        password: str | None = ...,
        version: str = ...,
        tls: bool = ...,
        user_agent_product: str | None = ...,
        gga: GgaFeed | None = ...,
        stall_timeout_s: float = ...,
        reconnect: ReconnectPolicy = ...,
    ) -> None: ...
    def __enter__(self) -> NtripClient: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...
    def close(self) -> None: ...
    def sourcetable(self) -> sidereon.Sourcetable: ...
    def stream(self) -> Iterator[bytes]: ...
    def messages(self) -> Iterator[sidereon.RtcmMessage]: ...
    def stream_into(
        self,
        store: sidereon.SsrCorrectionStore,
        week_of: Callable[[_dt.datetime], tuple[int, float]] | None = ...,
    ) -> Iterator[sidereon.RtcmMessage]: ...
