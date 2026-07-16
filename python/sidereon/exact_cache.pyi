from pathlib import Path
from typing import ContextManager

from .distribution import DistributionSource, ProductIdentity

CONTROL_DIRECTORY: str

class CacheLockTimeout(OSError): ...
class CacheFormatError(OSError): ...

class CacheFiles:
    product: Path
    archive: Path
    provenance: Path
    entry_id: str
    product_bytes: bytes
    archive_bytes: bytes
    provenance_bytes: bytes

class ExactProductCache:
    def __init__(
        self,
        path: Path,
        identity: ProductIdentity,
        source: DistributionSource,
        timeout_s: float = ...,
    ) -> None: ...
    def committed_files(self) -> CacheFiles | None: ...
    def publish(
        self, product: bytes, archive: bytes, provenance: bytes
    ) -> CacheFiles: ...
    def cleanup_abandoned(self) -> None: ...
    def close(self) -> None: ...

def entry_lock(
    path: Path,
    identity: ProductIdentity,
    source: DistributionSource,
    timeout_s: float = ...,
) -> ContextManager[ExactProductCache]: ...
def read(
    path: Path, identity: ProductIdentity, source: DistributionSource
) -> CacheFiles | None: ...
