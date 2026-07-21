"""Types for exact SP3 request validation."""

import datetime as dt
import enum
import os
from dataclasses import dataclass
from typing import Union

from . import Sp3
from .distribution import ProductIdentity

class ExactSp3ValidationError(ValueError): ...

class ExactSp3Coverage(enum.Enum):
    HALF_OPEN = "half_open"
    INCLUSIVE = "inclusive"

@dataclass(frozen=True)
class ExactSp3Request:
    @property
    def date(self) -> dt.date: ...
    @property
    def span(self) -> str: ...
    @property
    def sample(self) -> str: ...
    @property
    def issue(self) -> str | None: ...
    @property
    def expected_agency(self) -> str | None: ...
    @property
    def format_version(self) -> str | None: ...
    def __init__(
        self,
        date: dt.date,
        span: str,
        sample: str,
        issue: str | None = ...,
        expected_agency: str | None = ...,
    ) -> None: ...
    @classmethod
    def from_identity(cls, identity: ProductIdentity) -> ExactSp3Request: ...

def parse_exact_sp3(
    source: Union[bytes, bytearray, str, os.PathLike[str]],
    request: ExactSp3Request,
) -> tuple[Sp3, ExactSp3Coverage]: ...
def validate_exact_sp3(sp3: Sp3, request: ExactSp3Request) -> ExactSp3Coverage: ...
