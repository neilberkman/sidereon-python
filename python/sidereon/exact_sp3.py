"""Exact SP3 request validation backed by the Sidereon Rust core."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Union

from ._sidereon import Sp3
from ._sidereon import _exact_sp3_request_from_identity as _core_from_identity
from ._sidereon import _parse_exact_sp3 as _core_parse_exact_sp3
from ._sidereon import _validate_exact_sp3 as _core_validate_exact_sp3
from ._sidereon import (
    _validate_exact_sp3_request as _core_validate_exact_sp3_request,
)


class ExactSp3ValidationError(ValueError):
    """An SP3 product does not satisfy its exact requested identity."""


class ExactSp3Coverage(Enum):
    """Regular-grid boundary representation present in an exact SP3 product."""

    HALF_OPEN = "half_open"
    INCLUSIVE = "inclusive"


@dataclass(frozen=True, init=False)
class ExactSp3Request:
    """Date, span, cadence, and optional agency required from one SP3 product.

    Use :meth:`from_identity` when validating acquired product bytes. It retains
    every SP3-relevant constraint carried by the catalog identity, including the
    content format revision when one was declared.
    """

    _date: dt.date = field(repr=False)
    _span: str = field(repr=False)
    _sample: str = field(repr=False)
    _issue: Optional[str] = field(repr=False)
    _expected_agency: Optional[str] = field(repr=False)
    _format_version: Optional[str] = field(default=None, repr=False)
    _identity_json: Optional[str] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        date: dt.date,
        span: str,
        sample: str,
        issue: Optional[str] = None,
        expected_agency: Optional[str] = None,
    ) -> None:
        if not isinstance(date, dt.date) or isinstance(date, dt.datetime):
            raise ExactSp3ValidationError("date must be a datetime.date")
        object.__setattr__(self, "_date", date)
        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_sample", sample)
        object.__setattr__(self, "_issue", issue)
        object.__setattr__(self, "_expected_agency", expected_agency)
        object.__setattr__(self, "_format_version", None)
        object.__setattr__(self, "_identity_json", None)
        self._validate()

    @property
    def date(self) -> dt.date:
        return self._date

    @property
    def span(self) -> str:
        return self._span

    @property
    def sample(self) -> str:
        return self._sample

    @property
    def issue(self) -> Optional[str]:
        return self._issue

    @property
    def expected_agency(self) -> Optional[str]:
        return self._expected_agency

    @property
    def format_version(self) -> Optional[str]:
        return self._format_version

    def __repr__(self) -> str:
        return (
            "ExactSp3Request("
            f"date={self.date!r}, span={self.span!r}, sample={self.sample!r}, "
            f"issue={self.issue!r}, expected_agency={self.expected_agency!r}, "
            f"format_version={self.format_version!r})"
        )

    @classmethod
    def from_identity(cls, identity: object) -> "ExactSp3Request":
        """Build an exact SP3 request from a distribution product identity."""
        from .distribution import ProductIdentity

        if not isinstance(identity, ProductIdentity):
            raise ExactSp3ValidationError("identity must be a ProductIdentity")
        try:
            value = identity.to_dict()
            if not isinstance(value, Mapping):
                raise TypeError
            identity_json = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            (
                year,
                month,
                day,
                issue,
                span,
                sample,
                format_version,
                expected_agency,
            ) = _core_from_identity(identity_json)
        except (TypeError, ValueError) as exc:
            raise ExactSp3ValidationError(str(exc)) from None
        request = cls(
            dt.date(year, month, day),
            span,
            sample,
            issue=issue,
            expected_agency=expected_agency,
        )
        object.__setattr__(request, "_format_version", format_version)
        object.__setattr__(request, "_identity_json", identity_json)
        request._validate()
        return request

    def _validate(self) -> None:
        try:
            _core_validate_exact_sp3_request(
                self.date.year,
                self.date.month,
                self.date.day,
                self.issue,
                self.span,
                self.sample,
                self.expected_agency,
                self._identity_json,
            )
        except ValueError as exc:
            raise ExactSp3ValidationError(str(exc)) from None

    def _core_args(self) -> tuple[object, ...]:
        return (
            self.date.year,
            self.date.month,
            self.date.day,
            self.issue,
            self.span,
            self.sample,
            self.expected_agency,
            self._identity_json,
        )


def parse_exact_sp3(
    source: Union[bytes, bytearray, str, os.PathLike[str]],
    request: ExactSp3Request,
) -> tuple[Sp3, ExactSp3Coverage]:
    """Parse SP3 bytes and enforce the complete exact-product request."""
    if not isinstance(request, ExactSp3Request):
        raise TypeError("request must be an ExactSp3Request")
    if isinstance(source, (bytes, bytearray)):
        content = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        with open(os.fspath(source), "rb") as handle:
            content = handle.read()
    else:
        raise TypeError("source must be bytes, bytearray, or a path")
    try:
        sp3, coverage = _core_parse_exact_sp3(content, *request._core_args())
    except ValueError as exc:
        raise ExactSp3ValidationError(str(exc)) from None
    return sp3, ExactSp3Coverage(coverage)


def validate_exact_sp3(sp3: Sp3, request: ExactSp3Request) -> ExactSp3Coverage:
    """Validate an already parsed SP3 product against an exact request."""
    if not isinstance(request, ExactSp3Request):
        raise TypeError("request must be an ExactSp3Request")
    try:
        coverage = _core_validate_exact_sp3(sp3, *request._core_args())
    except ValueError as exc:
        raise ExactSp3ValidationError(str(exc)) from None
    return ExactSp3Coverage(coverage)


__all__ = [
    "ExactSp3Coverage",
    "ExactSp3Request",
    "ExactSp3ValidationError",
    "parse_exact_sp3",
    "validate_exact_sp3",
]
