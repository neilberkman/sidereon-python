#!/usr/bin/env python3
"""Verify that the Python release and its Rust engine dependencies move together."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="release tag, for example v0.26.0")
    args = parser.parse_args()

    with (ROOT / "pyproject.toml").open("rb") as handle:
        python_version = tomllib.load(handle)["project"]["version"]
    with (ROOT / "Cargo.toml").open("rb") as handle:
        cargo = tomllib.load(handle)
    with (ROOT / "uv.lock").open("rb") as handle:
        uv_packages = tomllib.load(handle)["package"]

    rust_version = cargo["package"]["version"]
    engine_dependencies = {
        name: cargo["dependencies"][name] for name in ("sidereon", "sidereon-core")
    }
    trust_region_version = cargo["dependencies"]["trust-region-least-squares"]
    if trust_region_version != "0.9.2":
        raise SystemExit(
            "trust-region-least-squares dependency must be the compliant "
            f"0.9.2 patch, found {trust_region_version!r}"
        )
    non_registry_pins = {
        name: value
        for name, value in engine_dependencies.items()
        if not isinstance(value, str)
    }
    if non_registry_pins:
        details = ", ".join(
            f"{name}={value!r}" for name, value in non_registry_pins.items()
        )
        raise SystemExit(
            "engine dependencies must be plain registry version pins; " + details
        )

    uv_project_versions = [
        package["version"]
        for package in uv_packages
        if package["name"] == "sidereon" and package.get("source") == {"editable": "."}
    ]
    if len(uv_project_versions) != 1:
        raise SystemExit(
            "uv.lock must contain exactly one editable sidereon project package"
        )

    expected = {
        "Python package": python_version,
        "Rust extension crate": rust_version,
        "sidereon dependency": engine_dependencies["sidereon"],
        "sidereon-core dependency": engine_dependencies["sidereon-core"],
        "uv project lock": uv_project_versions[0],
    }
    mismatches = {
        name: version for name, version in expected.items() if version != python_version
    }
    if mismatches:
        details = ", ".join(
            f"{name}={version!r}" for name, version in mismatches.items()
        )
        raise SystemExit(f"release versions must match {python_version!r}: {details}")

    changelog_heading = f"## [{python_version}]"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if changelog_heading not in changelog:
        raise SystemExit(f"CHANGELOG.md is missing {changelog_heading!r}")

    notices = (ROOT / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    required_notices = (
        "approx` 0.5.1",
        "nalgebra` 0.33.3",
        "nalgebra-macros` 0.2.2",
        "simba` 0.9.1",
        "Apache License",
        "Copyright © 2015, Simonas Kazlauskas",
        "IERS Conventions Software License",
        "e) The source code must be included",
        "third_party_licenses/ERFA-BSD-3-Clause.txt",
        "third_party_licenses/SciPy-BSD-3-Clause.txt",
    )
    missing_notices = [item for item in required_notices if item not in notices]
    if missing_notices:
        raise SystemExit(
            "THIRD-PARTY-NOTICES.md is missing required release notices: "
            + ", ".join(repr(item) for item in missing_notices)
        )

    third_party_licenses = {
        "ERFA-BSD-3-Clause.txt": (
            "b1858f9a263f22c438a455a32945da51a31a0ae25a21055da13bb7ed57cc3b51"
        ),
        "IERS-CONVENTIONS-SOFTWARE-LICENSE.txt": (
            "a441d8ffe8151ddd5f1e0a9f82ce88ed54bd2f55e83fee6a519e50b006a8cba2"
        ),
        "SciPy-BSD-3-Clause.txt": (
            "221e59f5e910fd7f94e44f0dac77436a11338c285c6346232e4a850a50da0e94"
        ),
    }
    license_root = ROOT / "third_party_licenses"
    for filename, expected_digest in third_party_licenses.items():
        license_file = license_root / filename
        if not license_file.is_file():
            raise SystemExit(f"missing third-party license {license_file}")
        digest = hashlib.sha256(license_file.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise SystemExit(
                f"third-party license {license_file} has digest {digest}, "
                f"expected {expected_digest} from the pinned upstream release"
            )

    tide_sources = {
        "mod.rs": "7c71cb8facbd81af8473d3634e4c63d97dda8cb37a2f59888d3397cfdde4d39b",
        "ocean.rs": "6bd72d6647b634f979b670040d8c0b659e1f581fa41fdeec41b74b85d8c26c01",
        "pole.rs": "b4cc4c16bdd8ce1d8f04073602ab47dfb85a002b946ab192e8d4d2d600f0a1f8",
    }
    tide_root = ROOT / "third_party_source" / "sidereon-core-0.34.0" / "tides"
    for filename, expected_digest in tide_sources.items():
        source = tide_root / filename
        if not source.is_file():
            raise SystemExit(f"missing IERS-derived source disclosure {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise SystemExit(
                f"IERS-derived source disclosure {source} has digest {digest}, "
                f"expected {expected_digest} from sidereon-core 0.34.0"
            )

    if args.tag is not None and args.tag != f"v{python_version}":
        raise SystemExit(
            f"tag {args.tag!r} does not match package version v{python_version}"
        )

    print(f"release metadata aligned at {python_version}")


if __name__ == "__main__":
    main()
