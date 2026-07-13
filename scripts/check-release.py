#!/usr/bin/env python3
"""Verify that the Python release and its Rust engine dependencies move together."""

from __future__ import annotations

import argparse
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

    if args.tag is not None and args.tag != f"v{python_version}":
        raise SystemExit(
            f"tag {args.tag!r} does not match package version v{python_version}"
        )

    print(f"release metadata aligned at {python_version}")


if __name__ == "__main__":
    main()
