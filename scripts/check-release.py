#!/usr/bin/env python3
"""Verify that the Python release and its Rust engine dependencies move together."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
CORE_GIT_URL = "https://github.com/neilberkman/sidereon"
FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
CORE_SOURCE_REVISION = "47eaf24e39afa71168670664e88361ed41011bc2"


def registry_version(name, value, package_version):
    if isinstance(value, str):
        requirement = value
    elif not isinstance(value, dict):
        raise SystemExit(f"{name} dependency has an unrecognized form: {value!r}")
    else:
        for disallowed in ("path", "git", "branch", "tag", "registry", "rev"):
            if disallowed in value:
                raise SystemExit(
                    f"{name} must use the crates.io registry, found {disallowed}="
                    f"{value[disallowed]!r}"
                )
        if value.get("package", name) != name:
            raise SystemExit(f"{name} registry dependency must select package {name!r}")
        if "version" not in value:
            raise SystemExit(f"{name} must state a registry version, found {value!r}")
        requirement = value["version"]
    expected_requirement = f"={package_version}"
    if requirement != expected_requirement:
        raise SystemExit(
            f"{name} registry dependency must require exactly "
            f"{expected_requirement!r}, found {requirement!r}"
        )
    return package_version


def candidate_fixture_revision(engine_dependencies, package_version):
    revisions = []
    for name in ("sidereon", "sidereon-core"):
        value = engine_dependencies[name]
        if not isinstance(value, dict):
            raise ValueError(f"{name} must use the approved candidate Git dependency")
        allowed_fields = {
            "default-features",
            "features",
            "git",
            "optional",
            "package",
            "rev",
            "version",
        }
        if set(value) - allowed_fields:
            raise ValueError(
                f"{name} candidate dependency has unsupported fields: "
                f"{sorted(set(value) - allowed_fields)!r}"
            )
        if value.get("package", name) != name:
            raise ValueError(
                f"{name} candidate dependency must select package {name!r}"
            )
        if value.get("git") != CORE_GIT_URL:
            raise ValueError(f"{name} candidate dependency must use {CORE_GIT_URL!r}")
        revision = value.get("rev")
        if (
            not isinstance(revision, str)
            or FULL_GIT_REVISION.fullmatch(revision) is None
        ):
            raise ValueError(
                f"{name} candidate dependency must pin a full 40-character Git revision"
            )
        if value.get("version") != package_version:
            raise ValueError(
                f"{name} candidate version must match package version "
                f"{package_version!r}"
            )
        revisions.append(revision)
    if revisions[0] != revisions[1]:
        raise ValueError(
            "sidereon and sidereon-core candidate dependencies must use "
            "the same Git revision"
        )
    return revisions[0]


def is_git_candidate(engine_dependencies):
    return any(
        isinstance(engine_dependencies[name], dict)
        and "git" in engine_dependencies[name]
        for name in ("sidereon", "sidereon-core")
    )


def registry_fixture_ref(package_version):
    return f"v{package_version}"


def ci_fixture_ref(engine_dependencies, package_version):
    if is_git_candidate(engine_dependencies):
        return candidate_fixture_revision(engine_dependencies, package_version)
    for name in ("sidereon", "sidereon-core"):
        registry_version(name, engine_dependencies[name], package_version)
    return registry_fixture_ref(package_version)


def validate_trust_region_dependency(
    value, engine_dependencies, package_version, *, allow_candidate
):
    """Accept the registry release or its exact coordinated Git candidate."""
    if value == "0.11.0":
        return
    if not allow_candidate:
        raise ValueError(
            "trust-region-least-squares dependency must be the compliant "
            f"0.11.0 release, found {value!r}"
        )
    if not isinstance(value, dict) or set(value) != {"version", "git", "rev"}:
        raise ValueError(
            "trust-region-least-squares candidate dependency must contain only "
            "version, git, and rev"
        )
    try:
        expected_revision = candidate_fixture_revision(
            engine_dependencies, package_version
        )
    except ValueError as error:
        raise ValueError(
            "trust-region-least-squares candidate requires the approved "
            f"coordinated engine revision: {error}"
        ) from error
    if value.get("version") != "0.11.0":
        raise ValueError(
            "trust-region-least-squares candidate version must be exactly "
            f"'0.11.0', found {value.get('version')!r}"
        )
    if value.get("git") != CORE_GIT_URL:
        raise ValueError(
            f"trust-region-least-squares candidate must use {CORE_GIT_URL!r}"
        )
    if value.get("rev") != expected_revision:
        raise ValueError(
            "trust-region-least-squares candidate revision must match the "
            f"coordinated engine revision {expected_revision!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tag", help="release tag, for example v0.26.0")
    modes.add_argument(
        "--candidate",
        action="store_true",
        help="validate only the coordinated full-revision Git candidate",
    )
    modes.add_argument(
        "--ci",
        action="store_true",
        help="validate CI registry or approved candidate inputs and select fixture ref",
    )
    parser.add_argument("--github-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.github_output and not args.ci:
        parser.error("--github-output is only valid with --ci")

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
    try:
        validate_trust_region_dependency(
            trust_region_version,
            engine_dependencies,
            python_version,
            allow_candidate=(
                args.candidate or (args.ci and is_git_candidate(engine_dependencies))
            ),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.candidate:
        try:
            fixture_ref = candidate_fixture_revision(
                engine_dependencies, python_version
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        engine_dependency_versions = {
            name: engine_dependencies[name]["version"] for name in engine_dependencies
        }
        source_kind = "candidate Git revision"
    elif args.ci:
        try:
            fixture_ref = ci_fixture_ref(engine_dependencies, python_version)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if is_git_candidate(engine_dependencies):
            engine_dependency_versions = {
                name: engine_dependencies[name]["version"]
                for name in engine_dependencies
            }
            source_kind = "candidate Git revision"
        else:
            engine_dependency_versions = {
                name: registry_version(name, value, python_version)
                for name, value in engine_dependencies.items()
            }
            source_kind = "registry versions"
    else:
        engine_dependency_versions = {
            name: registry_version(name, value, python_version)
            for name, value in engine_dependencies.items()
        }
        fixture_ref = registry_fixture_ref(python_version)
        source_kind = "registry versions"

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
        "sidereon dependency": engine_dependency_versions["sidereon"],
        "sidereon-core dependency": engine_dependency_versions["sidereon-core"],
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
        "mod.rs": "0703d1b3470f59528880ae34990f064897d34876d5ff30b4fc860afdcadf7433",
        "ocean.rs": "25946677944425671a92717860ac2d70f255de5403eeb1fbf98b361716821d5c",
        "pole.rs": "b4cc4c16bdd8ce1d8f04073602ab47dfb85a002b946ab192e8d4d2d600f0a1f8",
    }
    tide_root = ROOT / "third_party_source" / "sidereon-core-3.0.2" / "tides"
    for filename, expected_digest in tide_sources.items():
        source = tide_root / filename
        if not source.is_file():
            raise SystemExit(f"missing IERS-derived source disclosure {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise SystemExit(
                f"IERS-derived source disclosure {source} has digest {digest}, "
                f"expected {expected_digest} from sidereon-core 3.0.2 at "
                f"{CORE_SOURCE_REVISION}"
            )

    if args.tag is not None and args.tag != f"v{python_version}":
        raise SystemExit(
            f"tag {args.tag!r} does not match package version v{python_version}"
        )

    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"fixture_ref={fixture_ref}\n")
    print(f"release metadata aligned at {python_version} ({source_kind})")


if __name__ == "__main__":
    main()
