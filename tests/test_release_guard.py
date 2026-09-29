from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

RELEASE_GUARD = Path(__file__).resolve().parents[1] / "scripts" / "check-release.py"
ROOT = RELEASE_GUARD.parents[1]
GUARD = runpy.run_path(str(RELEASE_GUARD))


def candidate_dependencies(revision: str) -> dict[str, object]:
    return {
        "sidereon": {
            "git": "https://github.com/neilberkman/sidereon",
            "rev": revision,
            "version": "3.0.0",
        },
        "sidereon-core": {
            "git": "https://github.com/neilberkman/sidereon",
            "rev": revision,
            "version": "3.0.0",
            "features": ["mmap"],
        },
    }


def test_candidate_requires_matching_full_revision_and_versions() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    assert (
        GUARD["candidate_fixture_revision"](candidate_dependencies(revision), "3.0.0")
        == revision
    )


def test_trust_region_accepts_only_exact_coordinated_candidate_revision() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    dependencies = candidate_dependencies(revision)
    validate = GUARD["validate_trust_region_dependency"]
    validate("0.11.0", dependencies, "3.0.0", allow_candidate=False)
    validate(
        {
            "version": "0.11.0",
            "git": "https://github.com/neilberkman/sidereon",
            "rev": revision,
        },
        dependencies,
        "3.0.0",
        allow_candidate=True,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda candidate: candidate.update(version="0.11.1"),
        lambda candidate: candidate.update(git="https://github.com/other/repo"),
        lambda candidate: candidate.update(rev="f" * 40),
        lambda candidate: candidate.update(branch="main"),
    ],
)
def test_trust_region_rejects_candidate_mismatches_and_extra_fields(mutation) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    dependencies = candidate_dependencies(revision)
    candidate = {
        "version": "0.11.0",
        "git": "https://github.com/neilberkman/sidereon",
        "rev": revision,
    }
    mutation(candidate)
    with pytest.raises(ValueError):
        GUARD["validate_trust_region_dependency"](
            candidate, dependencies, "3.0.0", allow_candidate=True
        )


def test_trust_region_candidate_is_rejected_in_registry_mode() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    with pytest.raises(ValueError):
        GUARD["validate_trust_region_dependency"](
            {
                "version": "0.11.0",
                "git": "https://github.com/neilberkman/sidereon",
                "rev": revision,
            },
            candidate_dependencies(revision),
            "3.0.0",
            allow_candidate=False,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda dependencies: dependencies["sidereon"].update(
            git="https://github.com/other/repo"
        ),
        lambda dependencies: dependencies["sidereon"].update(path="../sidereon"),
        lambda dependencies: dependencies["sidereon"].update(branch="main"),
        lambda dependencies: dependencies["sidereon"].update(tag="v3.0.0"),
        lambda dependencies: dependencies["sidereon"].update(package="other"),
        lambda dependencies: dependencies["sidereon"].update(rev="main"),
        lambda dependencies: dependencies["sidereon"].update(rev="0" * 39),
        lambda dependencies: dependencies["sidereon"].update(version="3.0.1"),
        lambda dependencies: dependencies["sidereon-core"].update(rev="f" * 40),
    ],
)
def test_candidate_rejects_unapproved_sources_or_mismatches(mutation) -> None:
    dependencies = candidate_dependencies("a" * 40)
    mutation(dependencies)
    with pytest.raises(ValueError):
        GUARD["candidate_fixture_revision"](dependencies, "3.0.0")


def test_registry_guard_rejects_git_and_path_dependencies() -> None:
    registry_version = GUARD["registry_version"]
    for source in (
        {"version": "3.0.0", "git": "https://github.com/neilberkman/sidereon"},
        {"version": "3.0.0", "path": "../sidereon"},
        {"version": "=3.0.0", "registry": "alternate"},
        {"version": "=3.0.0", "rev": "a" * 40},
        {"version": "=3.0.0", "package": "other"},
    ):
        with pytest.raises(SystemExit):
            registry_version("sidereon", source, "3.0.0")


def test_registry_guard_requires_exact_release_requirement() -> None:
    registry_version = GUARD["registry_version"]
    assert registry_version("sidereon", {"version": "=3.0.0"}, "3.0.0") == "3.0.0"
    assert registry_version("sidereon", "=3.0.0", "3.0.0") == "3.0.0"
    for requirement in ("3.0.0", "^3.0.0", ">=3.0.0, <4.0.0"):
        with pytest.raises(SystemExit, match="require exactly"):
            registry_version("sidereon", {"version": requirement}, "3.0.0")


def test_registry_ci_fixture_ref_is_version_tag() -> None:
    dependencies = {"sidereon": "=3.0.0", "sidereon-core": "=3.0.0"}
    assert GUARD["ci_fixture_ref"](dependencies, "3.0.0") == "v3.0.0"


def test_candidate_ci_fixture_ref_is_the_validated_revision() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    assert (
        GUARD["ci_fixture_ref"](candidate_dependencies(revision), "3.0.0") == revision
    )


def test_ci_rejects_mixed_candidate_and_registry_sources() -> None:
    dependencies = candidate_dependencies("a" * 40)
    dependencies["sidereon-core"] = "3.0.0"
    with pytest.raises(ValueError):
        GUARD["ci_fixture_ref"](dependencies, "3.0.0")


def test_registry_ci_fixture_accepts_only_exact_crates_io_requirements() -> None:
    dependencies = {
        "sidereon": {"version": "=3.0.0", "features": ["mmap"]},
        "sidereon-core": {"version": "=3.0.0"},
    }
    assert GUARD["ci_fixture_ref"](dependencies, "3.0.0") == "v3.0.0"
    dependencies["sidereon-core"] = {"version": "^3.0.0"}
    with pytest.raises(SystemExit, match="require exactly"):
        GUARD["ci_fixture_ref"](dependencies, "3.0.0")
    dependencies["sidereon-core"] = {"version": "=3.0.0", "registry": "other"}
    with pytest.raises(SystemExit, match="crates.io registry"):
        GUARD["ci_fixture_ref"](dependencies, "3.0.0")
    dependencies["sidereon-core"] = {"version": "=3.0.0", "package": "other"}
    with pytest.raises(SystemExit, match="select package"):
        GUARD["ci_fixture_ref"](dependencies, "3.0.0")


def test_maturin_and_cargo_enforce_a_current_lockfile(tmp_path: Path) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["maturin"]["locked"] is True
    cargo = shutil.which("cargo")
    maturin = shutil.which("maturin")
    if cargo is None or maturin is None:
        pytest.skip("Cargo and pinned Maturin are required for the PEP 517 probe")

    crate = tmp_path / "Cargo.toml"
    lock = tmp_path / "Cargo.lock"
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.rs").write_text("fn main() {}\n")
    crate.write_text(
        '[package]\nname = "stale-lock-probe"\nversion = "1.0.0"\n'
        'edition = "2021"\n\n[[bin]]\nname = "stale-lock-probe"\n'
        'path = "src/main.rs"\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["maturin==1.14.1"]\n'
        'build-backend = "maturin"\n\n[project]\n'
        'name = "stale-lock-probe"\nversion = "1.0.0"\n\n'
        '[tool.maturin]\nbindings = "bin"\nlocked = true\n'
    )
    valid_lock = (
        'version = 4\n\n[[package]]\nname = "stale-lock-probe"\nversion = "1.0.0"\n'
    ).encode()
    lock.write_bytes(valid_lock)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    backend_call = (
        "import maturin, sys; maturin.prepare_metadata_for_build_wheel(sys.argv[1])"
    )
    cargo_env = os.environ.copy()
    cargo_env["CARGO_TERM_COLOR"] = "never"
    valid = subprocess.run(
        [sys.executable, "-c", backend_call, str(metadata_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=cargo_env,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr

    stale_lock = valid_lock.replace(b'version = "1.0.0"', b'version = "0.9.0"')
    lock.write_bytes(stale_lock)
    before_failure = lock.read_bytes()
    rejected = subprocess.run(
        [sys.executable, "-c", backend_call, str(metadata_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=cargo_env,
        check=False,
    )
    assert rejected.returncode != 0
    assert "error: cannot update the lock file" in rejected.stderr, rejected.stderr
    locked_refusal = "because --locked was passed to prevent this"
    assert locked_refusal in rejected.stderr, rejected.stderr
    assert lock.read_bytes() == before_failure


def test_build_workflows_use_the_locked_cargo_graph() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert "cargo clippy --locked --all-targets" in ci
    assert "maturin develop --locked" in ci
    assert release.count("args: --release --locked --out dist") == 3
    assert "cargo metadata --locked --format-version 1" in release
    assert '--config-settings="maturin.build-args=--locked"' in release
