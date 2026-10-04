"""The generated fixtures identify the exact resolved engine source.

Git candidates are checked against their full locked commits. Published
packages are checked against Cargo's locked archive checksum and the VCS
metadata inside that archive, so registry fixtures retain the source commit
without relying on an environment variable or a hand-entered revision.
"""

import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

import pytest
from _helpers import FIXTURES

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = Path(FIXTURES)
_SCRIPTS = _ROOT / "scripts"
_RELEASE_VERSION = "3.0.1"
_REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_GIT_URL = "https://github.com/neilberkman/sidereon"
_DEPENDENCIES = ("sidereon", "sidereon-core")
_PROJECTS = (
    ("root", _ROOT / "Cargo.toml", _ROOT / "Cargo.lock"),
    (
        "core_goldens",
        _SCRIPTS / "core_goldens/Cargo.toml",
        _SCRIPTS / "core_goldens/Cargo.lock",
    ),
    (
        "ppp_esbc_expected",
        _SCRIPTS / "ppp_esbc_expected/Cargo.toml",
        _SCRIPTS / "ppp_esbc_expected/Cargo.lock",
    ),
)


def _manifest_dependencies(path):
    text = path.read_text()
    found = {}
    dependency_pattern = r"^\s*(sidereon(?:-core)?)\s*=\s*\{([^}\n]*)\}"
    for name, body in re.findall(dependency_pattern, text, re.M):
        assert name not in found, f"{path}: duplicate direct dependency {name}"
        fields = r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*"([^"]*)"'
        found[name] = dict(re.findall(fields, body))
    assert set(found) == set(_DEPENDENCIES), (
        f"{path}: engine dependencies are {sorted(found)}"
    )
    return found


def _locked_packages(path):
    text = path.read_text()
    packages = {}
    blocks = re.split(r"^\[\[package\]\]\s*$", text, flags=re.M)[1:]
    for block in blocks:
        fields = dict(re.findall(r'^([a-z_]+)\s*=\s*"([^"]*)"\s*$', block, re.M))
        name = fields.get("name")
        if name not in _DEPENDENCIES:
            continue
        assert name not in packages, f"{path}: duplicate locked package {name}"
        packages[name] = fields
    assert set(packages) == set(_DEPENDENCIES), f"{path}: missing an engine package"
    return packages


def _registry_commit(name, version, checksum, cargo_home):
    cache_root = cargo_home / "registry" / "cache"
    archives = list(cache_root.glob(f"*/{name}-{version}.crate"))
    matching = []
    for archive_path in archives:
        archive_bytes = archive_path.read_bytes()
        if hashlib.sha256(archive_bytes).hexdigest() != checksum:
            continue
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/.cargo_vcs_info.json")
            ]
            assert len(members) == 1, (
                f"{archive_path}: expected one published VCS record"
            )
            manifests = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/Cargo.toml")
            ]
            assert len(manifests) == 1, (
                f"{archive_path}: expected one published manifest"
            )
            manifest_stream = archive.extractfile(manifests[0])
            assert manifest_stream is not None, (
                f"{archive_path}: unreadable published manifest"
            )
            manifest = manifest_stream.read().decode()
            package_pattern = r"(?ms)^\[package\]\s*$([\s\S]*?)(?=^\[|\Z)"
            package_section = re.search(package_pattern, manifest)
            assert package_section is not None, (
                f"{archive_path}: missing package metadata"
            )
            package_fields = dict(
                re.findall(
                    r'^([a-z_]+)\s*=\s*"([^"]*)"\s*$',
                    package_section.group(1),
                    re.M,
                )
            )
            assert (
                package_fields.get("name") == name
                and package_fields.get("version") == version
            ), f"{archive_path}: package identity does not match its lock entry"
            stream = archive.extractfile(members[0])
            assert stream is not None, (
                f"{archive_path}: unreadable published VCS record"
            )
            vcs = json.load(stream)
        revision = vcs.get("git", {}).get("sha1")
        assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{archive_path}: malformed published Git commit {revision!r}"
        )
        assert vcs.get("git", {}).get("dirty") is not True, (
            f"{archive_path}: published Git source is marked dirty"
        )
        matching.append(revision)
    assert matching, (
        f"no {name} {version} archive matches Cargo.lock checksum {checksum}"
    )
    assert len(set(matching)) == 1, (
        f"cached {name} {version} archives disagree on VCS commit"
    )
    return matching[0]


def _source_identity(manifest_path, lock_path, cargo_home=None):
    dependencies = _manifest_dependencies(manifest_path)
    packages = _locked_packages(lock_path)
    for name in _DEPENDENCIES:
        assert packages[name].get("version") == _RELEASE_VERSION, (
            f"{lock_path}: {name} must resolve to {_RELEASE_VERSION}"
        )

    manifest_kinds = set()
    manifest_revisions = set()
    for name in _DEPENDENCIES:
        dependency = dependencies[name]
        forbidden = {"path", "branch", "tag", "registry"} & dependency.keys()
        assert not forbidden, (
            f"{manifest_path}: {name} has unsupported source selectors {forbidden}"
        )
        git = dependency.get("git")
        revision = dependency.get("rev")
        expected_requirement = (
            _RELEASE_VERSION
            if git is not None or revision is not None
            else f"={_RELEASE_VERSION}"
        )
        assert dependency.get("version") == expected_requirement, (
            f"{manifest_path}: {name} must request exactly "
            f"{expected_requirement!r} for its source"
        )
        assert dependency.get("package", name) == name, (
            f"{manifest_path}: {name} must resolve the {name} package"
        )
        if git is not None or revision is not None:
            assert git == _GIT_URL, (
                f"{manifest_path}: {name} uses noncanonical Git URL {git!r}"
            )
            is_full_revision = isinstance(revision, str) and re.fullmatch(
                r"[0-9a-f]{40}", revision
            )
            assert is_full_revision, (
                f"{manifest_path}: {name} must pin one full Git revision"
            )
            manifest_kinds.add("git")
            manifest_revisions.add(revision)
        else:
            manifest_kinds.add("registry")
    assert len(manifest_kinds) == 1, (
        f"{manifest_path}: mixed Git and registry dependencies"
    )
    if manifest_kinds == {"git"}:
        assert len(manifest_revisions) == 1, (
            f"{manifest_path}: facade/core Git pins differ"
        )
        (manifest_revision,) = manifest_revisions
        lock_revisions = set()
        expected_source = f"git+{_GIT_URL}?rev={manifest_revision}#{manifest_revision}"
        for name in _DEPENDENCIES:
            assert packages[name].get("source") == expected_source, (
                f"{lock_path}: {name} does not resolve the manifest's full Git pin"
            )
            lock_revisions.add(packages[name]["source"].rsplit("#", 1)[1])
        assert lock_revisions == {manifest_revision}
        return "git", manifest_revision, ()

    assert not manifest_revisions
    checksums = set()
    for name in _DEPENDENCIES:
        package = packages[name]
        assert package.get("source") == _REGISTRY_SOURCE, (
            f"{lock_path}: {name} is not from the approved crates.io registry"
        )
        checksum = package.get("checksum")
        assert isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum), (
            f"{lock_path}: {name} has missing or malformed registry checksum"
        )
        checksums.add((name, checksum))
    if cargo_home is None:
        cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    revisions = {
        _registry_commit(name, _RELEASE_VERSION, checksum, cargo_home)
        for name, checksum in checksums
    }
    assert len(revisions) == 1, f"{lock_path}: published facade/core VCS commits differ"
    (revision,) = revisions
    return "registry", revision, tuple(sorted(checksums))


def test_root_and_generators_resolve_the_fixture_source():
    source_kind, revision, _ = _assert_projects_match(_PROJECTS)
    assert source_kind in {"git", "registry"}
    for fixture in ("core_goldens.json", "ppp_esbc.json"):
        with (_FIXTURES / fixture).open() as handle:
            assert json.load(handle)["core_revision"] == revision, (
                f"{fixture}: fixture source differs from {source_kind} source "
                f"{revision}"
            )


def _write_candidate(tmp_path, revision):
    manifest = tmp_path / "Cargo.toml"
    lock = tmp_path / "Cargo.lock"
    manifest.write_text(
        "[dependencies]\n"
        f'sidereon = {{ version = "3.0.1", git = "{_GIT_URL}", rev = "{revision}" }}\n'
        f'sidereon-core = {{ version = "3.0.1", git = "{_GIT_URL}", '
        f'rev = "{revision}" }}\n'
    )
    source = f"git+{_GIT_URL}?rev={revision}#{revision}"
    lock.write_text(
        '[[package]]\nname = "sidereon"\nversion = "3.0.1"\n'
        f'source = "{source}"\n\n'
        '[[package]]\nname = "sidereon-core"\nversion = "3.0.1"\n'
        f'source = "{source}"\n'
    )
    return manifest, lock


def _assert_projects_match(projects):
    identities = {
        name: _source_identity(manifest, lock) for name, manifest, lock in projects
    }
    assert len(set(identities.values())) == 1, f"project sources differ: {identities}"
    return next(iter(identities.values()))


def test_git_source_mismatch_fails_even_when_locks_are_consistent(tmp_path):
    manifest, lock = _write_candidate(tmp_path, "a" * 40)
    assert _source_identity(manifest, lock) == ("git", "a" * 40, ())
    manifest.write_text(manifest.read_text().replace("a" * 40, "b" * 40))
    with pytest.raises(AssertionError, match="does not resolve"):
        _source_identity(manifest, lock)


@pytest.mark.parametrize(
    "changed_project", ["root", "core_goldens", "ppp_esbc_expected"]
)
def test_one_project_cannot_resolve_a_different_commit(tmp_path, changed_project):
    projects = []
    for name in ("root", "core_goldens", "ppp_esbc_expected"):
        directory = tmp_path / name
        directory.mkdir()
        revision = "b" * 40 if name == changed_project else "a" * 40
        manifest, lock = _write_candidate(directory, revision)
        projects.append((name, manifest, lock))
    with pytest.raises(AssertionError, match="sources differ"):
        _assert_projects_match(projects)


def _write_registry_archive(cargo_home, name, version, revision, dirty=False):
    registry = cargo_home / "registry" / "cache" / "index.crates.io-test"
    registry.mkdir(parents=True, exist_ok=True)
    archive_path = registry / f"{name}-{version}.crate"
    body = json.dumps(
        {"git": {"sha1": revision, "dirty": dirty}, "path_in_vcs": ""}
    ).encode()
    manifest_body = f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest_member = tarfile.TarInfo(f"{name}-{version}/Cargo.toml")
        manifest_member.size = len(manifest_body)
        archive.addfile(manifest_member, io.BytesIO(manifest_body))
        member = tarfile.TarInfo(f"{name}-{version}/.cargo_vcs_info.json")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def test_registry_vcs_commit_is_bound_to_both_locked_checksums(tmp_path):
    revision = "c" * 40
    cargo_home = tmp_path / "cargo"
    checksums = {
        name: _write_registry_archive(cargo_home, name, _RELEASE_VERSION, revision)
        for name in _DEPENDENCIES
    }
    manifest = tmp_path / "Cargo.toml"
    lock = tmp_path / "Cargo.lock"
    manifest.write_text(
        '[dependencies]\nsidereon = { version = "=3.0.1" }\n'
        'sidereon-core = { version = "=3.0.1" }\n'
    )
    packages = []
    for name in _DEPENDENCIES:
        packages.append(
            f'[[package]]\nname = "{name}"\nversion = "3.0.1"\n'
            f'source = "{_REGISTRY_SOURCE}"\nchecksum = "{checksums[name]}"\n'
        )
    lock.write_text("\n".join(packages))
    identity = _source_identity(manifest, lock, cargo_home)
    assert identity[:2] == ("registry", revision)
    expected_checksums = tuple(
        sorted((name, checksums[name]) for name in _DEPENDENCIES)
    )
    assert identity[2] == expected_checksums

    bad_lock = lock.read_text().replace(checksums["sidereon-core"], "0" * 64)
    lock.write_text(bad_lock)
    with pytest.raises(AssertionError, match="checksum"):
        _source_identity(manifest, lock, cargo_home)


def test_registry_lock_rejects_wrong_source_version_and_missing_checksum(tmp_path):
    cargo_home = tmp_path / "cargo"
    checksums = {
        name: _write_registry_archive(cargo_home, name, _RELEASE_VERSION, "f" * 40)
        for name in _DEPENDENCIES
    }
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[dependencies]\nsidereon = { version = "=3.0.1" }\n'
        'sidereon-core = { version = "=3.0.1" }\n'
    )
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "3.0.1"\n'
            f'source = "{_REGISTRY_SOURCE}"\nchecksum = "{checksums[name]}"\n'
            for name in _DEPENDENCIES
        )
    )
    valid_lock = lock.read_text()
    invalid_cases = (
        (
            valid_lock.replace(_REGISTRY_SOURCE, "registry+https://example.invalid"),
            "approved",
        ),
        (valid_lock.replace('version = "3.0.1"', 'version = "3.0.2"', 1), "resolve"),
        (
            re.sub(r'^checksum = "[0-9a-f]{64}"\n', "", valid_lock, flags=re.M),
            "checksum",
        ),
    )
    for invalid_lock, error_text in invalid_cases:
        lock.write_text(invalid_lock)
        with pytest.raises(AssertionError, match=error_text):
            _source_identity(manifest, lock, cargo_home)


def test_registry_vcs_metadata_must_name_a_full_commit(tmp_path):
    cargo_home = tmp_path / "cargo"
    checksums = {
        name: _write_registry_archive(cargo_home, name, _RELEASE_VERSION, "bad")
        for name in _DEPENDENCIES
    }
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[dependencies]\nsidereon = { version = "=3.0.1" }\n'
        'sidereon-core = { version = "=3.0.1" }\n'
    )
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "3.0.1"\n'
            f'source = "{_REGISTRY_SOURCE}"\nchecksum = "{checksums[name]}"\n'
            for name in _DEPENDENCIES
        )
    )
    with pytest.raises(AssertionError, match="malformed published Git commit"):
        _source_identity(manifest, lock, cargo_home)


def test_registry_dirty_vcs_metadata_is_rejected(tmp_path):
    cargo_home = tmp_path / "cargo"
    checksums = {
        name: _write_registry_archive(
            cargo_home, name, _RELEASE_VERSION, "a" * 40, dirty=True
        )
        for name in _DEPENDENCIES
    }
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[dependencies]\nsidereon = { version = "=3.0.1" }\n'
        'sidereon-core = { version = "=3.0.1" }\n'
    )
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "3.0.1"\n'
            f'source = "{_REGISTRY_SOURCE}"\nchecksum = "{checksums[name]}"\n'
            for name in _DEPENDENCIES
        )
    )
    with pytest.raises(AssertionError, match="marked dirty"):
        _source_identity(manifest, lock, cargo_home)


def test_manifest_package_alias_cannot_substitute_engine_package(tmp_path):
    manifest, lock = _write_candidate(tmp_path, "a" * 40)
    manifest.write_text(
        manifest.read_text().replace("sidereon = {", 'sidereon = { package = "other",')
    )
    with pytest.raises(AssertionError, match="must resolve the sidereon package"):
        _source_identity(manifest, lock)


def test_registry_packages_from_different_commits_fail(tmp_path):
    cargo_home = tmp_path / "cargo"
    checksums = {
        "sidereon": _write_registry_archive(
            cargo_home, "sidereon", _RELEASE_VERSION, "d" * 40
        ),
        "sidereon-core": _write_registry_archive(
            cargo_home, "sidereon-core", _RELEASE_VERSION, "e" * 40
        ),
    }
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(
        '[dependencies]\nsidereon = { version = "=3.0.1" }\n'
        'sidereon-core = { version = "=3.0.1" }\n'
    )
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "3.0.1"\n'
            f'source = "{_REGISTRY_SOURCE}"\nchecksum = "{checksums[name]}"\n'
            for name in _DEPENDENCIES
        )
    )
    with pytest.raises(AssertionError, match="commits differ"):
        _source_identity(manifest, lock, cargo_home)
