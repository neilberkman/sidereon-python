"""Process and crash-boundary tests for the exact-product cache."""

import gzip
import hashlib
import json
import multiprocessing
import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import sidereon._exact_cache as exact_cache
import sidereon.distribution as distribution
from test_distribution import _client, _gzip_response, _sp3_bytes, _sp3_request


def _process_acquire(source_kind, source_path, cache_dir, ready, start, results):
    if source_kind == "local_file":
        source = distribution.Distribution.local_file(source_path)
    else:
        source = distribution.Distribution.in_memory(_sp3_bytes())
    request = _sp3_request(source)
    ready.put(True)
    start.wait()
    try:
        acquired = distribution.acquire(request, cache_dir=cache_dir)
        results.put(
            (
                "ok",
                acquired.provenance.cache_hit,
                acquired.provenance.distribution_source.value,
                acquired.provenance.sha256,
                acquired.path,
            )
        )
    except BaseException as error:  # pragma: no cover - returned to parent
        results.put(("error", repr(error)))


def _process_crash_commit(path, content, archive, provenance, step):
    def stop_at(current):
        if current == step:
            os._exit(86)

    exact_cache._hit_failpoint = stop_at
    distribution._commit_cache(Path(path), content, archive, provenance)


def _process_hold_lock(path, ready, release):
    stable = Path(path)
    with exact_cache.entry_lock(stable, 5.0):
        orphan = stable.parent / ".sidereon-cache-v2" / "entries" / ("f" * 32)
        orphan.mkdir(parents=True)
        ready.put(str(orphan))
        release.wait()


def _spawn_context():
    return multiprocessing.get_context("spawn")


def _start_racing_acquisitions(tmp_path, source_kinds):
    context = _spawn_context()
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    source_path = tmp_path / "source.SP3.gz"
    source_path.write_bytes(gzip.compress(_sp3_bytes(), mtime=0))
    processes = [
        context.Process(
            target=_process_acquire,
            args=(
                kind,
                str(source_path),
                str(tmp_path / "cache"),
                ready,
                start,
                results,
            ),
        )
        for kind in source_kinds
    ]
    for process in processes:
        process.start()
    for _ in processes:
        assert ready.get(timeout=10) is True
    start.set()
    outcomes = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
    return outcomes


def _alternate_candidate(first):
    content = _sp3_bytes().replace(b"EOF", b"/* alternate acquisition\nEOF")
    archive = gzip.compress(content, mtime=0)
    resolved = distribution._validate_product(
        first.provenance.requested_identity, content
    )
    provenance = replace(
        first.provenance,
        resolved_identity=resolved,
        retrieved_at="2030-01-01T00:00:00+00:00",
        byte_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        etag="alternate",
        cache_hit=False,
        archive_byte_length=len(archive),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )
    return content, archive, provenance


def test_two_os_processes_racing_same_source_acquire_once(tmp_path):
    outcomes = _start_racing_acquisitions(tmp_path, ["local_file", "local_file"])
    assert sorted(outcome[1] for outcome in outcomes) == [False, True]
    assert len({outcome[3] for outcome in outcomes}) == 1
    assert len({outcome[4] for outcome in outcomes}) == 1


def test_two_os_processes_using_allowed_distributors_keep_source_identity(tmp_path):
    outcomes = _start_racing_acquisitions(tmp_path, ["local_file", "in_memory"])
    assert not any(outcome[1] for outcome in outcomes)
    assert {outcome[2] for outcome in outcomes} == {"local_file", "in_memory"}
    assert len({outcome[3] for outcome in outcomes}) == 1
    assert len({outcome[4] for outcome in outcomes}) == 2


@pytest.mark.parametrize(
    "step",
    [
        "after_payload",
        "after_archive",
        "after_metadata",
        "after_entry_sync",
        "after_marker_write",
        "after_marker_rename",
        "after_commit_sync",
    ],
)
def test_process_death_at_every_commit_boundary_keeps_old_or_new_entry(tmp_path, step):
    cache_dir = tmp_path / step
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    first = distribution.acquire(request, cache_dir=cache_dir)
    stable = distribution._cache_path(
        cache_dir, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    old_digest = first.provenance.sha256
    content, archive, provenance = _alternate_candidate(first)

    context = _spawn_context()
    process = context.Process(
        target=_process_crash_commit,
        args=(str(stable), content, archive, provenance, step),
    )
    process.start()
    process.join(20)
    assert process.exitcode == 86

    accepted = distribution._load_cached(
        stable,
        request.identity,
        distribution.DistributionSource.IN_MEMORY,
        (),
        None,
    )
    assert accepted is not None
    assert accepted.provenance.sha256 in {old_digest, provenance.sha256}
    assert Path(accepted.path).read_bytes() in {_sp3_bytes(), content}
    assert hashlib.sha256(Path(accepted.path).read_bytes()).hexdigest() == (
        accepted.provenance.sha256
    )


def test_existing_entry_is_readable_until_refresh_commit_marker_moves(
    tmp_path, monkeypatch
):
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    first = distribution.acquire(request, cache_dir=tmp_path)
    stable = distribution._cache_path(
        tmp_path, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    content, archive, provenance = _alternate_candidate(first)
    staged = threading.Event()
    release = threading.Event()

    def pause_before_marker(step):
        if step == "after_entry_sync":
            staged.set()
            assert release.wait(10)

    monkeypatch.setattr(exact_cache, "_hit_failpoint", pause_before_marker)
    writer = threading.Thread(
        target=distribution._commit_cache,
        args=(stable, content, archive, provenance),
    )
    writer.start()
    assert staged.wait(10)
    during = distribution._load_cached(
        stable,
        request.identity,
        distribution.DistributionSource.IN_MEMORY,
        (),
        None,
    )
    assert during is not None
    assert during.provenance.sha256 == first.provenance.sha256
    release.set()
    writer.join(10)
    assert not writer.is_alive()
    after = distribution._load_cached(
        stable,
        request.identity,
        distribution.DistributionSource.IN_MEMORY,
        (),
        None,
    )
    assert after is not None
    assert after.provenance.sha256 == provenance.sha256


def test_mismatched_and_corrupt_committed_payloads_are_rejected(tmp_path):
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    acquired = distribution.acquire(request, cache_dir=tmp_path)
    stable = distribution._cache_path(
        tmp_path, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    files = exact_cache.committed_files(stable)
    assert files is not None
    original = files.product.read_bytes()
    files.product.write_bytes(b"mismatched payload")
    with pytest.raises(distribution.CacheReadFailure):
        distribution._load_cached(
            stable,
            request.identity,
            distribution.DistributionSource.IN_MEMORY,
            (),
            None,
        )
    files.product.write_bytes(original + b"corrupt")
    with pytest.raises(distribution.CacheReadFailure):
        distribution._load_cached(
            stable,
            request.identity,
            distribution.DistributionSource.IN_MEMORY,
            (),
            None,
        )
    assert (
        acquired.provenance.sha256
        != hashlib.sha256(files.product.read_bytes()).hexdigest()
    )


def test_reader_parses_the_provenance_bytes_bound_by_the_commit(tmp_path, monkeypatch):
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    distribution.acquire(request, cache_dir=tmp_path)
    stable = distribution._cache_path(
        tmp_path, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    files = exact_cache.committed_files(stable)
    assert files is not None
    real_read_bytes = Path.read_bytes
    provenance_reads = 0

    def controlled_read(path):
        nonlocal provenance_reads
        if path == files.provenance:
            provenance_reads += 1
            if provenance_reads > 1:
                return b'{"unbound":"replacement"}'
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", controlled_read)
    accepted = distribution._load_cached(
        stable,
        request.identity,
        distribution.DistributionSource.IN_MEMORY,
        (),
        None,
    )
    assert accepted is not None
    assert provenance_reads == 1


def test_legacy_verified_triple_is_migrated_to_one_committed_entry(tmp_path):
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    acquired = distribution.acquire(request, cache_dir=tmp_path / "source")
    stable = distribution._cache_path(
        tmp_path, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    stable.parent.mkdir(parents=True)
    Path(stable).write_bytes(Path(acquired.path).read_bytes())
    Path(f"{stable}.archive").write_bytes(Path(f"{acquired.path}.archive").read_bytes())
    Path(f"{stable}.provenance.json").write_text(
        json.dumps(acquired.provenance.to_dict())
    )
    migrated = distribution.acquire(request, cache_dir=tmp_path)
    assert migrated.provenance.cache_hit
    assert exact_cache.committed_files(stable) is not None
    assert Path(migrated.path).parent.name != stable.parent.name


def test_failed_legacy_migration_is_terminal_before_transport(tmp_path, monkeypatch):
    with _client(_gzip_response) as client:
        acquired = distribution.acquire(
            _sp3_request(), cache_dir=tmp_path / "source", http_client=client
        )
    request = _sp3_request()
    stable = distribution._cache_path(
        tmp_path / "target",
        request.identity,
        distribution.DistributionSource.NASA_CDDIS,
    )
    stable.parent.mkdir(parents=True)
    stable.write_bytes(Path(acquired.path).read_bytes())
    Path(f"{stable}.archive").write_bytes(Path(f"{acquired.path}.archive").read_bytes())
    Path(f"{stable}.provenance.json").write_bytes(
        Path(f"{acquired.path}.provenance.json").read_bytes()
    )
    calls = 0

    def handler(http_request):
        nonlocal calls
        calls += 1
        return _gzip_response(http_request)

    def fail_publish(*_args):
        raise OSError("read-only cache")

    monkeypatch.setattr(exact_cache, "publish", fail_publish)
    with _client(handler) as client, pytest.raises(distribution.CacheWriteFailure):
        distribution.acquire(request, cache_dir=tmp_path / "target", http_client=client)
    assert calls == 0


def test_abandoned_cleanup_waits_for_live_writer_lock(tmp_path):
    request = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    stable = distribution._cache_path(
        tmp_path, request.identity, distribution.DistributionSource.IN_MEMORY
    )
    context = _spawn_context()
    ready = context.Queue()
    release = context.Event()
    process = context.Process(
        target=_process_hold_lock, args=(str(stable), ready, release)
    )
    process.start()
    orphan = Path(ready.get(timeout=10))
    assert orphan.is_dir()
    with pytest.raises(exact_cache.CacheLockTimeout):
        with exact_cache.entry_lock(stable, 0.05):
            exact_cache.cleanup_abandoned(stable)
    assert orphan.is_dir()
    fallback_source = tmp_path / "fallback.SP3"
    fallback_source.write_bytes(_sp3_bytes())
    two_sources = _sp3_request(
        distribution.Distribution.in_memory(_sp3_bytes()),
        distribution.Distribution.local_file(fallback_source),
    )
    with pytest.raises(distribution.CacheWriteFailure):
        distribution.acquire(
            two_sources,
            cache_dir=tmp_path,
            cache_lock_timeout_s=0.05,
        )
    release.set()
    process.join(10)
    assert process.exitcode == 0
    with exact_cache.entry_lock(stable, 1.0):
        exact_cache.cleanup_abandoned(stable)
    assert not orphan.exists()
