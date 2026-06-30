#!/usr/bin/env python3
"""Fleet propagation benchmark: serial vs GIL-free parallel batch.

Sidereon's ``propagate_batch`` releases the GIL for the whole compute
(``Python::allow_threads``) and runs the independent satellites across a rayon
thread pool, so one call can use multiple cores and also run concurrently when
driven from several Python threads. This script measures that against
single-threaded baselines, including Skyfield and the sgp4 C library.

This script measures, for a fleet of N satellites over M epochs across a day:

  A. sidereon serial          ``propagate_batch(..., parallel=False)``  (TEME)
  B. sidereon rayon-parallel  ``propagate_batch(..., parallel=True)``   (TEME)
                              GIL released for the whole compute.
  C. Skyfield                 per-satellite ``EarthSatellite.at(t)`` loop (GCRS)
                              a per-satellite fleet workflow.
  D. sgp4 SatrecArray         C-vectorized ``SatrecArray.sgp4`` (TEME)
                              the raw C-backed SGP4 lower bound, single thread.

Plus a GIL-release demonstration: the *serial* sidereon kernel driven from T
Python threads (each thread propagates a fleet slice with ``parallel=False``),
alongside the Skyfield loop driven from the same T threads. The serial kernel
scales across threads because the GIL is released during the Rust compute; a
GIL-holding Python loop does not.

Calibration note (read the README): this measures the parallel and cores-scaling
path, not raw scalar SGP4. Skyfield and ``sgp4`` use a C SGP4, which runs faster
per element than sidereon's scalar serial path. Every number below is measured
with ``time.perf_counter`` (warmup + best-of-3).

Run from ``bindings/python`` with the binding installed (``maturin develop``):

    python bench/bench_188.py                  # default 1000/5000/10000 sats
    python bench/bench_188.py --sizes 1000 5000
    python bench/bench_188.py --threads 8
"""

import argparse
import datetime as dt
import json
import os
import threading
import time

import numpy as np
import sidereon

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, os.pardir, "tests", "fixtures", "batch_fleet.json")

# A day of epochs at a 10-minute cadence (144 intervals + the start epoch).
EPOCH_STEP_SECONDS = 600
N_EPOCHS = 145
WARMUP = 1
REPEATS = 3
UNIX_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def load_base_tles():
    """The committed Vallado SGP4-VER subset (8 sats sharing TLE epoch 06176)."""
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    tles = [(t["line1"], t["line2"]) for t in fx["tles"]]
    return tles, fx["opsmode"], fx["epochs_unix_us"][0]


def make_fleet(base_tles, n):
    """Tile the committed real TLEs up to N satellites (documented in README)."""
    return [base_tles[i % len(base_tles)] for i in range(n)]


def make_epoch_grids(start_unix_us):
    """Return the epoch grid in three shapes the libraries each want.

    - ``epochs_us``: int64 unix microseconds for sidereon.propagate_batch.
    - ``datetimes``: aware UTC datetimes for Skyfield's timescale.
    - ``(jd, fr)``: split Julian date arrays for sgp4's SatrecArray.
    """
    base = UNIX_EPOCH + dt.timedelta(microseconds=int(start_unix_us))
    datetimes = [
        base + dt.timedelta(seconds=i * EPOCH_STEP_SECONDS) for i in range(N_EPOCHS)
    ]
    epochs_us = np.array(
        [start_unix_us + i * EPOCH_STEP_SECONDS * 1_000_000 for i in range(N_EPOCHS)],
        dtype=np.int64,
    )
    return epochs_us, datetimes


def best_of(fn, warmup=WARMUP, repeats=REPEATS):
    """Run ``fn`` warmup times (discarded), then return the best of ``repeats``.

    Best-of (minimum) is the standard timing estimator: it is the run least
    perturbed by the OS scheduler, GC, and other noise.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples), samples


def fmt(seconds):
    return f"{seconds * 1000:9.2f} ms"


def bench_size(base_tles, opsmode, epochs_us, datetimes, n, n_threads):
    fleet = make_fleet(base_tles, n)
    n_prop = n * N_EPOCHS
    print(f"\n=== N = {n} satellites x {N_EPOCHS} epochs = {n_prop:,} propagations ===")

    # --- A. sidereon serial -------------------------------------------------
    a, _ = best_of(
        lambda: sidereon.propagate_batch(
            fleet, epochs_us, opsmode=opsmode, parallel=False
        )
    )
    print(f"  A sidereon serial            {fmt(a)}")

    # --- B. sidereon rayon-parallel + GIL released --------------------------
    b, _ = best_of(
        lambda: sidereon.propagate_batch(
            fleet, epochs_us, opsmode=opsmode, parallel=True
        )
    )
    print(f"  B sidereon rayon-parallel    {fmt(b)}   (GIL released)")

    # --- C. Skyfield per-satellite loop -------------------------------------
    c = None
    try:
        from skyfield.api import EarthSatellite, load

        ts = load.timescale(builtin=True)
        sky_times = ts.utc(datetimes)
        sky_sats = [EarthSatellite(l1, l2, ts=ts) for l1, l2 in fleet]

        def run_skyfield():
            out = np.empty((n, N_EPOCHS, 3))
            for i, sat in enumerate(sky_sats):
                out[i] = sat.at(sky_times).position.km.T
            return out

        c, _ = best_of(run_skyfield)
        print(f"  C Skyfield loop (GCRS)       {fmt(c)}")
    except ImportError:
        print("  C Skyfield loop              [skyfield not installed]")

    # --- D. sgp4 SatrecArray, C-vectorized single thread (TEME) -------------
    d = None
    try:
        from sgp4.api import Satrec, SatrecArray, jday

        jds, frs = [], []
        for t in datetimes:
            jd, fr = jday(
                t.year,
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second + t.microsecond * 1e-6,
            )
            jds.append(jd)
            frs.append(fr)
        jds = np.array(jds)
        frs = np.array(frs)
        sgp4_sats = [Satrec.twoline2rv(l1, l2) for l1, l2 in fleet]
        sgp4_arr = SatrecArray(sgp4_sats)

        d, _ = best_of(lambda: sgp4_arr.sgp4(jds, frs))
        print(f"  D sgp4 SatrecArray (TEME)    {fmt(d)}   (C, single thread)")
    except ImportError:
        print("  D sgp4 SatrecArray           [sgp4 not installed]")

    print(f"  -> serial/parallel speedup:  {a / b:5.2f}x  on {os.cpu_count()} cores")
    if c is not None:
        print(f"  -> sidereon-parallel vs Skyfield loop:    {c / b:5.2f}x")
    if d is not None:
        print(f"  -> sidereon-parallel vs sgp4 SatrecArray: {d / b:5.2f}x")
        print(
            f"  -> sidereon-serial   vs sgp4 SatrecArray: {d / a:5.2f}x  (scalar calib)"
        )

    return {
        "n": n,
        "n_epochs": N_EPOCHS,
        "serial_s": a,
        "parallel_s": b,
        "skyfield_s": c,
        "sgp4_satrecarray_s": d,
        "speedup": a / b,
    }


def bench_gil_scaling(base_tles, opsmode, epochs_us, datetimes, n, n_threads):
    """Demonstrate the GIL is released: drive the *serial* sidereon kernel from
    T Python threads and show wall-clock throughput scales ~T, because each
    thread's Rust compute runs with the GIL released. Contrast with the Skyfield
    Python loop driven from the same T threads, which cannot scale (GIL held)."""
    fleet = make_fleet(base_tles, n)
    chunks = [fleet[i::n_threads] for i in range(n_threads)]
    print(
        f"\n=== GIL-release thread scaling: N = {n} sats, T = {n_threads} threads ==="
    )
    print("    (each thread runs the SERIAL kernel; speedup over 1 thread is")
    print("     purely the effect of releasing the GIL, not rayon)")

    # 1 thread (serial kernel, whole fleet).
    one, _ = best_of(
        lambda: sidereon.propagate_batch(
            fleet, epochs_us, opsmode=opsmode, parallel=False
        )
    )

    def threaded_sidereon():
        def worker(chunk):
            sidereon.propagate_batch(chunk, epochs_us, opsmode=opsmode, parallel=False)

        threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    many, _ = best_of(threaded_sidereon)
    print(f"  sidereon serial kernel, 1 thread:   {fmt(one)}")
    print(
        f"  sidereon serial kernel, {n_threads} threads:  {fmt(many)}   "
        f"-> {one / many:5.2f}x  (GIL released)"
    )

    # Skyfield in threads, same shape, for contrast.
    try:
        from skyfield.api import EarthSatellite, load

        ts = load.timescale(builtin=True)
        sky_times = ts.utc(datetimes)
        sky_chunks = [
            [EarthSatellite(l1, l2, ts=ts) for l1, l2 in chunk] for chunk in chunks
        ]
        sky_all = [EarthSatellite(l1, l2, ts=ts) for l1, l2 in fleet]

        def sky_one():
            for sat in sky_all:
                sat.at(sky_times).position.km

        sky1, _ = best_of(sky_one)

        def sky_threaded():
            def worker(sats):
                for sat in sats:
                    sat.at(sky_times).position.km

            threads = [threading.Thread(target=worker, args=(c,)) for c in sky_chunks]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        skyT, _ = best_of(sky_threaded)
        print(f"  Skyfield loop, 1 thread:            {fmt(sky1)}")
        print(
            f"  Skyfield loop, {n_threads} threads:           {fmt(skyT)}   "
            f"-> {sky1 / skyT:5.2f}x  (GIL held)"
        )
    except ImportError:
        print("  Skyfield loop                       [skyfield not installed]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
        help="fleet sizes (number of satellites)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="thread count for the GIL-release demonstration",
    )
    args = parser.parse_args()

    base_tles, opsmode, start_unix_us = load_base_tles()
    epochs_us, datetimes = make_epoch_grids(start_unix_us)

    print("Sidereon fleet-propagation benchmark")
    print(
        f"  cores: {os.cpu_count()}   epochs/sat: {N_EPOCHS} over 1 day "
        f"({EPOCH_STEP_SECONDS}s cadence)   opsmode: {opsmode!r}"
    )
    print(f"  base fleet: {len(base_tles)} committed Vallado SGP4-VER TLEs, tiled to N")
    print(f"  timing: time.perf_counter, warmup={WARMUP}, best-of-{REPEATS}")

    results = []
    for n in args.sizes:
        results.append(
            bench_size(base_tles, opsmode, epochs_us, datetimes, n, args.threads)
        )

    bench_gil_scaling(
        base_tles, opsmode, epochs_us, datetimes, max(args.sizes), args.threads
    )

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
