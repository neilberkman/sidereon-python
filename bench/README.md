# Fleet propagation benchmark

`propagate_batch` marshals a set of TLEs into plain Rust data, runs the whole
compute inside `Python::allow_threads` (the GIL is released for the duration, and
no Python object is touched in that window), and fans the independent satellites
across a rayon thread pool. Because the GIL is released, the same call also runs
concurrently when driven from multiple Python threads.

This benchmark measures that batch against single-threaded baselines, including
Skyfield and the `sgp4` C library, on a fleet of N satellites over M epochs. The
numbers below are measurements on one machine, not claims of superiority.

## What is measured

For a fleet of N satellites over M = 145 epochs spanning one day (10-minute
cadence):

|   | Path                                                     | Frame | Threading               |
| - | -------------------------------------------------------- | ----- | ----------------------- |
| A | sidereon serial `propagate_batch(parallel=False)`        | TEME  | 1 core                  |
| B | sidereon rayon-parallel `propagate_batch(parallel=True)` | TEME  | all cores, GIL released |
| C | Skyfield `EarthSatellite.at(t)` per-satellite loop       | GCRS  | 1 core, GIL held        |
| D | `sgp4` `SatrecArray.sgp4` C-vectorized                   | TEME  | 1 core, C               |

Plus a GIL-release measurement: the serial sidereon kernel driven from T Python
threads, each propagating a slice of the fleet. With rayon off, any speedup over
one thread comes only from the GIL being released during the Rust compute. The
Skyfield loop is driven from the same T threads alongside it.

## How to read these numbers

Stated plainly, including where sidereon is slower:

- sidereon's scalar SGP4 is a pure-Rust port and is slower per element than the C
  SGP4 used by `sgp4` and Skyfield: sidereon serial is about 0.3x of `sgp4`'s
  `SatrecArray` (A vs D).
- The closest single-thread, same-frame baseline is D (`sgp4` `SatrecArray`, C
  SGP4, TEME). sidereon's GIL-released rayon batch (B) runs about 1.2x of it by
  using all cores.
- Against the Skyfield fleet loop (C), B runs about 2.3x; part of that gap is that
  C also does a TEME to GCRS rotation the batch does not.
- The serial-to-parallel speedup is about 4.2x on 10 cores, not 10x: building the
  numpy result arrays from the Rust Vecs runs serially under the GIL after the
  compute, which Amdahl-limits a single call.
- In the thread test, the serial kernel from 10 threads runs about 4.7x faster
  than from 1 thread, because `allow_threads` lets the threads' Rust computes run
  concurrently. The GIL-held Skyfield loop from 10 threads does not speed up.

## Reproduce

```sh
cd bindings/python
maturin develop                 # build + install the extension into the venv
pip install skyfield sgp4       # the C-backed comparison baselines
python bench/bench_fleet.py       # default sizes 1000 / 5000 / 10000
```

The fleet is built by tiling the committed Vallado SGP4-VER subset
(`tests/fixtures/batch_fleet.json`: 8 real TLEs sharing epoch 06176) up to N. This
keeps the benchmark reproducible with no network access; SGP4 does the same
per-satellite work regardless of how the fleet was assembled. Timing uses
`time.perf_counter` with 1 warmup and best-of-3 (the minimum is the estimator
least perturbed by scheduler/GC noise). Absolute milliseconds vary by machine and
run; the ratios are stable.

## Measured run (verbatim)

Machine: Apple M4, 10 cores, macOS 15.3.1, Python 3.12.11, numpy 2.4.6, skyfield
1.54, sgp4 2.25.

```
Sidereon fleet-propagation benchmark
  cores: 10   epochs/sat: 145 over 1 day (600s cadence)   opsmode: 'improved'
  base fleet: 8 committed Vallado SGP4-VER TLEs, tiled to N
  timing: time.perf_counter, warmup=1, best-of-3

=== N = 1000 satellites x 145 epochs = 145,000 propagations ===
  A sidereon serial               128.66 ms
  B sidereon rayon-parallel        30.76 ms   (GIL released)
  C Skyfield loop (GCRS)           69.31 ms
  D sgp4 SatrecArray (TEME)        38.43 ms   (C, single thread)
  -> serial/parallel speedup:   4.18x  on 10 cores
  -> sidereon-parallel vs Skyfield loop:     2.25x
  -> sidereon-parallel vs sgp4 SatrecArray:  1.25x
  -> sidereon-serial   vs sgp4 SatrecArray:  0.30x  (scalar calibration)

=== N = 5000 satellites x 145 epochs = 725,000 propagations ===
  A sidereon serial               649.20 ms
  B sidereon rayon-parallel       154.16 ms   (GIL released)
  C Skyfield loop (GCRS)          349.65 ms
  D sgp4 SatrecArray (TEME)       190.19 ms   (C, single thread)
  -> serial/parallel speedup:   4.21x  on 10 cores
  -> sidereon-parallel vs Skyfield loop:     2.27x
  -> sidereon-parallel vs sgp4 SatrecArray:  1.23x
  -> sidereon-serial   vs sgp4 SatrecArray:  0.29x  (scalar calibration)

=== N = 10000 satellites x 145 epochs = 1,450,000 propagations ===
  A sidereon serial              1308.18 ms
  B sidereon rayon-parallel       309.24 ms   (GIL released)
  C Skyfield loop (GCRS)          699.68 ms
  D sgp4 SatrecArray (TEME)       381.15 ms   (C, single thread)
  -> serial/parallel speedup:   4.23x  on 10 cores
  -> sidereon-parallel vs Skyfield loop:     2.26x
  -> sidereon-parallel vs sgp4 SatrecArray:  1.23x
  -> sidereon-serial   vs sgp4 SatrecArray:  0.29x  (scalar calibration)

=== GIL-release thread scaling: N = 10000 sats, T = 10 threads ===
    (each thread runs the SERIAL kernel; speedup over 1 thread is
     purely the effect of releasing the GIL, not rayon)
  sidereon serial kernel, 1 thread:     1299.85 ms
  sidereon serial kernel, 10 threads:     277.00 ms   ->  4.69x  (GIL released)
  Skyfield loop, 1 thread:               690.88 ms
  Skyfield loop, 10 threads:              815.39 ms   ->  0.85x  (GIL held)

Done.
```

### Reading the numbers

- Cores scaling (B vs A): the rayon batch is about 4.2x the serial path on 10
  cores at every size, stable as N grows from 1k to 10k.
- Vs C SGP4 (B vs D): the GIL-released rayon batch runs about 1.23x of
  single-threaded `sgp4` `SatrecArray`, even though sidereon's scalar SGP4 is
  slower per element; the cores account for the difference.
- GIL released: running the serial kernel from 10 Python threads scales about
  4.7x, because `allow_threads` lets the threads' Rust computes run concurrently.
  The GIL-held loop from 10 threads does not scale.
