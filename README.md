# sidereon

GNSS and astrodynamics for Python, with answers you can trust.

`sidereon` is the Python interface to the sidereon engine: a single GNSS and
astrodynamics core, written in Rust, exposed here as an idiomatic Python
package. You get orbit propagation, precise positioning, frames and time,
ephemeris handling, and format parsing through plain Python objects and numpy
arrays. The engine is reference-validated (SGP4 is bit-exact to Vallado's
implementation; frames and time check against Skyfield and IERS; the
positioning stack checks against IGS products), so the numbers match the
sources the field already trusts.

The Rust core is compiled into the wheel and linked statically, so the package
is fast and the only runtime dependency is numpy. There is no separate native
install.

## Install

```
pip install sidereon
```

```python
import sidereon
print(sidereon.__version__)
```

## Example: where is the ISS in the sky right now?

No data files and no setup: give it a two-line element set and a ground
station, and ask for the look angles. Everything that takes time takes unix
microseconds (int64) and arrays propagate in one call.

```python
import numpy as np
import sidereon

tle = sidereon.Tle(
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009",
    "2 25544  51.6400 208.8657 0002644 250.3037 109.7782 15.49560812999990",
)
station = sidereon.GroundStation(latitude_deg=51.5, longitude_deg=-0.1, altitude_m=10.0)

# Epochs are unix microseconds (int64); arrays propagate in one call.
epochs_us = np.array([1_704_110_400_000_000], dtype=np.int64)
look = tle.look_angles(station, epochs_us)
print(look.azimuth_deg, look.elevation_deg, look.range_km)
```

`Tle` also gives you `propagate()` (TEME state arcs as numpy arrays) and
`find_passes()` (rise/set/peak over a window). The positioning side has the same
shape: a typed config in, a result object with numpy positions and scalar
attributes out.

```python
import sidereon

sp3 = sidereon.load_sp3(open("igs_product.sp3", "rb").read())

config = sidereon.SppConfig(
    observations=[
        sidereon.SppObservation("G08", 23_825_519.8),  # PRN, pseudorange (m)
        sidereon.SppObservation("G10", 22_717_690.1),
        # ...more satellites
    ],
    t_rx_j2000_s=646_272_000.0,
    t_rx_second_of_day_s=43_200.0,
    day_of_year=176.5,
    initial_guess=[4_500_000, 500_000, 4_500_000, 0.0],
    corrections=sidereon.SppCorrections(ionosphere=True, troposphere=True),
    with_geodetic=True,
)

solution = sidereon.solve_spp(sp3, config)
print(solution.position)     # numpy [x, y, z] ECEF metres
print(solution.rx_clock_s)   # receiver clock bias, seconds
```

## Capabilities

The Python package mirrors the full breadth of the engine.

- **Orbit propagation:** SGP4/SDP4 from TLE/OMM, numerical propagation with a
  composable force model (spherical-harmonic geopotential to selectable degree
  and order, Sun/Moon third-body, solar radiation pressure, relativistic
  correction, atmospheric drag) and orbital-decay estimation with a post-decay
  validity latch, batch and constellation arcs, pass prediction,
  look angles, and coverage analysis.
- **Orbital mechanics:** classical, equinoctial, and modified equinoctial
  elements, anomaly conversions and Kepler propagation, Lambert transfers,
  initial orbit determination (IOD), batch least-squares orbit fitting against
  precise ephemerides (including terrestrial-frame SP3 products through the
  Earth-orientation chain) with a per-satellite residual ledger, and relative
  motion in RIC/RTN/LVLH frames with Clohessy-Wiltshire propagation.
- **GNSS positioning:** single-point positioning (SPP), RTK (float and fixed),
  PPP (float and fixed), DGNSS, a robust solve driver, and DOP.
- **Integrity and error bounds:** RAIM fault detection and exclusion,
  multi-constellation ARAIM protection levels, SBAS protection levels
  (DO-229), per-observation reliability (minimal detectable bias,
  internal/external), observability classification of every solution (rank,
  redundancy, conditioning), and covariance-derived error metrics (CEP, R95,
  SEP, error ellipse) that report wide or flagged bounds for weak geometry
  rather than fabricated confidence.
- **GNSS corrections and products:** SBAS and RTCM SSR corrections applied to
  broadcast ephemeris, Bias-SINEX code and phase biases (DCB/OSB), Klobuchar
  and NeQuick-G ionosphere, IONEX maps, troposphere models, and NTRIP client
  stream handling.
- **Ephemeris and time:** broadcast ephemeris and precise SP3 products, JPL SPK
  (DAF/.bsp) kernels, uniform satellite-state sampling across broadcast and
  precise sources with batched multi-satellite interpolation, scale-aware time
  (UTC/TAI/TT/TDB/UT1 and the GNSS system times) with leap-second handling,
  and Earth orientation parameters (EOP).
- **Timing and clocks:** Allan-family stability analysis (ADEV/MDEV/HDEV/TDEV)
  and power-law clock-noise identification with a five-coefficient fit
  (IEEE 1139).
- **Estimation and detection:** scalar Kalman and alpha-beta trackers,
  innovation gating, robust statistics, CFAR detection thresholds, and
  source localization (ToA/TDOA) from arrival times at known sensors.
- **Geodesy and monitoring:** geodesic direct and inverse problems (Karney),
  an epoch-aware terrestrial reference frame catalog with published ITRF and
  ETRF Helmert parameter sets, robust station velocity (MIDAS), trajectory
  fitting with seasonal terms and offsets, step detection, network motion
  fields with common-mode removal, and repeating-geometry (sidereal)
  filtering.
- **Geometry and events:** reference frames, geodetic and ECEF conversions,
  look angles, eclipse, conjunction screening with collision probability, and
  angular geometry (separation, position angle, phase angle, beta angle).
- **Observation and almanac:** apparent places for the Sun, Moon, and any SPK
  body (astrometric and apparent RA/Dec plus az/el, with refraction and polar
  motion), sub-solar and sub-observer points, the terminator, parallactic
  angle, satellite visual magnitude, moonrise/moonset, seasons, moon phases,
  planetary events, meridian transits, and lunar and solar eclipses.
- **Observation quality:** RINEX observation QC (completeness, multipath,
  cycle slips), carrier-phase combinations, and Hatch smoothing.
- **Terrain:** DTED elevation lookup with batch probes, a memory-mappable
  terrain store, and geoid height conversion from EGM96 and EGM2008 grids.
- **RF:** link budget (FSPL, EIRP, C/N0, antenna gain).
- **Formats:** parse and serialize TLE/OMM, CCSDS OEM/OPM/CDM/TDM, RINEX,
  CRINEX, SP3, IONEX, ANTEX, Bias-SINEX, SBAS logs, RTCM, and NMEA.
- **Data acquisition:** the `sidereon.data` module downloads and caches GNSS
  products (SP3 and IONEX from IGS/MGEX analysis centers, including merged
  multi-center SP3) and DTED terrain tiles.

The binding adds no modeling of its own: every result is exactly what the engine
computes, returned as numpy arrays, typed objects, and real Python exceptions
(`sidereon.SidereonError` and friends). Full signatures live in the bundled type
stubs (`sidereon/__init__.pyi`).

## One engine, every language

sidereon is one validated core with first-class interfaces, so the numbers are
the same everywhere:

- [sidereon](https://github.com/neilberkman/sidereon): the Rust core and engine
- [sidereon-c](https://github.com/neilberkman/sidereon-c): C interface
- [sidereon-ex](https://github.com/neilberkman/sidereon-ex): Elixir interface
- [sidereon-wasm](https://github.com/neilberkman/sidereon-wasm): WebAssembly interface

See the live demo and docs at [sidereon.dev](https://sidereon.dev).

## Building from source

For contributors: `pip install maturin`, then `maturin develop` from the repo.
Run the tests with `pytest`.

## License

MIT.
