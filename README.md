# sidereon

GNSS and astrodynamics for Python: propagate satellites, predict passes, solve
precise positions (SPP / RTK / PPP), and convert between coordinate frames and
time scales — checked against the references the field trusts (Vallado, Skyfield,
IGS, IERS).

Under the hood it's a Rust engine compiled into the wheel, so it's fast and the
only runtime dependency is numpy. You just `pip install sidereon`.

## Install

```
pip install sidereon
```

```python
import sidereon
print(sidereon.__version__)
```

## Quickstart: when does the ISS fly over you?

No data files, no setup — give it a two-line element set and a ground station,
and ask when the satellite is above the horizon.

```python
import sidereon
from datetime import datetime, timedelta, timezone

# Real orbital elements (grab fresh ones from CelesTrak any time).
iss = sidereon.Tle(
    "1 25544U 98067A   26178.50947090  .00006280  00000+0  12016-3 0  9996",
    "2 25544  51.6322 248.9966 0004278 238.4942 121.5629 15.49454046573359",
)

# A ground station: latitude, longitude in degrees (altitude in metres, optional).
berkeley = sidereon.GroundStation(37.87, -122.27)

# Every pass above 10° over the next 24 hours.
now = datetime.now(timezone.utc)
us = lambda t: int(t.timestamp() * 1_000_000)  # epochs are UTC unix microseconds
passes = iss.find_passes(
    berkeley, us(now), us(now + timedelta(days=1)), elevation_mask_deg=10.0
)

for p in passes:
    rise = datetime.fromtimestamp(p.aos_unix_us / 1e6, timezone.utc)
    print(f"{rise:%H:%M} UTC · {p.duration_s / 60:4.1f} min · peak {p.max_elevation_deg:2.0f}°")
```

`Tle` also gives you `propagate()` (TEME state arcs as numpy arrays) and
`look_angles()` (azimuth/elevation/range over a time grid). Everything that
takes time takes UTC unix microseconds and returns numpy arrays.

## Precise positioning

The positioning engine is the other half of the library: feed it pseudoranges
and a precise-ephemeris product and it returns a least-squares fix.

```python
import sidereon

sp3 = sidereon.load_sp3(open("igs_product.sp3", "rb").read())

config = sidereon.SppConfig(
    observations=[
        sidereon.SppObservation("G01", 21_000_123.4),  # PRN, pseudorange (m)
        sidereon.SppObservation("G08", 22_517_889.1),
        # ...more satellites
    ],
    t_rx_j2000_s=...,          # receiver time
    t_rx_second_of_day_s=...,
    day_of_year=...,
    initial_guess=[0.0, 0.0, 0.0, 0.0],
    corrections=sidereon.SppCorrections(ionosphere=True, troposphere=True),
    with_geodetic=True,
)

fix = sidereon.solve_spp(sp3, config)
print(fix.position)   # numpy [x, y, z] ECEF metres
print(fix.geodetic)   # (lat_rad, lon_rad, height_m)
print(fix.used_sats)  # satellites that contributed
```

`solve_rtk_float`, `solve_rtk_fixed`, `solve_ppp_float`, and `solve_ppp_fixed`
follow the same shape — typed config in, a result object with numpy positions,
scalar attributes, and enum statuses out. Need the products? `sidereon.data`
fetches and caches SP3, RINEX, and IONEX from the public archives.

## What's in the box

- **Orbits** — SGP4/TLE and OMM, numerical propagation, passes, look angles, visibility
- **Frames & time** — TEME ↔ GCRS ↔ ITRS, GMST/GAST, geodetic ↔ ECEF, UTC/TT/TDB/UT1
- **Bodies** — Sun/Moon positions, eclipse, plus JPL SPK (DAF/.bsp) kernels
- **Positioning** — SPP, RTK (float/fixed), PPP (float/fixed), DOP, velocity
- **GNSS data** — SP3, RINEX (obs/nav/clock), CRINEX, ANTEX, IONEX, broadcast ephemeris
- **Space situational awareness** — conjunction/TCA screening, collision probability, CDM, covariance
- **RF** — link budget (FSPL, EIRP, C/N0, antenna gain)

The binding adds no modeling of its own: every result is exactly what the engine
computes, returned as numpy arrays, typed objects, and real Python exceptions
(`sidereon.SidereonError` and friends). Full signatures live in the bundled type
stubs (`sidereon/__init__.pyi`).

## Other languages

sidereon is one validated engine with first-class interfaces in **Rust**,
**Python**, **C**, **Elixir**, and **WebAssembly** — same numbers everywhere.
See the live demo and docs at [sidereon.dev](https://sidereon.dev).

## How it's validated

The SGP4 propagator is a Rust port of David Vallado's reference implementation,
bit-exact to it. Frames and time are checked against Skyfield and IERS; the
positioning stack is checked against IGS products. The wheel links the Rust
`sidereon-core` engine statically, so there's no separate native install.

*Building from source (for contributors): `pip install maturin`, then
`maturin develop` from the repo. Tests: `pytest`.*
