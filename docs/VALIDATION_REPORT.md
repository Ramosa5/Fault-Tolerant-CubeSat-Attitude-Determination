# Phase 1 validation report

This report records the default Phase-1 run packaged with the project.

## Configuration

- Start epoch: 2026-01-01 00:00:00 UTC
- Nominal semi-major-axis altitude: 500 km
- Eccentricity: 0
- Inclination: 97.4 deg
- RAAN: 90 deg
- Propagation duration: 2 orbital periods
- Sampling: 5 s
- Gravity: two-body + J2
- Environment: low-precision Sun vector, cylindrical eclipse, tilted centered magnetic dipole

The RAAN is deliberately selected for the demonstration configuration so the short run
contains both eclipse and sunlit intervals and a broad range of Sun--magnetic-vector
geometry. It is a research/diagnostic scenario, not a claim about the final mission orbit.

## Default-run numerical results

- Samples: 2272
- Kepler period: 94.6163 min
- Mean osculating altitude: 495.136 km
- Altitude range: 492.672--500.000 km
- Mean speed: 7.6128 km/s
- Eclipse fraction: 37.6%
- Magnetic field magnitude: 24.88--49.90 microtesla
- Sun--magnetic-vector angle: 9.5--166.9 deg
- Conserved-energy relative span: 3.87e-10
- Conserved h_z relative span: 1.94e-10

The altitude variation in the J2 run is an osculating radial variation caused by the
perturbed gravity model; it is not integrator drift. The separate two-body validation
configuration maintains the configured 500 km circular altitude to numerical precision.

## Automated tests

`pytest -q` result at packaging time:

```text
7 passed
```

Tests cover:

1. reasonable 500 km orbital period;
2. LVLH DCM orthonormality and handedness;
3. Sun-vector unit norm and Earth--Sun distance sanity;
4. plausible dipole-field magnitude in LEO;
5. two-body mechanical-energy conservation;
6. finite environment outputs and eclipse occurrence;
7. conserved energy and h_z for the axisymmetric J2 propagation.

## Visual inspection sequence

1. `01_orbit_3d.png`
2. `02_altitude_speed.png`
3. `04_ground_track.png`
4. `05_reference_vectors_lvlh.png`
5. `06_eclipse_and_geometry.png`
6. `07_orbit_invariants.png`
7. `phase1_dashboard.html`
