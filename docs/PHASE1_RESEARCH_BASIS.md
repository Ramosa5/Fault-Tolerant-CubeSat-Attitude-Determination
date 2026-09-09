# Phase 1 research basis and modelling decisions

Phase 1 is the reference environment for the later attitude-determination, FDIR and
virtual-sensor experiments.  It deliberately stops before spacecraft attitude dynamics
and sensor noise are introduced.

## Why these outputs exist

The literature review used for the thesis indicates that low-cost CubeSat attitude
determination commonly fuses a gyroscope with magnetic and Sun-vector observations.
The RAX work is a representative flight example of this architecture, while the review
also identifies eclipse as a predictable loss of Sun-sensor information and poor
reference-vector geometry as a source of degraded attitude observability.

For that reason Phase 1 produces, at every simulation sample:

- orbit state in ECI and ECEF;
- diagnostic latitude/longitude/altitude;
- Sun direction in ECI and LVLH;
- magnetic-field direction/magnitude in ECI and LVLH;
- eclipse state;
- angle between the Sun and magnetic reference vectors;
- numerical orbit invariants for correctness checks.

These are the clean environmental truths that Phase 2/3 will transform into body-frame
measurements and corrupt with realistic sensor noise/faults.

## Deliberately lightweight models

- Orbit: numerical two-body propagation with optional J2 perturbation.
- Sun: low-precision analytical ephemeris.
- Eclipse: cylindrical umbra approximation.
- Magnetic field: centered tilted dipole.
- Ground track: spherical-Earth latitude/longitude diagnostic.

They are intended for repeatable estimator/FDIR research rather than mission-grade
navigation.  The interfaces are isolated so a higher-fidelity Sun/IGRF/SGP4 model can be
substituted later without changing downstream sensor, EKF, or ML code.

## Literature connections

Representative sources already recorded in the thesis bibliography/literature matrix:

- Springmann et al. / Springmann & Cutler — low-cost CubeSat gyro + magnetometer + Sun
  sensor attitude-determination architecture and RAX flight results.
- Mmopelwa et al. — adaptive EKF behaviour during Sun-sensor loss/eclipses.
- Hajiyev & Cilden-Guler — reduced-sensor/eclipsed attitude-estimation modes.
- Frezza et al. — environmental effects can perturb Sun-sensor measurements.
- Markley & Crassidis — spacecraft attitude, vector observations and reference frames.

## Phase-1 acceptance criteria

Before Phase 2 begins:

1. all automated tests pass;
2. the 3D orbit and ground track are visually plausible;
3. altitude/speed remain physically plausible for the configured orbit;
4. Sun-vector norm remains one;
5. magnetic-field magnitude remains in a plausible LEO range for the low-order model;
6. LVLH transforms remain orthonormal;
7. eclipse intervals are visible in the dashboard;
8. Sun–magnetic-vector geometry varies continuously except for coordinate wrap effects;
9. CSV and summary JSON are reproducible from the YAML configuration.
