# CubeSat FDIR Master Thesis — Phase 1 rebuilt

Research-grade foundation for:

**On-board Fault-Tolerant Attitude Determination for a CubeSat Using Lightweight Machine Learning for Sensor Fault Detection and Loss Recovery**

Phase 1 provides the orbital/environmental truth needed later by attitude dynamics,
gyroscope/magnetometer/Sun-sensor models, MEKF, fault injection, FDIR, and ML virtual
sensors.

## What changed from the first Phase-1 prototype

The old prototype mainly demonstrated a circular orbit and a few plots. This rebuilt
version adds:

- configurable Keplerian initial orbit;
- numerical propagation with optional J2;
- explicit ECI, ECEF and LVLH frame handling;
- approximate Sun ephemeris;
- eclipse detection;
- tilted-dipole geomagnetic reference vector;
- Sun–magnetic-vector geometry diagnostic;
- orbital invariants for numerical correctness checks;
- 7 automatic static diagnostic figures;
- an interactive Plotly dashboard;
- CSV + JSON outputs for reproducibility;
- automated tests for physics/geometry sanity;
- interfaces intended to feed the later body-frame sensor simulation directly.

## Project tree

```text
cubesat_fdir_phase1_rebuilt/
├── main.py
├── requirements.txt
├── README.md
├── configs/
│   └── phase1.yaml
├── docs/
│   └── PHASE1_RESEARCH_BASIS.md
├── src/
│   ├── simulation/
│   │   ├── constants.py
│   │   ├── orbit_model.py
│   │   ├── frames.py
│   │   ├── environment.py
│   │   └── pipeline.py
│   └── visualization/
│       ├── static_plots.py
│       └── dashboard.py
├── tests/
│   └── test_phase1.py
└── data/
    ├── generated/
    │   └── phase1_environment.csv
    └── results/
        ├── phase1_summary.json
        ├── 01_orbit_3d.png
        ├── 02_altitude_speed.png
        ├── 03_eci_position.png
        ├── 04_ground_track.png
        ├── 05_reference_vectors_lvlh.png
        ├── 06_eclipse_and_geometry.png
        ├── 07_orbit_invariants.png
        └── phase1_dashboard.html
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Use another configuration:

```bash
python main.py --config configs/phase1.yaml
```

Run tests:

```bash
pytest -q
```

## Default research scenario

The default YAML uses a representative, configurable 500 km near-polar circular LEO with J2 enabled. Its RAAN is deliberately chosen so the two-orbit demonstration contains eclipse and a broad range of Sun--magnetic-vector geometry. It is **not** claimed to be the final mission orbit. Change altitude, inclination, RAAN, eccentricity and start epoch directly in the YAML once the mission scenario is fixed.

## How to inspect correctness

Start with the figures in this order:

1. `01_orbit_3d.png` — does the orbital plane/orbit look physically plausible?
2. `02_altitude_speed.png` — are altitude and speed smooth and plausible?
3. `04_ground_track.png` — does the inclination match the geographic latitude range?
4. `05_reference_vectors_lvlh.png` — do Sun and magnetic vectors vary continuously?
5. `06_eclipse_and_geometry.png` — are natural Sun-loss intervals and vector geometry visible?
6. `07_orbit_invariants.png` — is numerical drift acceptably small?
7. `phase1_dashboard.html` — interactive combined inspection.

## Interface to Phase 2/3

The CSV columns `sun_eci_*`, `mag_eci_*`, `r_eci_*`, `v_eci_*` and `eclipse` are the
primary hand-off. Phase 2 will add truth attitude/body angular rate. Phase 3 will rotate
the Sun and magnetic vectors into the body frame and then apply sensor noise, bias,
field-of-view and failure models.

See `docs/PHASE1_RESEARCH_BASIS.md` for the rationale and limitations and `docs/VALIDATION_REPORT.md` for the packaged-run checks.
