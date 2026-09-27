# CubeSat Phase 1 + Orekit Verification

Reconstructed from the previously created Phase-1 specification and extended with an Orekit reference propagator.
Both propagators intentionally use the same ideal two-body Keplerian model and constants; this verifies implementation consistency, not high-fidelity orbital accuracy.

## Recommended installation (Windows/macOS/Linux)
Install Miniconda/Anaconda, then from this folder:

    conda env create -f environment.yml
    conda activate cubesat-orekit

## Run
Baseline:

    python main.py

Unit tests (5 automated tests):

    pytest -v

Orekit verification campaign (6 verification cases):

    python -m src.verification.run_verification

Outputs are written to `data/results/verification_summary.csv` plus one time-series CSV per case.

## Verification cases
V1 nominal 500 km / 51.6 deg / 1 orbit; V2 low inclination; V3 high inclination; V4 400 km; V5 700 km; V6 10-orbit duration.

## Thesis interpretation
Report position RMSE/max and velocity RMSE/max. Do not claim agreement with real flight data: both models are ideal Keplerian two-body propagators. The experiment validates the custom implementation against Orekit under equivalent assumptions.
