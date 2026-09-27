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

## Phase 5 — fault injection and ML dataset
Run:
```powershell
python -m src.experiments.run_phase5
```
Outputs are under `data/results/phase5/`. `phase5_scenario_summary.csv` is the experiment summary; `phase5_sample_dataset.csv` contains time-labelled features; `windows_N*.npz` contains temporal ML windows for N = 1, 5, 10, 20, 50, 100.

**Important:** ML train/validation/test splitting must be performed by `run_id` (complete simulation runs), never by randomly splitting overlapping windows, to avoid leakage.

## Phase 6: Monte Carlo ML fault detection

Phase 6 generates independent randomized simulation runs and compares MLP, 1D-CNN and GRU classifiers using run-level train/validation/test splits. Overlapping windows from one simulation run are never distributed across different splits.

Install the two new ML dependencies into the existing environment once:

```powershell
conda activate cubesat-orekit
conda install -c conda-forge scikit-learn pytorch
```

Recommended first smoke test (fast):

```powershell
python -m src.experiments.run_phase6 --runs-per-class 5 --duration 30 --epochs 2 --windows 20 --models mlp,cnn,gru --force-data
```

Main experiment:

```powershell
python -m src.experiments.run_phase6 --runs-per-class 50 --epochs 20 --force-data
```

For a more robust final thesis campaign, 100 runs/class can be used if runtime permits:

```powershell
python -m src.experiments.run_phase6 --runs-per-class 100 --epochs 25 --force-data
```

Outputs are stored under `data/results/phase6/` and `plots/phase6/`. The main table is `phase6_model_comparison.csv`.
