from pathlib import Path
import copy
import tomllib

from src.scenarios.registry import SCENARIOS
from src.scenarios.thesis_results_campaign import run


def test_thesis_results_scenario_registered_and_configured():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / 'experiment_config.toml').read_text())
    assert 'thesis_results_campaign' in SCENARIOS
    assert cfg['scenario']['active'] == 'thesis_results_campaign'
    sc = cfg['scenarios']['thesis_results_campaign']
    assert sc['run_orbit'] is True
    assert sc['run_nominal_estimation'] is True
    assert sc['run_aocs'] is True
    assert sc['run_experiment1'] is True
    assert sc['run_experiment2'] is True
    assert sc['save_component_timeseries'] is False


def test_thesis_results_smoke_generates_curated_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    global_cfg = tomllib.loads((root / 'experiment_config.toml').read_text())
    sc = copy.deepcopy(global_cfg['scenarios']['thesis_results_campaign'])
    sc.update({
        'run_orbit': False,
        'run_nominal_estimation': True,
        'run_aocs': True,
        'run_experiment1': False,
        'run_experiment2': False,
        'nominal_duration_s': 4.0,
        'nominal_dt_s': 0.1,
        'aocs_duration_s': 6.0,
        'aocs_dt_s': 0.1,
        'aocs_magnetic_field_model': 'dipole',
        'save_nominal_timeseries': False,
    })
    # Make publication output light for the test while exercising the same code path.
    global_cfg = copy.deepcopy(global_cfg)
    global_cfg['visualization']['formats'] = ['png']
    global_cfg['visualization']['dpi'] = 80
    out = tmp_path / 'thesis'
    result = run(sc, global_cfg, out)
    assert result['scenario'] == 'thesis_results_campaign'
    assert (out / 'tables' / 'thesis_master_metrics.csv').exists()
    assert (out / 'tables' / 'thesis_figure_manifest.csv').exists()
    assert (out / 'THESIS_RESULTS_README.md').exists()
    assert (out / 'figures' / '02_attitude_dynamics_verification.png').exists()
    assert (out / 'figures' / '03_sensor_model_verification.png').exists()
    assert (out / 'figures' / '04_mekf_nominal_performance.png').exists()
    assert (out / 'figures' / '05_reaction_wheel_aocs_closed_loop.png').exists()
