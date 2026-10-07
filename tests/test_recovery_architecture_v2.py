from pathlib import Path
import tomllib
from src.scenarios.reaction_wheel_ml_model_campaign import BRANCHES


def test_recovery_v2_has_fdir_only_baseline():
    assert BRANCHES == ('healthy_mekf','faulty_mekf','fdir_only','mlp_recovery','cnn_recovery','gru_recovery')


def test_recovery_v2_config_contains_severity_sweeps():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    cases=cfg['scenarios']['reaction_wheel_ml_model_campaign']['cases']
    names={c['name'] for c in cases}
    assert {'G1_gyro_bias_0p001','G4_gyro_bias_0p012','M1_mag_misalignment_5deg','M4_mag_misalignment_30deg','C1_moderate_gyro_bias_mag_misalignment'} <= names


def test_gyro_cross_sensor_gate_is_configured():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    gate=cfg['scenarios']['reaction_wheel_sun_pointing_comparison']['ml_recovery_gate']
    assert gate['gyro_cross_sensor_max_disagreement_deg'] > 0
    assert gate['gyro_cross_sensor_min_references'] >= 1
