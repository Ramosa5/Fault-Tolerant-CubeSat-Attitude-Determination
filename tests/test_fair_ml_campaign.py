from pathlib import Path
import tomllib
import numpy as np

from src.ml.fair_orbital_models import generate_shared_dataset, prepare_variant_target
from src.scenarios.reaction_wheel_fair_ml_campaign import _generated_cases, _branch_specs
from src.scenarios.registry import SCENARIOS


def _tiny_dataset_cfg():
    return {
        'runs':6,'duration_s':2.0,'dt_s':0.2,'window':3,'seed':1234,'split_seed':4321,
        'train_fraction':0.5,'validation_fraction':0.25,'altitude_m_range':[450000.,460000.],
        'inclination_deg_range':[45.,55.],'epoch_span_days':2.,'training_magnetic_field_model':'dipole',
        'masking':{'probability':.3,'max_outage_samples':4},'degradations':{'enable_nonmissing_faults':False},
    }


def test_shared_dataset_split_is_by_complete_runs_and_disjoint():
    d=generate_shared_dataset(_tiny_dataset_cfg())
    m=d['manifest']; tr=set(m['train_run_ids']); va=set(m['validation_run_ids']); te=set(m['test_run_ids'])
    assert tr and va and te
    assert not (tr&va or tr&te or va&te)
    for rid in np.unique(d['RUN']):
        memberships=sum(rid in x for x in (tr,va,te))
        assert memberships==1


def test_variant_feature_dimensions_and_common_sample_count():
    d=generate_shared_dataset(_tiny_dataset_cfg())
    counts=[]
    for variant,nf in [('sensor_only',12),('orbital_context',25),('physics_residual',25)]:
        X,Y=prepare_variant_target(d,variant,'gyro'); counts.append(len(X)); assert X.shape[-1]==nf; assert Y.shape[-1]==3
    assert len(set(counts))==1


def test_generated_fault_campaign_matches_A_B_C_design():
    cfg={'fault_scenarios':{'enabled':True,'groups':['A','B','C'],'start_s':10.,'end_s':20.,
        'single_sensor_loss':['sun','gyro','mag'],'dropout_sensors':['gyro','sun','mag'],'dropout_probabilities':[.1,.25,.5,.75],
        'gyro_bias_levels_rad_s':[.001,.003,.006,.012],'gyro_drift_levels_rad_s2':[1e-6,3e-6,8e-6],'mag_misalignment_levels_deg':[5.,10.,20.,30.]}}
    cases=_generated_cases(cfg)
    assert len(cases)==26
    assert any(c['name']=='A_sun_loss' for c in cases)
    assert any(c['name']=='B_gyro_dropout_75pct' for c in cases)
    assert any(c['name']=='C_mag_misalignment_30deg' for c in cases)


def test_default_branch_matrix_is_three_by_three_plus_baselines():
    cfg={'models':{'architectures':['mlp','cnn','gru'],'variants':['sensor_only','orbital_context','physics_residual']}}
    specs=_branch_specs(cfg); assert len(specs)==12
    names={x[0] for x in specs}
    assert {'healthy_mekf','faulty_mekf','fdir_only','mlp_sensor_only','cnn_orbital_context','gru_physics_residual'} <= names


def test_config_disables_animations_and_registers_fair_campaign():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    sc=cfg['scenarios']['reaction_wheel_fair_ml_campaign']
    assert cfg['scenario']['active']=='reaction_wheel_fair_ml_campaign'
    assert sc['animation']['enabled'] is False
    assert 'reaction_wheel_fair_ml_campaign' in SCENARIOS


def test_generated_group_d_contains_multi_sensor_cases_only():
    cfg={'fault_scenarios':{'enabled':True,'groups':['D'],'start_s':10.,'end_s':20.,
        'combined_dropout_probability':.5,'combined_gyro_bias_rad_s':.006,
        'combined_gyro_drift_rad_s2':3e-6,'combined_mag_misalignment_deg':20.}}
    cases=_generated_cases(cfg)
    assert len(cases)==10
    assert all(c['group']=='D_multi_sensor_faults' for c in cases)
    assert any(c['name']=='D1_gyro_loss_plus_sun_loss' for c in cases)
    assert any(c['name']=='D4_gyro_bias_plus_mag_misalignment' for c in cases)
    assert any(c['name']=='D9_all_three_degraded' for c in cases)
    assert any(c['name']=='D10_all_three_loss' for c in cases)
    for c in cases:
        active=sum(str((c.get(k,{}) or {}).get('type','none')).lower()!='none' for k in ('sun','gyro','mag'))
        assert active >= 2


def test_config_defaults_to_group_d_only_for_next_campaign():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    fc=cfg['scenarios']['reaction_wheel_fair_ml_campaign']['fault_scenarios']
    assert fc['groups']==['D']
    assert len(fc['combined_cases'])==10
