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
    for variant,nf in [('sensor_only',12),('orbital_context',29),('physics_residual',29)]:
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
    assert cfg['scenario']['active'] in {'reaction_wheel_ml_fdir_campaign','thesis_results_campaign'}
    assert sc['animation']['enabled'] is False
    assert sc['fdir']['mode']=='classical_v31'
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


def test_mission_orbit_context_uses_imperfect_known_orbit():
    c=_tiny_dataset_cfg()
    c['mission_orbit']={
        'enabled':True,'nominal_altitude_m':500000.,'nominal_inclination_deg':51.6,
        'nominal_raan_deg':0.,'randomize_phase_each_run':True,
        'actual_inclination_injection_half_range_deg':0.1,
        'knowledge_inclination_uncertainty_deg':0.5,
        'knowledge_altitude_uncertainty_m':1000.,
        'knowledge_raan_uncertainty_deg':0.25,'knowledge_phase_uncertainty_deg':1.0,
    }
    d=generate_shared_dataset(c)
    assert d['X25'].shape[-1]==29
    assert d['manifest']['mission_orbit_context_enabled'] is True
    errs=[abs(x['inclination_knowledge_error_deg']) for x in d['manifest']['run_metadata']]
    assert max(errs)<=0.5000001
    assert any(x>1e-6 for x in errs)


def test_runtime_mission_context_accepts_string_epoch():
    import numpy as np
    from src.scenarios.reaction_wheel_fair_ml_campaign import _runtime_mission_context
    from src.simulation.orbit_model import propagate_circular_two_body
    from src.simulation.illumination import sun_direction_series, eclipse_geometry
    from src.simulation.magnetic_field import magnetic_field_eci
    from datetime import datetime, timezone

    t=np.arange(0.0,2.0,0.2)
    actual=propagate_circular_two_body(t,500000.0,51.6,0.0,0.0)
    sun,sun_pos=sun_direction_series(t,'analytic',epoch='2026-01-01T00:00:00+00:00')
    _,illum=eclipse_geometry(actual[:,:3],sun_pos)
    B=magnetic_field_eci(actual[:,:3],t,epoch_utc=datetime(2026,1,1,tzinfo=timezone.utc),model='dipole')

    base={
        'pointing_axis':'x',
        'illumination':{'epoch_utc':'2026-01-01T00:00:00+00:00'},
        'magnetic_field':{'model':'dipole'},
    }
    global_cfg={'orbit':{'altitude_m':500000.0,'inclination_deg':51.6,'raan_deg':0.0,'phase_deg':0.0}}
    dcfg={'mission_orbit':{
        'enabled':True,
        'runtime_knowledge_inclination_error_deg':0.25,
        'runtime_knowledge_altitude_error_m':0.0,
        'runtime_knowledge_raan_error_deg':0.0,
        'runtime_knowledge_phase_error_deg':0.0,
    }}
    ctx,baseline,meta=_runtime_mission_context(t,base,global_cfg,dcfg,sun,sun_pos,actual,B,illum)
    assert ctx.shape==(len(t),17)
    assert baseline.shape==(len(t),9)
    assert meta['enabled'] is True
