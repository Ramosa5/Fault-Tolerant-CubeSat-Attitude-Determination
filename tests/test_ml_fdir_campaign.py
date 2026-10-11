from pathlib import Path
import tomllib
import numpy as np
import pandas as pd

from src.ml.fdir_classifier import (
    choose_multilabel_thresholds, apply_probability_filter,
    choose_runtime_multilabel_thresholds, evaluate_runtime_multilabel_thresholds,
)
from src.scenarios.reaction_wheel_ml_fdir_campaign import (
    _evaluation_cases, _sample_training_case, _stratified_run_split,
    _select_validation_heterogeneous_hybrid, _simulate_sequence, _base_case,
    FEATURE_NAMES_BASE, CONTEXT_NAMES,
)
from src.scenarios.registry import SCENARIOS


def test_ml_fdir_scenario_registered_and_active():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / 'experiment_config.toml').read_text())
    assert cfg['scenario']['active'] in {'reaction_wheel_ml_fdir_campaign','thesis_results_campaign'}
    assert 'reaction_wheel_ml_fdir_campaign' in SCENARIOS
    sc = cfg['scenarios']['reaction_wheel_ml_fdir_campaign']
    assert sc['models']['architectures'] == ['mlp', 'cnn', 'gru']
    assert sc['keep_uncompressed_timeseries'] is False


def test_ml_fdir_feature_definition_is_fixed_and_context_aware():
    assert len(FEATURE_NAMES_BASE) == 31
    assert len(CONTEXT_NAMES) == 17
    assert len(FEATURE_NAMES_BASE + CONTEXT_NAMES) == 48
    assert len(set(FEATURE_NAMES_BASE + CONTEXT_NAMES)) == 48


def test_thresholds_are_selected_independently_per_sensor():
    y = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,1],[0,0,1]], int)
    p = np.array([[.1,.1,.1],[.9,.2,.2],[.8,.8,.1],[.2,.9,.9],[.1,.2,.8]], float)
    th = choose_multilabel_thresholds(y, p)
    assert th.shape == (3,)
    assert np.all((th >= .1) & (th <= .9))


def test_probability_filter_requires_persistence_and_resets():
    p = np.zeros((12,3), float)
    p[2:7,0] = .95
    smooth, det = apply_probability_filter(p, np.array([.5,.5,.5]), alpha=1.0, persistence_samples=3, reset_samples=2)
    assert not det[2,0]
    assert not det[3,0]
    assert det[4,0]
    assert det[6,0]
    assert det[8,0] is np.False_ or not bool(det[8,0])
    assert np.allclose(smooth[:,1:], 0.0)


def test_training_case_sampler_covers_multisensor_soft_faults():
    rng = np.random.default_rng(123)
    c = _sample_training_case('combined_soft', rng, 10.0, 20.0)
    active = sum((c[k].get('type','none') != 'none') for k in ('gyro','sun','mag'))
    assert active >= 2


def test_evaluation_contains_soft_hard_and_multisensor_cases():
    cases = _evaluation_cases({})
    names = {c['name'] for c in cases}
    assert 'E00_healthy_control' in names
    assert 'E_gyro_bias_0.006' in names
    assert 'E_gyro_drift_3e-06' in names
    assert 'E_mag_misalignment_20deg' in names
    assert 'E_gyro_loss' in names
    assert 'E_D9_all_three_degraded' in names
    assert len(cases) >= 20


def test_stratified_split_keeps_complete_runs_disjoint():
    rec=[]
    rid=0
    for fam in ['healthy','gyro_bias','gyro_drift']:
        for _ in range(4):
            rec.append({'run_id':rid,'family':fam}); rid+=1
    s=_stratified_run_split(rec,123)
    tr=set(s['train']); va=set(s['validation']); te=set(s['test'])
    assert tr and va and te
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(range(12))



def test_probability_filter_resets_before_arming_and_requires_new_persistence():
    p=np.zeros((10,3),float)
    p[:,0]=.99
    active=np.zeros((10,3),bool)
    active[5:,0]=True
    _,det=apply_probability_filter(
        p,np.array([.5,.5,.5]),alpha=1.0,persistence_samples=3,reset_samples=2,
        active_mask=active,
    )
    assert not det[:7,0].any()
    assert det[7,0]


def test_runtime_threshold_calibration_respects_far_limit():
    # Two complete validation sequences. Fault probability is high only during
    # positive intervals; a lower threshold would also trigger healthy samples.
    records=[]
    for _ in range(2):
        p=np.zeros((30,3),float)
        y=np.zeros((30,3),bool)
        mask=np.ones((30,3),bool)
        classical=np.zeros((30,3),bool)
        y[15:,0]=True
        p[:15,0]=.30
        p[15:,0]=.85
        records.append({'probabilities':p,'truth':y,'eval_mask':mask,'classical':classical})
    th,cal=choose_runtime_multilabel_thresholds(
        records,alpha=1.0,persistence_samples=1,reset_samples=1,
        max_false_alarm_rate_pct=1.0,
        grid=np.array([.2,.4,.6,.8]),
    )
    gyro=cal[cal.sensor=='gyro'].iloc[0]
    assert th[0] >= .4
    assert gyro.false_alarm_rate_pct <= 1.0 + 1e-9
    assert gyro.recall_pct > 90.0


def test_hybrid_calibration_preserves_classical_and_limits_incremental_far():
    p=np.zeros((40,3),float); y=np.zeros((40,3),bool); mask=np.ones((40,3),bool)
    classical=np.zeros((40,3),bool)
    y[20:,0]=True
    classical[24:,0]=True  # classical detects late but with no false alarms
    p[:20,0]=.2; p[20:,0]=.9
    records=[{'probabilities':p,'truth':y,'eval_mask':mask,'classical':classical}]
    th,cal=choose_runtime_multilabel_thresholds(
        records,alpha=1.0,persistence_samples=1,reset_samples=1,
        hybrid=True,max_incremental_far_pct=0.0,
        grid=np.array([.1,.3,.5,.7,.9]),
    )
    scored=evaluate_runtime_multilabel_thresholds(
        records,th,alpha=1.0,persistence_samples=1,reset_samples=1,hybrid=True,
    )
    gyro=scored[scored.sensor=='gyro'].iloc[0]
    assert gyro.false_alarm_rate_pct == 0.0
    assert gyro.recall_pct >= 80.0  # never worse than the classical 16/20 = 80%


def test_experiment2_v2_config_uses_more_runs_matched_repeats_and_runtime_thresholds():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    sc=cfg['scenarios']['reaction_wheel_ml_fdir_campaign']
    assert sc['dataset']['runs_per_family'] >= 12
    assert sc['dataset']['validation_fraction'] >= .15
    assert sc['dataset']['test_fraction'] >= .15
    assert sc['dataset']['window_stride_samples'] >= 5
    assert sc['evaluation']['repeats'] >= 3
    assert sc['models']['threshold_max_false_alarm_rate_pct'] <= 1.0
    assert sc['models']['hybrid_max_incremental_far_pct'] <= 1.0


def test_severity_sweeps_have_common_match_groups():
    cases=_evaluation_cases({})
    bias=[c for c in cases if c['family']=='gyro_bias']
    drift=[c for c in cases if c['family']=='gyro_drift']
    mis=[c for c in cases if c['family']=='mag_misalignment']
    assert len({c['match_group'] for c in bias})==1
    assert len({c['match_group'] for c in drift})==1
    assert len({c['match_group'] for c in mis})==1


def test_heterogeneous_hybrid_selection_requires_run_level_stability():
    rows=[]
    for arch,vals in {
        'mlp': {'gyro':(42.,40.), 'sun':(31.0,31.0), 'mag':(35.,34.)},
        'cnn': {'gyro':(45.,40.), 'sun':(31.1,31.0), 'mag':(44.,34.)},
        'gru': {'gyro':(43.,40.), 'sun':(31.2,31.0), 'mag':(41.,34.)},
    }.items():
        for sensor,(f1,base) in vals.items():
            rows.append({
                'architecture':arch,'sensor':sensor,'mode':'hybrid_augmentation',
                'threshold':.7,'precision_pct':f1,'recall_pct':f1,
                'f1_pct':f1,'false_alarm_rate_pct':.8,'far_limit_pct':1.0,
                'classical_baseline_precision_pct':base,
                'classical_baseline_recall_pct':base,
                'classical_baseline_f1_pct':base,
                'classical_baseline_far_pct':.5,'feasible':True,
            })
    cal=pd.DataFrame(rows)

    stability=[]
    for arch in ('mlp','cnn','gru'):
        for sensor in ('gyro','sun','mag'):
            # Three positive-fault validation runs plus one healthy/negative run.
            gains = {
                ('cnn','gyro'):[5.,4.,3.],
                ('cnn','mag'):[6.,5.,4.],
                ('gru','gyro'):[3.,2.,2.],
                ('gru','mag'):[4.,3.,2.],
                ('mlp','gyro'):[2.,1.,1.],
                ('mlp','mag'):[1.,1.,0.5],
            }.get((arch,sensor), [0.1,0.0,-0.1])
            for rid,gain in enumerate(gains):
                stability.append({
                    'architecture':arch,'sensor':sensor,'run_id':rid,
                    'fault_family':'soft','threshold':.7,'has_positive_fault':True,
                    'positive_samples':100,'negative_samples':100,
                    'classical_precision_pct':50.,'classical_recall_pct':40.,
                    'classical_f1_pct':40.,'classical_far_pct':.5,
                    'hybrid_precision_pct':55.,'hybrid_recall_pct':45.,
                    'hybrid_f1_pct':40.+gain,'hybrid_far_pct':.8,
                    'f1_gain_pct':gain,'far_increment_pct':.3,
                })
            stability.append({
                'architecture':arch,'sensor':sensor,'run_id':99,
                'fault_family':'healthy','threshold':.7,'has_positive_fault':False,
                'positive_samples':0,'negative_samples':200,
                'classical_precision_pct':np.nan,'classical_recall_pct':np.nan,
                'classical_f1_pct':np.nan,'classical_far_pct':0.,
                'hybrid_precision_pct':np.nan,'hybrid_recall_pct':np.nan,
                'hybrid_f1_pct':np.nan,'hybrid_far_pct':.4,
                'f1_gain_pct':np.nan,'far_increment_pct':.4,
            })

    sel=_select_validation_heterogeneous_hybrid(
        cal,pd.DataFrame(stability),min_f1_gain_pct=.25,
        min_positive_run_improvement_fraction=.60,
        min_positive_run_nondegradation_fraction=.75,
        f1_degradation_tolerance_pct=.25,
        max_per_run_incremental_far_pct=2.0,
        max_p95_incremental_far_pct=1.0,
    )
    mapping=dict(zip(sel.sensor,sel.selected_architecture))
    assert mapping['gyro']=='cnn'
    assert mapping['mag']=='cnn'
    assert mapping['sun']=='classical_only'


def test_heterogeneous_selector_rejects_aggregate_winner_with_one_far_spike():
    cal=pd.DataFrame([{
        'architecture':'cnn','sensor':'gyro','mode':'hybrid_augmentation',
        'threshold':.7,'precision_pct':80.,'recall_pct':50.,'f1_pct':60.,
        'false_alarm_rate_pct':.9,'far_limit_pct':1.0,
        'classical_baseline_precision_pct':75.,
        'classical_baseline_recall_pct':40.,
        'classical_baseline_f1_pct':50.,
        'classical_baseline_far_pct':.5,'feasible':True,
    }])
    st=[]
    for rid,far_inc in enumerate([.1,.2,15.0]):
        st.append({
            'architecture':'cnn','sensor':'gyro','run_id':rid,'fault_family':'gyro_bias',
            'threshold':.7,'has_positive_fault':True,'positive_samples':100,'negative_samples':100,
            'classical_precision_pct':75.,'classical_recall_pct':40.,
            'classical_f1_pct':50.,'classical_far_pct':.5,
            'hybrid_precision_pct':80.,'hybrid_recall_pct':50.,
            'hybrid_f1_pct':60.,'hybrid_far_pct':.5+far_inc,
            'f1_gain_pct':10.,'far_increment_pct':far_inc,
        })
    # Provide harmless rows for the other sensors so the helper can complete.
    for sensor in ('sun','mag'):
        cal.loc[len(cal)] = {
            'architecture':'cnn','sensor':sensor,'mode':'hybrid_augmentation',
            'threshold':.7,'precision_pct':50.,'recall_pct':50.,'f1_pct':50.,
            'false_alarm_rate_pct':.5,'far_limit_pct':1.0,
            'classical_baseline_precision_pct':50.,
            'classical_baseline_recall_pct':50.,
            'classical_baseline_f1_pct':50.,
            'classical_baseline_far_pct':.5,'feasible':True,
        }
        st.append({
            'architecture':'cnn','sensor':sensor,'run_id':10,'fault_family':'soft',
            'threshold':.7,'has_positive_fault':True,'positive_samples':100,'negative_samples':100,
            'classical_precision_pct':50.,'classical_recall_pct':50.,
            'classical_f1_pct':50.,'classical_far_pct':.5,
            'hybrid_precision_pct':50.,'hybrid_recall_pct':50.,
            'hybrid_f1_pct':50.,'hybrid_far_pct':.5,
            'f1_gain_pct':0.,'far_increment_pct':0.,
        })
    sel=_select_validation_heterogeneous_hybrid(
        cal,pd.DataFrame(st),min_f1_gain_pct=.25,
        max_per_run_incremental_far_pct=2.0,max_p95_incremental_far_pct=1.0,
    )
    gyro=sel[sel.sensor=='gyro'].iloc[0]
    assert gyro.selected_architecture=='classical_only'
    assert gyro.selection_reason=='classical_retained_run_level_stability_failed'

def test_experiment2_final_config_enables_validation_selected_hybrid():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    m=cfg['scenarios']['reaction_wheel_ml_fdir_campaign']['models']
    assert m['heterogeneous_hybrid_enabled'] is True
    assert m['heterogeneous_min_validation_f1_gain_pct'] > 0.0
    assert m['heterogeneous_min_positive_run_improvement_fraction'] >= .5
    assert m['heterogeneous_max_per_run_incremental_far_pct'] <= 2.0
    ev=cfg['scenarios']['reaction_wheel_ml_fdir_campaign']['evaluation']
    assert ev['seed'] == 17011
    assert ev['repeats'] >= 6


def test_mag_loss_does_not_poison_bdot_controller_with_nan():
    root=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((root/'experiment_config.toml').read_text())
    sc=dict(cfg['scenarios']['reaction_wheel_ml_fdir_campaign'])
    base=dict(cfg['scenarios']['reaction_wheel_sun_pointing_comparison'])
    base['magnetic_field']=dict(base.get('magnetic_field',{}))
    base['magnetic_field']['model']='dipole'
    fdir=dict(cfg['scenarios']['reaction_wheel_fair_ml_campaign']['fdir'])
    fdir.update(dict(sc.get('classical_fdir',{})))
    fdir['arming_time_s']=1.0

    case=_base_case('test_mag_loss','hard_loss',1.5,2.5)
    case['mag']={'type':'loss'}
    seq=_simulate_sequence(
        case,base_template=base,global_cfg=cfg,duration_s=4.0,dt_s=0.1,
        seed=13001,fdir_cfg=fdir,mission_cfg=sc['mission_orbit'],
        initial_error_max_deg=10.0,body_rate_max_rad_s=0.02,
        orbit_dispersion=sc['dataset']['orbit_dispersion'],
    )
    assert np.all(np.isfinite(seq['pointing_error_deg']))
    assert np.all(np.isfinite(seq['attitude_error_deg']))
    post=seq['time']>2.5
    assert not seq['hard_unavailable'][post,2].any()
