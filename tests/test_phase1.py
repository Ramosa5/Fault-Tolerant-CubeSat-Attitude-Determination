from pathlib import Path
import numpy as np

from src.simulation.constants import R_EARTH_KM
from src.simulation.environment import sun_vector_eci, magnetic_field_ecef_tesla
from src.simulation.frames import lvlh_dcm_eci_to_lvlh
from src.simulation.orbit_model import OrbitalElements, elements_to_state, kepler_period_s, propagate_orbit
from src.simulation.pipeline import build_dataframe, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_500km_period_is_reasonable():
    period_min = kepler_period_s(R_EARTH_KM + 500.0) / 60.0
    assert 94.0 < period_min < 96.0


def test_lvlh_dcm_is_orthonormal():
    elements = OrbitalElements(R_EARTH_KM + 500, 0.001, np.deg2rad(97.4), 0.2, 0.1, 0.3)
    r, v = elements_to_state(elements)
    c = lvlh_dcm_eci_to_lvlh(r, v)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(c), 1.0, atol=1e-12)


def test_sun_vector_is_unit_length():
    s, distance = sun_vector_eci(2461041.5)
    assert np.isclose(np.linalg.norm(s), 1.0, atol=1e-12)
    assert 1.45e8 < distance < 1.53e8


def test_dipole_field_is_plausible_in_leo():
    b = magnetic_field_ecef_tesla(np.array([R_EARTH_KM + 500.0, 0.0, 0.0]), 31.2, 80.65, -72.68)
    b_uT = np.linalg.norm(b) * 1e6
    assert 15.0 < b_uT < 65.0


def test_two_body_energy_is_conserved():
    elements = OrbitalElements(R_EARTH_KM + 500, 0.001, np.deg2rad(45), 0.0, 0.0, 0.0)
    r0, v0 = elements_to_state(elements)
    period = kepler_period_s(elements.semi_major_axis_km)
    t, r, v = propagate_orbit(r0, v0, period, 20.0, use_j2=False)
    energy = 0.5*np.sum(v*v, axis=1) - 398600.4418/np.linalg.norm(r, axis=1)
    relative_span = (energy.max()-energy.min())/abs(energy.mean())
    assert relative_span < 1e-8


def test_pipeline_outputs_finite_reference_vectors_and_eclipse():
    config = load_config(ROOT / "configs/phase1.yaml")
    config["simulation"]["duration_orbits"] = 1.0
    config["simulation"]["sample_time_s"] = 20.0
    df, summary = build_dataframe(config)
    required = ["sun_lvlh_x","sun_lvlh_y","sun_lvlh_z","mag_magnitude_uT","sun_mag_angle_deg","eclipse"]
    assert np.isfinite(df[required].to_numpy()).all()
    assert 10.0 < df.mag_magnitude_uT.min() < 80.0
    assert set(df.eclipse.unique()).issubset({0,1})
    assert 0.0 < summary["eclipse_fraction"] < 1.0


def test_j2_axisymmetric_invariants_are_numerically_conserved():
    from src.simulation.orbit_model import specific_total_energy, angular_momentum_z
    elements = OrbitalElements(R_EARTH_KM + 500, 0.0, np.deg2rad(97.4), np.deg2rad(90), 0.0, 0.0)
    r0, v0 = elements_to_state(elements)
    period = kepler_period_s(elements.semi_major_axis_km)
    _t, r, v = propagate_orbit(r0, v0, 2*period, 20.0, use_j2=True, rtol=1e-11, atol=1e-13)
    e = specific_total_energy(r, v, use_j2=True)
    hz = angular_momentum_z(r, v)
    assert (e.max()-e.min())/abs(e.mean()) < 1e-8
    assert (hz.max()-hz.min())/abs(hz.mean()) < 1e-8
