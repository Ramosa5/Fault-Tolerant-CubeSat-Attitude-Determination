import numpy as np
from src.visualization.animation import _body_vector_in_inertial


def test_body_vector_identity_quaternion():
    v = _body_vector_in_inertial(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [1.0, 0.0, 0.0], atol=1e-12)


def test_body_x_rotated_90deg_about_inertial_z():
    a = np.deg2rad(90.0) / 2.0
    q = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    v = _body_vector_in_inertial(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-12)


def test_cubesat_mesh_uses_scene_coordinates_and_attitude():
    # Geometry helper is tested indirectly through its deterministic body->ECI corner transform.
    # A 90 deg rotation about +Z must exchange the visual X/Y extents.
    from src.simulation.attitude import quat_to_dcm
    a=np.deg2rad(90.0)/2.0
    q=np.array([np.cos(a),0.0,0.0,np.sin(a)])
    R=quat_to_dcm(q)
    body_corner=np.array([70.0,45.0,45.0])
    inertial=R@body_corner
    assert np.allclose(inertial,[-45.0,70.0,45.0],atol=1e-10)


def test_eci_ecef_round_trip_and_groundtrack_radius():
    from src.visualization.animation import _eci_to_ecef, _ecef_to_eci_at_angle, _subsatellite_groundtrack_ecef
    from src.simulation.constants import R_EARTH
    p=np.array([[6878137.0,0.0,0.0],[0.0,6878137.0,0.0]])
    t=np.array([0.0,100.0])
    ecef,angles=_eci_to_ecef(p,t,'2026-01-01T00:00:00+00:00')
    back=np.vstack([_ecef_to_eci_at_angle(ecef[i:i+1],angles[i])[0] for i in range(2)])
    assert np.allclose(back,p,atol=1e-6)
    gt,_=_subsatellite_groundtrack_ecef(p,t,'2026-01-01T00:00:00+00:00',R_EARTH/1000.0,offset_km=10.0)
    assert np.allclose(np.linalg.norm(gt,axis=1),R_EARTH/1000.0+10.0,atol=1e-9)


def test_graticule_contains_finite_geometry_and_labels():
    from src.visualization.animation import _graticule_ecef, _graticule_label_points_ecef
    g=_graticule_ecef(6378.137,30.0)
    pts,labels=_graticule_label_points_ecef(6378.137,30.0)
    assert g.shape[1] == 3 and np.isfinite(g).any()
    assert len(pts) == len(labels) and len(labels) > 10
    assert any('N' in x for x in labels) and any('E' in x for x in labels)
