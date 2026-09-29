from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from src.simulation.attitude import quat_to_dcm, quat_angle_error_deg
from src.simulation.orbit_model import propagate_circular_two_body, circular_orbit_period
from src.simulation.constants import R_EARTH
from src.simulation.magnetic_field import gmst_angle_rad, EARTH_ROTATION_RAD_S

_AXIS = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _body_vector_in_inertial(q, body_vec):
    """quat_to_dcm() maps body coordinates to inertial coordinates."""
    return quat_to_dcm(q) @ np.asarray(body_vec, dtype=float)


def _segment_trace(go, origin_km, direction, length_km, name, color, width=6, dash='solid', showlegend=True):
    d = _unit(direction) * float(length_km)
    end = origin_km + d
    return go.Scatter3d(
        x=[origin_km[0], end[0]], y=[origin_km[1], end[1]], z=[origin_km[2], end[2]],
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        showlegend=showlegend,
        hoverinfo='name',
    )


def _arrowhead_trace(go, origin_km, direction, length_km, name, color, size_km=120.0, showlegend=False):
    """Data-coordinate arrowhead, so it scales naturally with 3D zoom."""
    d = _unit(direction)
    end = np.asarray(origin_km, dtype=float) + d * float(length_km)
    return go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        anchor='tip', sizemode='absolute', sizeref=float(size_km),
        colorscale=[[0.0, color], [1.0, color]], showscale=False,
        name=name, showlegend=showlegend, hoverinfo='name',
    )


def _label_trace(go, origin_km, direction, length_km, text, color, showlegend=False):
    d = _unit(direction)
    end = np.asarray(origin_km, dtype=float) + d * float(length_km)
    return go.Scatter3d(
        x=[end[0]], y=[end[1]], z=[end[2]], mode='text', text=[text],
        textfont=dict(color=color, size=14), name=text, showlegend=showlegend, hoverinfo='skip'
    )


def _cubesat_mesh_trace(go, origin_km, q_body_to_inertial, size_km, *, name='CubeSat', color='#30343B', showlegend=True):
    """Render an attitude-oriented CubeSat as real 3D geometry.

    ``size_km`` is deliberately a visualization scale, not physical spacecraft size.
    Mesh3d vertices live in scene coordinates, so the spacecraft grows/shrinks when
    the user zooms instead of remaining a fixed-size screen marker.
    """
    dims = np.asarray(size_km, dtype=float)
    if dims.shape == ():
        dims = np.repeat(float(dims), 3)
    if dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError('satellite_visual_size_km must be a positive scalar or three positive values')
    hx, hy, hz = dims / 2.0
    verts_b = np.array([
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz],
    ], dtype=float)
    R = quat_to_dcm(np.asarray(q_body_to_inertial, dtype=float))
    verts_i = (R @ verts_b.T).T + np.asarray(origin_km, dtype=float)
    # 12 triangles, two per face
    i = [0,0,4,4,0,0,1,1,2,2,3,3]
    j = [1,2,5,6,1,5,2,6,3,7,0,4]
    k = [2,3,6,7,5,4,6,5,7,6,4,7]
    return go.Mesh3d(
        x=verts_i[:,0], y=verts_i[:,1], z=verts_i[:,2],
        i=i, j=j, k=k,
        color=color, opacity=0.92, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.25, roughness=0.8),
        lightposition=dict(x=10000, y=5000, z=10000),
        name=name, showlegend=showlegend, hovertemplate='CubeSat body (visual scale)<extra></extra>'
    )



def _parse_epoch_utc(value):
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _eci_to_ecef(points_eci, times_s, epoch_utc):
    """Rotate ECI coordinates into the Earth-fixed frame using GMST + Earth rotation."""
    pts = np.asarray(points_eci, dtype=float)
    tt = np.asarray(times_s, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if len(pts) != len(tt):
        raise ValueError('points_eci and times_s must have matching lengths')
    th0 = gmst_angle_rad(_parse_epoch_utc(epoch_utc))
    angles = th0 + EARTH_ROTATION_RAD_S * tt
    out = np.empty_like(pts)
    for i, a in enumerate(angles):
        c, sn = np.cos(a), np.sin(a)
        R = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])
        out[i] = R @ pts[i]
    return out, angles


def _ecef_to_eci_at_angle(points_ecef, angle_rad):
    """Rotate one or many ECEF points to ECI for a specified Earth rotation angle."""
    pts = np.asarray(points_ecef, dtype=float)
    c, sn = np.cos(angle_rad), np.sin(angle_rad)
    Rt = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    return (Rt @ pts.T).T


def _subsatellite_groundtrack_ecef(position_eci_m, times_s, epoch_utc, radius_km, offset_km=8.0):
    """Return sub-satellite points painted just above the Earth sphere in ECEF kilometres."""
    ecef_m, angles = _eci_to_ecef(position_eci_m, times_s, epoch_utc)
    norms = np.linalg.norm(ecef_m, axis=1, keepdims=True)
    unit = np.divide(ecef_m, norms, out=np.zeros_like(ecef_m), where=norms > 0)
    return unit * (float(radius_km) + float(offset_km)), angles


def _graticule_ecef(radius_km, step_deg=30.0, offset_km=4.0):
    """Combined latitude/longitude graticule in ECEF with None separators."""
    step = float(step_deg)
    if step <= 0 or step > 90:
        raise ValueError('graticule_step_deg must be in (0, 90]')
    r = float(radius_km) + float(offset_km)
    pieces = []
    # Parallels (omit poles because they collapse to points).
    for lat in np.arange(-90 + step, 90, step):
        lon = np.deg2rad(np.linspace(-180, 180, 145))
        la = np.deg2rad(lat)
        pieces.append(np.column_stack([r*np.cos(la)*np.cos(lon), r*np.cos(la)*np.sin(lon), np.full_like(lon, r*np.sin(la))]))
        pieces.append(np.array([[np.nan, np.nan, np.nan]]))
    # Meridians.
    for lon_deg in np.arange(-180, 180, step):
        lat = np.deg2rad(np.linspace(-90, 90, 91))
        lo = np.deg2rad(lon_deg)
        pieces.append(np.column_stack([r*np.cos(lat)*np.cos(lo), r*np.cos(lat)*np.sin(lo), r*np.sin(lat)]))
        pieces.append(np.array([[np.nan, np.nan, np.nan]]))
    return np.vstack(pieces)


def _graticule_label_points_ecef(radius_km, step_deg=30.0, offset_km=35.0):
    """Sparse ECEF label anchors and human-readable latitude/longitude labels."""
    step = float(step_deg); r = float(radius_km) + float(offset_km)
    pts=[]; labels=[]
    # Latitude labels placed near the prime meridian.
    for lat in np.arange(-90 + step, 90, step):
        la=np.deg2rad(lat); lo=0.0
        pts.append([r*np.cos(la)*np.cos(lo), r*np.cos(la)*np.sin(lo), r*np.sin(la)])
        hemi='N' if lat>0 else ('S' if lat<0 else '')
        labels.append(f'{abs(int(lat))}°{hemi}' if lat != 0 else '0° lat')
    # Longitude labels placed on equator.
    for lon in np.arange(-180, 180, step):
        lo=np.deg2rad(lon)
        pts.append([r*np.cos(lo), r*np.sin(lo), 0.0])
        if lon == 0:
            labels.append('0° lon')
        elif abs(lon) == 180:
            labels.append('180°')
        else:
            hemi='E' if lon>0 else 'W'
            labels.append(f'{abs(int(lon))}°{hemi}')
    return np.asarray(pts,float), labels


def _animation_frame_indices(times_s, frame_stride=20, max_frames=600, start_time_s=None, end_time_s=None):
    """Select animation samples without changing the underlying simulation output.

    ``frame_stride`` is a requested minimum stride. If it would produce more than
    ``max_frames``, the stride is increased automatically. Start/end times only
    limit the visualization; all numerical results remain untouched.
    """
    t=np.asarray(times_s,dtype=float)
    if t.ndim != 1 or len(t)==0:
        raise ValueError('times_s must be a non-empty one-dimensional array')
    start_none = start_time_s is None or str(start_time_s).strip().lower() in ('none','null','')
    end_none = end_time_s is None or str(end_time_s).strip().lower() in ('none','null','')
    lo=t[0] if start_none else float(start_time_s)
    hi=t[-1] if end_none else float(end_time_s)
    if hi < lo:
        raise ValueError('animation end_time_s must be >= start_time_s')
    valid=np.flatnonzero((t>=lo) & (t<=hi))
    if len(valid)==0:
        raise ValueError('animation time window does not overlap the simulation')
    requested=max(1,int(frame_stride))
    cap=max(2,int(max_frames))
    auto=max(1,int(np.ceil(len(valid)/cap)))
    stride=max(requested,auto)
    idx=valid[::stride]
    if idx[-1] != valid[-1]:
        idx=np.r_[idx,valid[-1]]
    return idx.astype(int)


def _downsample_indices(n, max_points=900):
    """Evenly retain at most ``max_points`` samples, always including endpoints."""
    n=int(n); cap=max(2,int(max_points))
    if n <= cap:
        return np.arange(n,dtype=int)
    return np.unique(np.linspace(0,n-1,cap,dtype=int))



def _save_compact_mp4(
    output_mp4, t, qt, qe, pos_km, vel, sun_arr, mag_arr, eclipse, modes,
    ground_ecef_km, earth_angles, track, *, frame_stride, max_frames,
    start_time_s, end_time_s, vector_length_km, body_pointing_axis,
    fps=20, dpi=110, title='CubeSat orbit and attitude', fault_name=None,
    fault_start_s=None, fault_end_s=None,
):
    """Save a compact Matplotlib/FFmpeg overview animation.

    This intentionally renders fewer objects than the interactive HTML. It is meant
    for long-duration playback and presentations, not detailed 3D inspection.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError as exc:
        raise RuntimeError('Matplotlib is required for MP4 animation export') from exc

    idx=_animation_frame_indices(t, frame_stride=frame_stride, max_frames=max_frames,
                                 start_time_s=start_time_s, end_time_s=end_time_s)
    baxis=_AXIS[body_pointing_axis.lower()]
    re_km=R_EARTH/1000.0
    fig=plt.figure(figsize=(9.5,8.0))
    ax=fig.add_subplot(111,projection='3d')
    # Coarse Earth surface keeps MP4 rendering fast.
    uu=np.linspace(0,2*np.pi,48); vv=np.linspace(0,np.pi,24)
    ex=re_km*np.outer(np.cos(uu),np.sin(vv)); ey=re_km*np.outer(np.sin(uu),np.sin(vv)); ez=re_km*np.outer(np.ones_like(uu),np.cos(vv))
    ax.plot_surface(ex,ey,ez,alpha=.18,linewidth=0,rstride=2,cstride=2)
    ax.plot(track[:,0],track[:,1],track[:,2],linewidth=1.0,label='Orbit')
    gt_idx=_downsample_indices(len(ground_ecef_km),600)
    gt,=ax.plot([],[],[],linestyle='--',alpha=.35,linewidth=1.2,label='Ground track')
    sat,=ax.plot([],[],[],marker='o',markersize=5,linestyle='None',label='CubeSat')
    bore,=ax.plot([],[],[],linewidth=2.5,label='True boresight')
    est,=ax.plot([],[],[],linestyle='--',linewidth=2.0,label='Estimated boresight')
    sun,=ax.plot([],[],[],linewidth=2.2,label='Sun')
    mag,=ax.plot([],[],[],linewidth=1.6,label='Magnetic field')
    txt=ax.text2D(.02,.98,'',transform=ax.transAxes,va='top',fontsize=9)
    lim=max(np.max(np.linalg.norm(track,axis=1)),re_km)+vector_length_km*1.5
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect((1,1,1)); ax.set_xlabel('ECI x [km]'); ax.set_ylabel('ECI y [km]'); ax.set_zlabel('ECI z [km]')
    ax.set_title(title+' — compact MP4'); ax.legend(loc='upper right',fontsize=8)

    def seg(origin,direction,length):
        d=_unit(direction)*float(length); e=origin+d
        return [origin[0],e[0]],[origin[1],e[1]],[origin[2],e[2]]

    def update(fi):
        k=int(idx[fi]); p=pos_km[k]; ang=earth_angles[k]
        gg=_ecef_to_eci_at_angle(ground_ecef_km[gt_idx],ang)
        gt.set_data(gg[:,0],gg[:,1]); gt.set_3d_properties(gg[:,2])
        sat.set_data([p[0]],[p[1]]); sat.set_3d_properties([p[2]])
        x,y,z=seg(p,_body_vector_in_inertial(qt[k],baxis),vector_length_km*1.15); bore.set_data(x,y); bore.set_3d_properties(z)
        if qe is not None:
            x,y,z=seg(p,_body_vector_in_inertial(qe[k],baxis),vector_length_km*1.05); est.set_data(x,y); est.set_3d_properties(z)
        x,y,z=seg(p,sun_arr[k],vector_length_km*1.35); sun.set_data(x,y); sun.set_3d_properties(z)
        x,y,z=seg(p,mag_arr[k],vector_length_km*.85); mag.set_data(x,y); mag.set_3d_properties(z)
        pe=np.degrees(np.arccos(np.clip(np.dot(_unit(_body_vector_in_inertial(qt[k],baxis)),sun_arr[k]),-1,1)))
        line=f't={t[k]:.1f} s\nSun-pointing error={pe:.3f}°\nillumination={"ECLIPSE" if eclipse[k] else "sunlight"}'
        if qe is not None: line+=f'\nMEKF error={quat_angle_error_deg(qe[k],qt[k]):.3f}°'
        if modes is not None: line+=f'\nmode={modes[k]}'
        if fault_name not in (None,'healthy') and fault_start_s is not None and fault_end_s is not None:
            line+=f'\nfault={"ACTIVE" if fault_start_s <= t[k] <= fault_end_s else "inactive"}'
        txt.set_text(line)
        return gt,sat,bore,est,sun,mag,txt

    anim=FuncAnimation(fig,update,frames=len(idx),interval=1000/max(1,int(fps)),blit=False)
    writer=FFMpegWriter(fps=max(1,int(fps)),codec='libx264',bitrate=1800,extra_args=['-pix_fmt','yuv420p'])
    output_mp4=Path(output_mp4); output_mp4.parent.mkdir(parents=True,exist_ok=True)
    anim.save(str(output_mp4),writer=writer,dpi=int(dpi))
    plt.close(fig)
    return output_mp4


def create_orbit_attitude_animation(
    output_html: str | Path,
    times_s,
    q_true,
    q_est=None,
    *,
    altitude_m=500_000.0,
    inclination_deg=51.6,
    raan_deg=0.0,
    phase_deg=0.0,
    body_pointing_axis='x',
    frame_stride=20,
    max_frames=600,
    start_time_s=None,
    end_time_s=None,
    ground_track_max_points=900,
    include_plotlyjs='cdn',
    playback_frame_ms=80,
    save_mp4=False,
    mp4_fps=20,
    mp4_max_frames=900,
    mp4_dpi=110,
    vector_length_km=900.0,
    satellite_visual_size_km=(140.0, 90.0, 90.0),
    sun_vector_length_scale=1.35,
    sun_arrowhead_size_km=140.0,
    show_true_axes=True,
    show_estimated_axes=True,
    show_sun_vector=True,
    show_mag_vector=True,
    show_velocity_vector=True,
    show_nadir_vector=True,
    show_full_orbit=True,
    show_earth_graticule=True,
    graticule_step_deg=30.0,
    show_lat_lon_labels=True,
    show_ground_track=True,
    ground_track_opacity=0.38,
    ground_track_width=5,
    ground_track_dash='dash',
    ground_track_surface_offset_km=10.0,
    earth_epoch_utc='2026-01-01T00:00:00+00:00',
    show_sun_pointing_error=False,
    sun_i=(1.0, 0.2, 0.1),
    mag_i=(0.25, -0.10, 0.96),
    fault_name=None,
    severity=None,
    fault_start_s=None,
    fault_end_s=None,
    title='CubeSat orbit and attitude validation',
    eclipse_mask=None,
    position_eci_m=None,
    velocity_eci_mps=None,
    mode_series=None,
):
    """Create an interactive Plotly HTML animation of orbit + attitude.

    The animation uses the configured analytical orbit for ECI position/velocity and
    the supplied quaternions for body orientation. The bold boresight is a body-fixed
    axis rotated to ECI using the actual quaternion state.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError('Plotly is required for HTML animations. Install with: conda install -c conda-forge plotly') from exc

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(times_s, dtype=float)
    qt = np.asarray(q_true, dtype=float)
    qe = None if q_est is None else np.asarray(q_est, dtype=float)
    if len(t) != len(qt) or (qe is not None and len(qe) != len(t)):
        raise ValueError('times_s, q_true, and q_est must have matching lengths')
    if body_pointing_axis.lower() not in _AXIS:
        raise ValueError('body_pointing_axis must be x, y, or z')

    idx = _animation_frame_indices(
        t, frame_stride=frame_stride, max_frames=max_frames,
        start_time_s=start_time_s, end_time_s=end_time_s
    )

    if position_eci_m is None or velocity_eci_mps is None:
        states = propagate_circular_two_body(t, altitude_m, inclination_deg, raan_deg, phase_deg)
        pos_km = states[:, :3] / 1000.0
        vel = states[:, 3:]
    else:
        pos_km = np.asarray(position_eci_m, dtype=float) / 1000.0
        vel = np.asarray(velocity_eci_mps, dtype=float)
        if len(pos_km) != len(t) or len(vel) != len(t):
            raise ValueError('position_eci_m and velocity_eci_mps must match times_s length')

    if show_full_orbit:
        period = circular_orbit_period(altitude_m)
        track_t = np.linspace(0.0, period, 500)
        track = propagate_circular_two_body(track_t, altitude_m, inclination_deg, raan_deg, phase_deg)[:, :3] / 1000.0
    else:
        track = pos_km

    sun_arr = np.asarray(sun_i, dtype=float)
    if sun_arr.ndim == 1:
        sun_arr = np.repeat(_unit(sun_arr)[None,:], len(t), axis=0)
    elif sun_arr.shape == (len(t),3):
        sun_arr = np.array([_unit(v) for v in sun_arr])
    else:
        raise ValueError('sun_i must be a 3-vector or Nx3 array matching times_s')
    mag_arr = np.asarray(mag_i, dtype=float)
    if mag_arr.ndim == 1:
        mag_arr = np.repeat(_unit(mag_arr)[None,:], len(t), axis=0)
    elif mag_arr.shape == (len(t),3):
        mag_arr = np.array([_unit(v) for v in mag_arr])
    else:
        raise ValueError('mag_i must be a 3-vector or Nx3 array matching times_s')
    eclipse = np.zeros(len(t), dtype=bool) if eclipse_mask is None else np.asarray(eclipse_mask, dtype=bool)
    modes = None if mode_series is None else np.asarray(mode_series, dtype=object)
    if modes is not None and len(modes) != len(t):
        raise ValueError('mode_series must match times_s length')
    if len(eclipse) != len(t):
        raise ValueError('eclipse_mask must match times_s length')
    baxis = _AXIS[body_pointing_axis.lower()]
    axis_colors = {'x': '#D62728', 'y': '#2CA02C', 'z': '#1F77B4'}

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    re_km = R_EARTH / 1000.0
    earth_x = re_km*np.outer(np.cos(u), np.sin(v))
    earth_y = re_km*np.outer(np.sin(u), np.sin(v))
    earth_z = re_km*np.outer(np.ones_like(u), np.cos(v))

    # Earth-fixed geographic overlays. Ground track is computed from the actual ECI
    # spacecraft trajectory and Earth rotation, then all Earth-fixed geometry is rotated
    # into the current ECI scene at each animation frame.
    epoch_dt = _parse_epoch_utc(earth_epoch_utc)
    ground_ecef_km, earth_angles = _subsatellite_groundtrack_ecef(
        pos_km*1000.0, t, epoch_dt, re_km, offset_km=ground_track_surface_offset_km
    )
    # The full numerical trajectory can contain tens of thousands of samples. For
    # rendering, keep a bounded representation of the same ground track. This is
    # visual downsampling only and never changes saved simulation data.
    ground_draw_idx = _downsample_indices(len(ground_ecef_km), ground_track_max_points)
    ground_draw_ecef_km = ground_ecef_km[ground_draw_idx]
    grat_ecef = _graticule_ecef(re_km, graticule_step_deg, offset_km=4.0) if show_earth_graticule else None
    label_ecef, label_text = _graticule_label_points_ecef(re_km, graticule_step_deg, offset_km=45.0) if show_lat_lon_labels else (None, None)

    def earth_overlay_traces(k, legend=False):
        traces=[]; ang=earth_angles[k]
        if show_earth_graticule:
            gi=_ecef_to_eci_at_angle(grat_ecef, ang)
            traces.append(go.Scatter3d(
                x=gi[:,0], y=gi[:,1], z=gi[:,2], mode='lines',
                line=dict(color='rgba(80,80,80,0.34)', width=2),
                name='Latitude / longitude graticule', showlegend=legend, hoverinfo='skip'
            ))
        if show_lat_lon_labels:
            li=_ecef_to_eci_at_angle(label_ecef, ang)
            traces.append(go.Scatter3d(
                x=li[:,0], y=li[:,1], z=li[:,2], mode='text', text=label_text,
                textfont=dict(color='rgba(50,50,50,0.78)', size=10),
                name='Latitude / longitude labels', showlegend=False, hoverinfo='skip'
            ))
        if show_ground_track:
            gg=_ecef_to_eci_at_angle(ground_draw_ecef_km, ang)
            traces.append(go.Scatter3d(
                x=gg[:,0], y=gg[:,1], z=gg[:,2], mode='lines',
                line=dict(color='#2F4F4F', width=int(ground_track_width), dash=str(ground_track_dash)),
                opacity=float(ground_track_opacity), name='Ground track', showlegend=legend,
                hovertemplate='Projected ground track<extra></extra>'
            ))
            # Current sub-satellite point.
            cur=_ecef_to_eci_at_angle(ground_ecef_km[k:k+1], ang)[0]
            traces.append(go.Scatter3d(
                x=[cur[0]],y=[cur[1]],z=[cur[2]],mode='markers',
                marker=dict(size=4,color='#2F4F4F'),name='Current sub-satellite point',
                showlegend=legend,hoverinfo='name'
            ))
        return traces

    def traces_at(k, legend=False):
        p = pos_km[k]
        tr = earth_overlay_traces(k, legend=legend) + [
            _cubesat_mesh_trace(
                go, p, qt[k], satellite_visual_size_km,
                name='CubeSat body (visual scale)', color='#30343B', showlegend=legend
            )
        ]
        if show_true_axes:
            for ax_name, vec in _AXIS.items():
                d = _body_vector_in_inertial(qt[k], vec)
                tr.append(_segment_trace(go, p, d, vector_length_km*0.55,
                                         f'True body +{ax_name.upper()}', axis_colors[ax_name], width=4,
                                         showlegend=legend))
        if show_estimated_axes and qe is not None:
            for ax_name, vec in _AXIS.items():
                d = _body_vector_in_inertial(qe[k], vec)
                tr.append(_segment_trace(go, p, d, vector_length_km*0.45,
                                         f'Estimated body +{ax_name.upper()}', axis_colors[ax_name], width=3,
                                         dash='dash', showlegend=legend))

        d_true = _body_vector_in_inertial(qt[k], baxis)
        tr.append(_segment_trace(go, p, d_true, vector_length_km*1.15,
                                 f'True boresight (+{body_pointing_axis.upper()})', '#FF7F0E', width=10,
                                 showlegend=legend))
        if qe is not None:
            d_est = _body_vector_in_inertial(qe[k], baxis)
            tr.append(_segment_trace(go, p, d_est, vector_length_km*1.05,
                                     f'Estimated boresight (+{body_pointing_axis.upper()})', '#9467BD', width=8,
                                     dash='dash', showlegend=legend))
        if show_sun_vector:
            sun_color = '#8C8C8C' if eclipse[k] else '#FDB813'
            sun_len = vector_length_km * float(sun_vector_length_scale)
            tr.append(_segment_trace(go, p, sun_arr[k], sun_len, 'Sun direction (ECI)', sun_color, width=8, showlegend=legend))
            tr.append(_arrowhead_trace(go, p, sun_arr[k], sun_len, 'Sun direction arrowhead', sun_color,
                                       size_km=sun_arrowhead_size_km, showlegend=False))
            tr.append(_label_trace(go, p, sun_arr[k], sun_len*1.08, 'SUN', sun_color, showlegend=False))
        if show_mag_vector:
            tr.append(_segment_trace(go, p, mag_arr[k], vector_length_km*0.85, 'Magnetic field direction (ECI)', '#17BECF', width=5, showlegend=legend))
        if show_velocity_vector:
            tr.append(_segment_trace(go, p, vel[k], vector_length_km*0.75, 'Velocity direction', '#8C564B', width=5, showlegend=legend))
        if show_nadir_vector:
            tr.append(_segment_trace(go, p, -p, vector_length_km*0.75, 'Nadir direction', '#7F7F7F', width=5, showlegend=legend))
        return tr

    dynamic0 = traces_at(int(idx[0]), legend=True)
    static = [
        go.Surface(x=earth_x, y=earth_y, z=earth_z, opacity=0.28, showscale=False,
                   colorscale=[[0, '#DDEEFF'], [1, '#6EA8D7']], name='Earth', hoverinfo='skip'),
        go.Scatter3d(x=track[:,0], y=track[:,1], z=track[:,2], mode='lines',
                     line=dict(color='#8A8A8A', width=3), name='Configured orbit', showlegend=True,
                     hoverinfo='skip'),
    ]

    frames = []
    for k in idx:
        dyn = traces_at(int(k), legend=False)
        err = quat_angle_error_deg(qe[k], qt[k]) if qe is not None else np.nan
        active_fault = (
            fault_name not in (None, 'healthy') and fault_start_s is not None and fault_end_s is not None
            and float(fault_start_s) <= t[k] <= float(fault_end_s)
        )
        status = 'ACTIVE' if active_fault else 'inactive'
        annotation = f't = {t[k]:.1f} s'
        if qe is not None:
            annotation += f'<br>attitude error = {err:.4f}°'
        if show_sun_pointing_error:
            dtrue = _unit(_body_vector_in_inertial(qt[k], baxis))
            pe = np.degrees(np.arccos(np.clip(np.dot(dtrue, sun_arr[k]), -1.0, 1.0)))
            annotation += f'<br>true Sun-pointing error = {pe:.3f}°'
            if qe is not None:
                dest = _unit(_body_vector_in_inertial(qe[k], baxis))
                pee = np.degrees(np.arccos(np.clip(np.dot(dest, sun_arr[k]), -1.0, 1.0)))
                annotation += f'<br>estimated Sun-pointing error = {pee:.3f}°'
        annotation += f'<br>illumination = {"ECLIPSE" if eclipse[k] else "sunlight"}'
        if modes is not None:
            annotation += f'<br>ADCS mode = {modes[k]}'
        if fault_name is not None:
            annotation += f'<br>fault = {fault_name}'
            if severity is not None:
                annotation += f' ({severity})'
            annotation += f'<br>fault status = {status}'
        frames.append(go.Frame(
            data=dyn,
            traces=list(range(2, 2+len(dyn))),
            name=f'{t[k]:.3f}',
            layout=go.Layout(annotations=[dict(
                text=annotation, x=0.01, y=0.99, xref='paper', yref='paper',
                showarrow=False, align='left', bgcolor='rgba(255,255,255,0.82)', bordercolor='#777777'
            )])
        ))

    lim = (R_EARTH + altitude_m + vector_length_km*1000*1.35)/1000.0
    first_err = quat_angle_error_deg(qe[idx[0]], qt[idx[0]]) if qe is not None else np.nan
    initial_annotation = f't = {t[idx[0]]:.1f} s'
    if qe is not None:
        initial_annotation += f'<br>attitude error = {first_err:.4f}°'
    if show_sun_pointing_error:
        dtrue = _unit(_body_vector_in_inertial(qt[idx[0]], baxis))
        pe = np.degrees(np.arccos(np.clip(np.dot(dtrue, sun_arr[idx[0]]), -1.0, 1.0)))
        initial_annotation += f'<br>true Sun-pointing error = {pe:.3f}°'
        if qe is not None:
            dest = _unit(_body_vector_in_inertial(qe[idx[0]], baxis))
            pee = np.degrees(np.arccos(np.clip(np.dot(dest, sun_arr[idx[0]]), -1.0, 1.0)))
            initial_annotation += f'<br>estimated Sun-pointing error = {pee:.3f}°'

    initial_annotation += f'<br>illumination = {"ECLIPSE" if eclipse[idx[0]] else "sunlight"}'
    initial_annotation += '<br>CubeSat body size = visualization scale'
    if modes is not None:
        initial_annotation += f'<br>ADCS mode = {modes[idx[0]]}'
    fig = go.Figure(data=static+dynamic0, frames=frames)
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=1100, height=800,
        scene=dict(
            xaxis_title='ECI x [km]', yaxis_title='ECI y [km]', zaxis_title='ECI z [km]',
            xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim]), zaxis=dict(range=[-lim, lim]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
        legend=dict(x=0.01, y=0.82, bgcolor='rgba(255,255,255,0.75)'),
        annotations=[dict(text=initial_annotation, x=0.01, y=0.99, xref='paper', yref='paper',
                          showarrow=False, align='left', bgcolor='rgba(255,255,255,0.82)', bordercolor='#777777')],
        updatemenus=[dict(
            type='buttons', direction='left', x=0.05, y=0.02, xanchor='left', yanchor='bottom',
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, {'frame': {'duration': int(playback_frame_ms), 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]),
                dict(label='❚❚ Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]),
            ]
        )],
        sliders=[dict(
            active=0, x=0.12, y=0.0, len=0.82,
            currentvalue=dict(prefix='Time: ', suffix=' s'),
            steps=[dict(method='animate', args=[[f.name], {'mode':'immediate','frame':{'duration':0,'redraw':True},'transition':{'duration':0}}], label=f'{float(f.name):.0f}') for f in frames]
        )],
        margin=dict(l=0, r=0, t=55, b=35),
    )
    fig.write_html(output_html, include_plotlyjs=include_plotlyjs, full_html=True, auto_play=False)
    if bool(save_mp4):
        _save_compact_mp4(
            output_html.with_suffix('.mp4'), t, qt, qe, pos_km, vel, sun_arr, mag_arr, eclipse, modes,
            ground_ecef_km, earth_angles, track, frame_stride=frame_stride, max_frames=mp4_max_frames,
            start_time_s=start_time_s, end_time_s=end_time_s, vector_length_km=vector_length_km,
            body_pointing_axis=body_pointing_axis, fps=mp4_fps, dpi=mp4_dpi, title=title,
            fault_name=fault_name, fault_start_s=fault_start_s, fault_end_s=fault_end_s
        )
    return output_html
