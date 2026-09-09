"""Interactive Plotly dashboard for visual verification of Phase 1."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulation.constants import R_EARTH_KM


def _add_ground_track(fig, df: pd.DataFrame, row: int, col: int) -> None:
    lon = df.longitude_deg.to_numpy()
    lat = df.latitude_deg.to_numpy()
    jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0]
    start = 0
    for j in np.append(jumps, len(lon) - 1):
        fig.add_trace(
            go.Scatter(x=lon[start:j+1], y=lat[start:j+1], mode="lines", showlegend=False),
            row=row, col=col,
        )
        start = j + 1
    fig.add_trace(go.Scatter(x=[lon[0]], y=[lat[0]], mode="markers", name="Ground-track start"), row=row, col=col)


def build_dashboard(df: pd.DataFrame, output_html: Path) -> None:
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D ECI orbit", "Ground track",
            "Altitude", "Orbital speed",
            "Sun vector in LVLH", "Magnetic vector in LVLH",
            "Eclipse state", "Sun–magnetic-vector angle",
        ),
        vertical_spacing=0.065,
    )

    fig.add_trace(
        go.Scatter3d(x=df.r_eci_x_km, y=df.r_eci_y_km, z=df.r_eci_z_km, mode="lines", name="Orbit"),
        row=1, col=1,
    )
    th = np.linspace(0, 2*np.pi, 120)
    for z_fraction in np.linspace(-0.8, 0.8, 5):
        radius = R_EARTH_KM * np.sqrt(1-z_fraction**2)
        fig.add_trace(
            go.Scatter3d(
                x=radius*np.cos(th), y=radius*np.sin(th), z=np.full_like(th, R_EARTH_KM*z_fraction),
                mode="lines", line={"width": 1}, showlegend=False,
            ), row=1, col=1,
        )

    _add_ground_track(fig, df, row=1, col=2)
    fig.add_trace(go.Scatter(x=df.time_orbits, y=df.altitude_km, name="Altitude"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.time_orbits, y=df.speed_km_s, name="Speed"), row=2, col=2)

    for key, name in [("sun_lvlh_x", "Sun x"), ("sun_lvlh_y", "Sun y"), ("sun_lvlh_z", "Sun z")]:
        fig.add_trace(go.Scatter(x=df.time_orbits, y=df[key], name=name), row=3, col=1)
    for key, name in [("mag_lvlh_x_uT", "B x"), ("mag_lvlh_y_uT", "B y"), ("mag_lvlh_z_uT", "B z")]:
        fig.add_trace(go.Scatter(x=df.time_orbits, y=df[key], name=name), row=3, col=2)

    fig.add_trace(go.Scatter(x=df.time_orbits, y=df.eclipse, name="Eclipse", line_shape="hv"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.time_orbits, y=df.sun_mag_angle_deg, name="Sun–B angle"), row=4, col=2)

    fig.update_xaxes(title_text="Longitude [deg]", range=[-180, 180], row=1, col=2)
    fig.update_yaxes(title_text="Latitude [deg]", range=[-90, 90], row=1, col=2)
    for row, col in [(2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]:
        fig.update_xaxes(title_text="Elapsed orbital periods", row=row, col=col)
    fig.update_yaxes(title_text="Altitude [km]", row=2, col=1)
    fig.update_yaxes(title_text="Speed [km/s]", row=2, col=2)
    fig.update_yaxes(title_text="Unit vector", row=3, col=1)
    fig.update_yaxes(title_text="Magnetic field [µT]", row=3, col=2)
    fig.update_yaxes(title_text="0=sunlit, 1=eclipse", range=[-0.1, 1.1], row=4, col=1)
    fig.update_yaxes(title_text="Angle [deg]", range=[0, 180], row=4, col=2)

    fig.update_layout(
        height=1450,
        title="CubeSat FDIR Phase 1 — orbit and ADCS reference environment",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    # Embed Plotly JS so the dashboard works offline after extraction.
    fig.write_html(output_html, include_plotlyjs=True)
