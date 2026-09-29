from __future__ import annotations
import numpy as np


def box_inertia_diag(mass_kg: float, dimensions_m):
    """Principal inertia of a uniform rectangular cuboid about its centre.

    dimensions_m = [Lx, Ly, Lz] along body +X,+Y,+Z.
    """
    m = float(mass_kg)
    dims = np.asarray(dimensions_m, dtype=float)
    if m <= 0 or dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError('mass_kg and all three spacecraft dimensions must be positive')
    lx, ly, lz = dims
    return np.array([
        m * (ly*ly + lz*lz) / 12.0,
        m * (lx*lx + lz*lz) / 12.0,
        m * (lx*lx + ly*ly) / 12.0,
    ])


def inertia_from_config(spacecraft_cfg: dict):
    """Return inertia diagonal, deriving it from mass/geometry unless overridden."""
    if 'inertia_kg_m2' in spacecraft_cfg:
        arr = np.asarray(spacecraft_cfg['inertia_kg_m2'], dtype=float)
        if arr.shape != (3,) or np.any(arr <= 0):
            raise ValueError('spacecraft.inertia_kg_m2 must contain three positive values')
        return arr
    return box_inertia_diag(spacecraft_cfg['mass_kg'], spacecraft_cfg['dimensions_m'])
