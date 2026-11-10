"""
lidar3d.py
----------
Multi-channel LiDAR model shared by the fast simulator and the SITL adapter.

Geometry matches the Gazebo gpu_lidar on models/x500_lidar3d: 90 azimuth samples
from -178 to +178 deg (4 deg step) x 7 elevation channels from -30 to +30 deg
(10 deg step), 0.1-10 m, on a 0.40 m mast above the body (clears arms and props). Rays rotate with the full
vehicle attitude, as on a real drone.

Features (policy input), per 45 deg azimuth sector (0 = front, CCW, as in the 26-D
interface) and elevation band:
  level band  -10, 0, +10 deg  -> 8 values (obs[18:26], shield-compatible)
  down band   -30, -20 deg     -> 8 values
  up band     +20, +30 deg     -> 8 values
Each value is the minimum range in that sector/band (max_range if nothing is seen).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Lidar3DSpec:
    n_azimuth: int = 90
    az_min_deg: float = -178.0
    az_max_deg: float = 178.0
    elevations_deg: Tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
    min_range: float = 0.10
    max_range: float = 10.0
    # Sensor origin relative to the vehicle reference point (ground level when landed, like the
    # Gazebo model frame): x500 base_link is 0.24 m up, the LiDAR mast another 0.40 m.
    mount_offset_body: Tuple[float, float, float] = (0.0, 0.0, 0.64)

    @property
    def n_rays(self) -> int:
        return self.n_azimuth * len(self.elevations_deg)

    def azimuths_rad(self) -> np.ndarray:
        return np.radians(np.linspace(self.az_min_deg, self.az_max_deg, self.n_azimuth))

    def dirs_body(self) -> np.ndarray:
        """Unit ray directions in the body FLU frame, shape (n_elev * n_az, 3), row-major by elevation."""
        az = self.azimuths_rad()
        el = np.radians(np.array(self.elevations_deg))
        ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
        d = np.stack([ce * np.cos(az)[None, :], ce * np.sin(az)[None, :], np.broadcast_to(se, (len(el), len(az)))],
                     axis=2)
        return d.reshape(-1, 3)


DEFAULT_SPEC = Lidar3DSpec()
_SECTOR_OF_AZ = None
_BAND_ROWS = {"down": [0, 1], "level": [2, 3, 4], "up": [5, 6]}


def sector_index(spec: Lidar3DSpec = DEFAULT_SPEC) -> np.ndarray:
    """Sector (0..7) of each azimuth column: 0 front [-22.5, 22.5), 1 front-left, ... (CCW)."""
    az = np.degrees(spec.azimuths_rad())
    return (np.floor((az + 22.5) / 45.0).astype(int)) % 8


def scan(world_local, pos: np.ndarray, rot_body_to_world: np.ndarray, spec: Lidar3DSpec = DEFAULT_SPEC,
         dirs_body: Optional[np.ndarray] = None) -> np.ndarray:
    """Ranges (n_elev, n_az) seen from the vehicle pose in a (culled) World3D."""
    db = spec.dirs_body() if dirs_body is None else dirs_body
    dirs_world = db @ rot_body_to_world.T
    origin = np.asarray(pos, dtype=np.float64) + rot_body_to_world @ np.array(spec.mount_offset_body)
    r = world_local.raycast(origin, dirs_world, spec.max_range)
    r = np.where(r < spec.min_range, spec.min_range, r)
    return r.reshape(len(spec.elevations_deg), spec.n_azimuth)


def add_noise(ranges: np.ndarray, rng: np.random.Generator, noise_std: float, dropout: float,
              max_range: float) -> np.ndarray:
    r = ranges.copy()
    hit = r < max_range
    if noise_std > 0:
        r = np.where(hit, np.clip(r + rng.normal(0.0, noise_std, size=r.shape), 0.1, max_range), r)
    if dropout > 0:
        r = np.where(rng.random(r.shape) < dropout, max_range, r)
    return r


def filter_ground(ranges: np.ndarray, pos: np.ndarray, rot_body_to_world: np.ndarray,
                  spec: Lidar3DSpec = DEFAULT_SPEC, ground_z: float = 0.0, margin: float = 0.15) -> np.ndarray:
    """Treat returns from the ground plane as 'no return' (as a ground-segmentation stage would)."""
    db = spec.dirs_body()
    dirs_world = db @ rot_body_to_world.T
    origin = np.asarray(pos) + rot_body_to_world @ np.array(spec.mount_offset_body)
    r = ranges.reshape(-1)
    hit_z = origin[2] + r * dirs_world[:, 2]
    ground = (r < spec.max_range) & (hit_z < ground_z + margin)
    return np.where(ground, spec.max_range, r).reshape(ranges.shape)


def features(ranges: np.ndarray, spec: Lidar3DSpec = DEFAULT_SPEC) -> np.ndarray:
    """24 features: level(8), down(8), up(8) minimum range per sector."""
    global _SECTOR_OF_AZ
    if _SECTOR_OF_AZ is None or len(_SECTOR_OF_AZ) != spec.n_azimuth:
        _SECTOR_OF_AZ = sector_index(spec)
    out = np.full(24, spec.max_range, dtype=np.float32)
    for b, band in enumerate(("level", "down", "up")):
        band_min = ranges[_BAND_ROWS[band], :].min(axis=0)           # (n_az,)
        mins = np.full(8, spec.max_range)
        np.minimum.at(mins, _SECTOR_OF_AZ, band_min)
        out[8 * b:8 * b + 8] = mins
    return out
