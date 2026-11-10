"""
observation_26d.py
------------------
Experimental 26-dimensional observation space definition and builder for Milestone 4 Phase 4B.
Strictly isolated from frozen production adapters and the frozen 18-D baseline.

Layout (26 dimensions):
  obs[0:3]   : Relative goal position in body FLU frame [xg_b, yg_b, zg_b] (meters)
  obs[3:6]   : Linear velocity in body FLU frame [vx_b, vy_b, vz_b] (m/s)
  obs[6:9]   : Angular rates in body FLU frame [p, q, r] (rad/s)
  obs[9:13]  : Unit attitude quaternion [qw, qx, qy, qz] (dimensionless)
  obs[13]    : Relative altitude above ground datum h_rel (meters)
  obs[14:18] : Previous normalized action [a0, a1, a2, a3] in [-1, +1]^4
  obs[18:26] : 8-sector directional obstacle clearances [d0, ..., d7] in meters

Sector Ordering:
  obs[18] : Sector 0 - Front        [-22.5 deg, +22.5 deg]
  obs[19] : Sector 1 - Front-Left   [+22.5 deg, +67.5 deg]
  obs[20] : Sector 2 - Left         [+67.5 deg, +112.5 deg]
  obs[21] : Sector 3 - Back-Left    [+112.5 deg, +157.5 deg]
  obs[22] : Sector 4 - Back         [+157.5 deg, +180 deg] / [-180 deg, -157.5 deg]
  obs[23] : Sector 5 - Back-Right   [-157.5 deg, -112.5 deg]
  obs[24] : Sector 6 - Right        [-112.5 deg, -67.5 deg]
  obs[25] : Sector 7 - Front-Right  [-67.5 deg, -22.5 deg]

Conservative Deterministic Sensor Handling Policy:
  - Missing, stale, corrupt, or unobserved sensor data is NEVER silently interpreted as clear space (10.0 m).
  - Genuine measured clear space (+Inf or > max_range on active beams) = 10.0 m (MEASURED_CLEAR).
  - Genuine obstacle hits [min_range, max_range] = min(rays) (VALID_HIT).
  - Sub-minimum range returns (< 0.10 m) = 0.10 m (SUBMIN_PROXIMITY hazard).
  - Unobserved sectors (e.g. rear sectors with forward-only LiDAR), corrupt scans (NaN/Inf/-ve),
    or stale scans (> staleness_timeout_s) default to a conservative safe bound:
    0.0 m under CONSERVATIVE_ZERO (default: zero guaranteed clearance / hazard stop).
  - Explicit validity metadata is tracked and returned to distinguish measured clear space from invalid/stale data.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.observation import canonical_unit_quaternion, quat_to_rotation_matrix


class SectorStatus(str, Enum):
    VALID_HIT = "VALID_HIT"                    # Genuine obstacle detected in range [min_range, max_range]
    MEASURED_CLEAR = "MEASURED_CLEAR"          # Sensor actively scanned sector; all rays clear (+Inf / > max_range)
    SUBMIN_PROXIMITY = "SUBMIN_PROXIMITY"      # Range < min_range detected (proximity hazard)
    UNOBSERVED_BLIND_ZONE = "UNOBSERVED"       # No physical beams cover this sector (e.g. rear hemisphere)
    CORRUPT_INVALID = "CORRUPT_INVALID"        # Rays contain NaN, -Inf, or negative values without valid hits
    STALE_TIMED_OUT = "STALE_TIMED_OUT"        # Sensor message timestamp is older than staleness threshold
    UNAVAILABLE = "UNAVAILABLE"                # Sensor message is None or completely absent


class InvalidClearancePolicy(str, Enum):
    CONSERVATIVE_ZERO = "CONSERVATIVE_ZERO"            # Default: 0.0 m (zero clearance, marks space as hazard)
    CONSERVATIVE_MIN_RANGE = "CONSERVATIVE_MIN_RANGE"  # 0.10 m (clamped to sensor min range)
    SAFE_MARGIN = "SAFE_MARGIN"                        # 1.0 m (conservative standoff distance)
    OPTIMISTIC_MAX = "OPTIMISTIC_MAX"                  # 10.0 m (UNSAFE: warns if used; kept only for comparative audit)


@dataclass
class Observation26DMetadata:
    is_sensor_stale: bool = False
    sensor_age_s: float = 0.0
    is_perception_valid: bool = True
    sector_status: List[SectorStatus] = field(default_factory=list)
    validity_mask: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=bool))
    raw_clearances: List[float] = field(default_factory=list)
    sanitized_clearances: List[float] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ObservationHandler26D:
    """
    26-dimensional observation handler for autonomous UAV navigation with LiDAR clearance.
    Maintains complete isolation from production 18-D flight adapters.
    """

    SECTOR_NAMES: List[str] = [
        "front",
        "front_left",
        "left",
        "back_left",
        "back",
        "back_right",
        "right",
        "front_right"
    ]

    def __init__(
        self,
        config: Optional[UAVEnvConfig] = None,
        invalid_policy: InvalidClearancePolicy = InvalidClearancePolicy.CONSERVATIVE_ZERO,
        staleness_timeout_s: float = 0.20,
        min_range: float = 0.10,
        max_range: float = 10.0,
        default_clear_range: float = 10.0
    ):
        self.config = config if config is not None else UAVEnvConfig()
        self.obs_dim: int = 26
        self.proprio_dim: int = 18
        self.lidar_dim: int = 8

        self.invalid_policy: InvalidClearancePolicy = invalid_policy
        self.staleness_timeout_s: float = float(staleness_timeout_s)
        self.min_range: float = float(min_range)
        self.max_range: float = float(max_range)
        self.default_clear_range: float = float(default_clear_range)

        # Performance cache for static beam geometry
        self._sector_cache_key: Optional[Tuple[int, float, float]] = None
        self._sector_cache_indices: Optional[List[int]] = None

        # Define 26D observation space bounds
        low = np.array([
            # 18-D Proprioceptive (identical to baseline ObservationHandler)
            -50.0, -50.0, -50.0,  # goal_body (x, y, z)
            -15.0, -15.0, -15.0,  # vel_body (vx, vy, vz)
            -10.0, -10.0, -10.0,  # rates_body (p, q, r)
            -1.0,  -1.0,  -1.0,  -1.0,  # quat (qw, qx, qy, qz)
            0.0,                  # relative_altitude
            -1.0,  -1.0,  -1.0,  -1.0,  # prev_action (a0, a1, a2, a3)
            # 8-D Obstacle Clearance Sectors (meters)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        high = np.array([
            # 18-D Proprioceptive (identical to baseline ObservationHandler)
            50.0,  50.0,  50.0,   # goal_body (x, y, z)
            15.0,  15.0,  15.0,   # vel_body (vx, vy, vz)
            10.0,  10.0,  10.0,   # rates_body (p, q, r)
            1.0,   1.0,   1.0,   1.0,   # quat (qw, qx, qy, qz)
            50.0,                 # relative_altitude
            1.0,   1.0,   1.0,   1.0,   # prev_action (a0, a1, a2, a3)
            # 8-D Obstacle Clearance Sectors (meters)
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def get_invalid_fallback_clearance(self) -> float:
        """Return the conservative clearance value based on the active policy."""
        if self.invalid_policy == InvalidClearancePolicy.CONSERVATIVE_ZERO:
            return 0.0
        elif self.invalid_policy == InvalidClearancePolicy.CONSERVATIVE_MIN_RANGE:
            return self.min_range
        elif self.invalid_policy == InvalidClearancePolicy.SAFE_MARGIN:
            return 1.0
        elif self.invalid_policy == InvalidClearancePolicy.OPTIMISTIC_MAX:
            return self.default_clear_range
        return 0.0

    @staticmethod
    def angle_to_sector(angle_rad: float) -> int:
        """Map an angle in [-pi, pi] to sector index 0..7."""
        ang = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
        deg = round(math.degrees(ang), 6)

        if -22.5 <= deg < 22.5:
            return 0  # Front
        elif 22.5 <= deg < 67.5:
            return 1  # Front-Left
        elif 67.5 <= deg < 112.5:
            return 2  # Left
        elif 112.5 <= deg < 157.5:
            return 3  # Back-Left
        elif deg >= 157.5 or deg < -157.5:
            return 4  # Back
        elif -157.5 <= deg < -112.5:
            return 5  # Back-Right
        elif -112.5 <= deg < -67.5:
            return 6  # Right
        elif -67.5 <= deg < -22.5:
            return 7  # Front-Right
        return 4

    def process_raw_scan(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        sensor_time: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> Tuple[np.ndarray, Observation26DMetadata]:
        """
        Processes an array of raw range beams into 8 conservative directional sector clearances.
        Applies strict invalid-data handling: unmeasured/stale/corrupt data is never treated as clear.
        """
        num_sectors = 8
        sector_all_rays: List[List[float]] = [[] for _ in range(num_sectors)]
        sector_valid_hits: List[List[float]] = [[] for _ in range(num_sectors)]
        sector_free_paths: List[int] = [0 for _ in range(num_sectors)]
        sector_submin: List[int] = [0 for _ in range(num_sectors)]
        sector_corrupt: List[int] = [0 for _ in range(num_sectors)]

        # Check staleness
        is_stale = False
        sensor_age = 0.0
        if sensor_time is not None and current_time is not None:
            sensor_age = max(0.0, current_time - sensor_time)
            if sensor_age > self.staleness_timeout_s:
                is_stale = True

        fallback_val = self.get_invalid_fallback_clearance()

        # If sensor is stale or ranges empty, trigger conservative fallback across all sectors
        if is_stale or len(ranges) == 0:
            status_list = [
                SectorStatus.STALE_TIMED_OUT if is_stale else SectorStatus.UNAVAILABLE
            ] * num_sectors
            mask = np.zeros(num_sectors, dtype=bool)
            clearances = np.full(num_sectors, fallback_val, dtype=np.float32)

            meta = Observation26DMetadata(
                is_sensor_stale=is_stale,
                sensor_age_s=sensor_age,
                is_perception_valid=False,
                sector_status=status_list,
                validity_mask=mask,
                raw_clearances=[float(r) for r in clearances],
                sanitized_clearances=[float(r) for r in clearances],
                diagnostics={"reason": "STALE_OR_EMPTY_SCAN", "sensor_age_s": sensor_age}
            )
            return clearances, meta

        # Bin rays into sectors
        num_rays = len(ranges)
        cache_key = (num_rays, float(angle_min), float(angle_increment))
        if self._sector_cache_key == cache_key and self._sector_cache_indices is not None:
            sector_indices = self._sector_cache_indices
        else:
            sector_indices = [self.angle_to_sector(angle_min + i * angle_increment) for i in range(num_rays)]
            self._sector_cache_key = cache_key
            self._sector_cache_indices = sector_indices

        for i in range(num_rays):
            sec_idx = sector_indices[i]
            r = float(ranges[i])

            sector_all_rays[sec_idx].append(r)

            if math.isnan(r) or (math.isinf(r) and r < 0.0) or r < 0.0:
                sector_corrupt[sec_idx] += 1
            elif (math.isinf(r) and r > 0.0) or r >= self.max_range:
                # Genuine clear space (no return within max range or reaches/exceeds max range)
                sector_free_paths[sec_idx] += 1
            elif r < self.min_range:
                # Proximity hazard (< min_range)
                sector_submin[sec_idx] += 1
            else:
                # Genuine finite obstacle hit in [min_range, max_range)
                sector_valid_hits[sec_idx].append(r)

        clearances = np.zeros(num_sectors, dtype=np.float32)
        status_list: List[SectorStatus] = []
        validity_mask = np.zeros(num_sectors, dtype=bool)

        for sec_idx in range(num_sectors):
            total_rays = len(sector_all_rays[sec_idx])
            valid_hits = sector_valid_hits[sec_idx]
            free_count = sector_free_paths[sec_idx]
            submin_count = sector_submin[sec_idx]
            corrupt_count = sector_corrupt[sec_idx]

            if total_rays == 0:
                # Unobserved blind zone (e.g. rear sectors with 180 deg forward lidar)
                # NEVER interpret as 10.0m clear space!
                clearance_val = fallback_val
                status = SectorStatus.UNOBSERVED_BLIND_ZONE
                is_valid = False
            elif len(valid_hits) > 0:
                # Valid obstacle detected
                clearance_val = float(min(valid_hits))
                status = SectorStatus.VALID_HIT
                is_valid = True
            elif submin_count > 0:
                # Proximity hazard (< min_range)
                clearance_val = self.min_range
                status = SectorStatus.SUBMIN_PROXIMITY
                is_valid = True
            elif free_count > 0 and corrupt_count == 0:
                # Genuine measured clear space: all rays in sector observed open path
                clearance_val = self.default_clear_range
                status = SectorStatus.MEASURED_CLEAR
                is_valid = True
            else:
                # Corrupt / invalid scan rays without any valid hits or clear returns
                clearance_val = fallback_val
                status = SectorStatus.CORRUPT_INVALID
                is_valid = False

            clearances[sec_idx] = np.clip(clearance_val, 0.0, self.max_range)
            status_list.append(status)
            validity_mask[sec_idx] = is_valid

        all_valid = bool(np.all(validity_mask[:3]) and validity_mask[6] and validity_mask[7])  # Frontal sectors valid

        meta = Observation26DMetadata(
            is_sensor_stale=False,
            sensor_age_s=sensor_age,
            is_perception_valid=all_valid,
            sector_status=status_list,
            validity_mask=validity_mask,
            raw_clearances=[float(c) for c in clearances],
            sanitized_clearances=[float(c) for c in clearances],
            diagnostics={
                "total_rays": num_rays,
                "active_sectors": int(np.sum(validity_mask)),
                "blind_sectors": int(np.sum([s == SectorStatus.UNOBSERVED_BLIND_ZONE for s in status_list]))
            }
        )
        return clearances, meta

    def sanitize_clearances_array(
        self,
        clearances: Sequence[float],
        sector_validity_mask: Optional[Sequence[bool]] = None,
        require_explicit_validity_for_inf: bool = True
    ) -> Tuple[np.ndarray, Observation26DMetadata]:
        """
        Sanitize an existing 8-element clearance array against non-finite or out-of-range values.
        Applies conservative policy to any corrupt, unobserved, or invalid element.

        Direct-Array Ambiguity Resolution:
          - A bare positive infinity (+Inf) in a direct array CANNOT be verified as an active
            measurement unless an explicit sector_validity_mask is provided with mask[i] == True.
          - If require_explicit_validity_for_inf is True and sector_validity_mask is None,
            bare +Inf is rejected and clamped to the conservative fallback (0.0 m) with
            SectorStatus.UNOBSERVED_BLIND_ZONE, preventing unobserved space from claiming 10.0 m.
        """
        if len(clearances) != 8:
            raise ValueError(f"Clearances array must have length 8, got {len(clearances)}")
        if sector_validity_mask is not None and len(sector_validity_mask) != 8:
            raise ValueError(f"sector_validity_mask must have length 8, got {len(sector_validity_mask)}")

        fallback_val = self.get_invalid_fallback_clearance()
        clean = np.zeros(8, dtype=np.float32)
        status_list: List[SectorStatus] = []
        mask = np.zeros(8, dtype=bool)

        for i, val in enumerate(clearances):
            v = float(val)
            is_explicitly_valid = (sector_validity_mask[i] is True) if sector_validity_mask is not None else None

            if is_explicitly_valid is False:
                # Caller explicitly flagged sector as unobserved or invalid
                clean[i] = fallback_val
                status_list.append(SectorStatus.UNOBSERVED_BLIND_ZONE)
                mask[i] = False
            elif math.isnan(v) or (math.isinf(v) and v < 0.0) or v < 0.0:
                clean[i] = fallback_val
                status_list.append(SectorStatus.CORRUPT_INVALID)
                mask[i] = False
            elif math.isinf(v) and v > 0.0:
                # Direct-Array Ambiguity Resolution Check:
                if require_explicit_validity_for_inf and is_explicitly_valid is not True:
                    # Bare +Inf without verified sensor coverage is treated conservatively as unobserved
                    clean[i] = fallback_val
                    status_list.append(SectorStatus.UNOBSERVED_BLIND_ZONE)
                    mask[i] = False
                else:
                    # Verified active scan confirmation
                    clean[i] = self.max_range
                    status_list.append(SectorStatus.MEASURED_CLEAR)
                    mask[i] = True
            elif v >= self.max_range:
                # Finite value >= max_range is bounded clear
                clean[i] = self.max_range
                status_list.append(SectorStatus.MEASURED_CLEAR)
                mask[i] = True
            elif v < self.min_range:
                clean[i] = self.min_range
                status_list.append(SectorStatus.SUBMIN_PROXIMITY)
                mask[i] = True
            else:
                clean[i] = v
                status_list.append(SectorStatus.VALID_HIT)
                mask[i] = True

        meta = Observation26DMetadata(
            is_sensor_stale=False,
            sensor_age_s=0.0,
            is_perception_valid=bool(np.all(mask)),
            sector_status=status_list,
            validity_mask=mask,
            raw_clearances=[float(c) for c in clearances],
            sanitized_clearances=[float(c) for c in clean],
            diagnostics={
                "direct_clearance_input": True,
                "explicit_validity_provided": sector_validity_mask is not None
            }
        )
        return clean, meta

    def compute_observation(
        self,
        pos_world: np.ndarray,
        vel_world: np.ndarray,
        quat: np.ndarray,
        rates_body: np.ndarray,
        goal_world: np.ndarray,
        prev_action: np.ndarray,
        clearances: Optional[Sequence[float]] = None,
        sector_validity_mask: Optional[Sequence[bool]] = None,
        raw_scan: Optional[Tuple[Sequence[float], float, float]] = None,
        ground_z: float = 0.0,
        sensor_time: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Construct the unified 26D observation vector.
        Matches Gymnasium Box specification. Returns np.ndarray of shape (26,), dtype np.float32.
        """
        obs, _meta = self.compute_observation_with_metadata(
            pos_world=pos_world,
            vel_world=vel_world,
            quat=quat,
            rates_body=rates_body,
            goal_world=goal_world,
            prev_action=prev_action,
            clearances=clearances,
            sector_validity_mask=sector_validity_mask,
            raw_scan=raw_scan,
            ground_z=ground_z,
            sensor_time=sensor_time,
            current_time=current_time
        )
        return obs

    def compute_observation_with_metadata(
        self,
        pos_world: np.ndarray,
        vel_world: np.ndarray,
        quat: np.ndarray,
        rates_body: np.ndarray,
        goal_world: np.ndarray,
        prev_action: np.ndarray,
        clearances: Optional[Sequence[float]] = None,
        sector_validity_mask: Optional[Sequence[bool]] = None,
        raw_scan: Optional[Tuple[Sequence[float], float, float]] = None,
        ground_z: float = 0.0,
        sensor_time: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> Tuple[np.ndarray, Observation26DMetadata]:
        """
        Construct the unified 26D observation vector and return detailed validity metadata.
        """
        # 1. Compute 18-D Proprioceptive State (strictly identical to baseline ObservationHandler)
        R = quat_to_rotation_matrix(quat)
        R_trans = R.T  # World to Body

        rel_pos_world = (goal_world - pos_world).astype(np.float32)
        goal_body = R_trans @ rel_pos_world

        vel_body = R_trans @ vel_world.astype(np.float32)
        rates = rates_body.astype(np.float32)

        q_unit = canonical_unit_quaternion(quat)

        rel_alt = float(pos_world[2] - ground_z)
        act_prev = np.asarray(prev_action, dtype=np.float32).flatten()

        proprio_18 = np.concatenate([
            goal_body,       # 3
            vel_body,        # 3
            rates,           # 3
            q_unit,          # 4
            [rel_alt],       # 1
            act_prev         # 4
        ]).astype(np.float32)

        # Sanitize 18D proprioceptive slice
        if not np.all(np.isfinite(proprio_18)):
            proprio_18 = np.nan_to_num(proprio_18, nan=0.0, posinf=1.0, neginf=-1.0)
        proprio_18 = np.clip(proprio_18, self.observation_space.low[:18], self.observation_space.high[:18])

        # 2. Compute / Sanitize 8-D LiDAR Clearances
        if raw_scan is not None:
            ranges, a_min, a_inc = raw_scan
            clean_clearances, meta = self.process_raw_scan(
                ranges=ranges,
                angle_min=a_min,
                angle_increment=a_inc,
                sensor_time=sensor_time,
                current_time=current_time
            )
        elif clearances is not None:
            clean_clearances, meta = self.sanitize_clearances_array(
                clearances=clearances,
                sector_validity_mask=sector_validity_mask
            )
        else:
            # Sensor completely unavailable / missing
            fallback_val = self.get_invalid_fallback_clearance()
            clean_clearances = np.full(8, fallback_val, dtype=np.float32)
            meta = Observation26DMetadata(
                is_sensor_stale=True,
                sensor_age_s=float("inf"),
                is_perception_valid=False,
                sector_status=[SectorStatus.UNAVAILABLE] * 8,
                validity_mask=np.zeros(8, dtype=bool),
                raw_clearances=[],
                sanitized_clearances=[float(c) for c in clean_clearances],
                diagnostics={"reason": "NO_SENSOR_DATA_PROVIDED"}
            )

        # Clip clearances to observation space bounds
        clean_clearances = np.clip(clean_clearances, self.observation_space.low[18:], self.observation_space.high[18:])

        # 3. Assemble Unified 26-D Vector
        obs_26d = np.concatenate([proprio_18, clean_clearances]).astype(np.float32)

        return obs_26d, meta

    @staticmethod
    def extract_proprioceptive_18d(obs_26d: np.ndarray) -> np.ndarray:
        """Extract the 18D proprioceptive state from a 26D observation vector."""
        if len(obs_26d) != 26:
            raise ValueError(f"Expected 26D vector, got length {len(obs_26d)}")
        return obs_26d[:18].copy()

    @staticmethod
    def extract_clearances_8d(obs_26d: np.ndarray) -> np.ndarray:
        """Extract the 8D directional clearance values from a 26D observation vector."""
        if len(obs_26d) != 26:
            raise ValueError(f"Expected 26D vector, got length {len(obs_26d)}")
        return obs_26d[18:26].copy()
