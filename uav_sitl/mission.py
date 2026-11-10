"""
mission.py
----------
M10: end-to-end autonomous mission state machine for PX4 (SITL or real), driven by a
learned navigation policy plus the safety shield.

States
  PREFLIGHT   wait for PX4 position estimate, LiDAR and rangefinder data
  ARM         switch to OFFBOARD and arm
  TAKEOFF     climb vertically until the rangefinder reads the takeoff height
  NAVIGATE    learned policy (+ shield) flies to the active waypoint
  HOVER       settle briefly at a reached waypoint
  RETURN      navigate back to the takeoff point after a failed leg
  LAND        controlled descent (LiDAR down-band guard), until touchdown
  DONE / ABORTED

Fail-safes
  sensor stall (no LiDAR for > 0.5 s)    -> zero velocity, then LAND if it persists 3 s
  waypoint leg timeout                   -> skip to RETURN
  imminent collision (any LiDAR return closer than the hull + 0.15 m) -> emergency brake
  geofence (horizontal or ceiling)       -> RETURN
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from rl_env.safety_shield import SafetyShieldConfig
from uav_sitl.gz_link import GzLink
from uav_sitl.policy_runtime import PolicyRuntime
from uav_sitl.px4_link import PX4Link


@dataclass
class MissionConfig:
    waypoints: List[List[float]]
    takeoff_height: float = 2.0
    goal_tolerance: float = 0.5
    leg_timeout_s: float = 50.0
    hover_s: float = 1.5
    geofence_half: float = 19.5
    ceiling: float = 7.0
    hull_radius: float = 0.35
    brake_margin: float = 0.15
    land_speed: float = 0.5


@dataclass
class MissionLog:
    events: List[dict] = field(default_factory=list)
    legs: List[dict] = field(default_factory=list)

    def event(self, t: float, state: str, msg: str) -> None:
        self.events.append({"t": round(t, 2), "state": state, "msg": msg})


class Mission:
    def __init__(self, px4: PX4Link, gz: GzLink, model_path: str, obs_mode: str, cfg: MissionConfig,
                 shield: Optional[SafetyShieldConfig] = None, on_state: Optional[Callable[[str], None]] = None,
                 world=None, cbf=False):
        self.px4, self.gz, self.cfg = px4, gz, cfg
        self.model_path, self.obs_mode, self.shield_cfg, self.cbf = model_path, obs_mode, shield, cbf
        self.on_state = on_state or (lambda s: None)
        self.world = world                    # ground truth, only used for *logging* safety metrics
        self.state = "PREFLIGHT"
        self.log = MissionLog()
        self.home = np.zeros(3)
        self.wp_index = 0
        self.runtime: Optional[PolicyRuntime] = None
        self.min_truth_clearance = float("inf")    # min of the two below
        self.min_clear_static = float("inf")       # hull to trees/terrain objects, whole flight
        self.min_clear_movers = float("inf")       # hull to moving hazards, only while airborne

    # ------------------------------------------------------------- helpers
    def _t(self) -> float:
        tr = self.gz.get_truth()
        return tr.stamp if tr is not None else 0.0

    def _goto(self, state: str, msg: str = "") -> None:
        self.state = state
        self.log.event(self._t(), state, msg)
        self.on_state(state)

    def nav_state(self):
        tel = self.px4.snapshot()
        h = self.gz.height_above_ground(tel.roll, tel.pitch)
        pos = np.array([tel.pos_enu[0], tel.pos_enu[1], h if h is not None else tel.pos_enu[2]])
        return tel, pos

    def _check_truth(self) -> None:
        if self.world is not None:
            tr = self.gz.get_truth()
            if tr is not None:
                hull = (tr.pos + np.array([0.0, 0.0, 0.24]))[None]   # x500 base_link above model origin
                self.min_clear_static = min(self.min_clear_static,
                                            float(self.world.distance(hull)[0]) - self.cfg.hull_radius)
                self.min_truth_clearance = min(self.min_clear_static, self.min_clear_movers)

    # ------------------------------------------------------------------ run
    def run(self, timeout_s: float = 1200.0) -> dict:
        # wall-clock safety net only (legs time out on sim time); generous because a throttled host can run
        # PX4/Gazebo far below real time
        t_start = time.monotonic()
        legs_ok = 0
        while self.state not in ("DONE", "ABORTED") and time.monotonic() - t_start < timeout_s:
            getattr(self, f"_state_{self.state.lower()}")()
            self._check_truth()
        if self.state not in ("DONE", "ABORTED"):
            self._goto("ABORTED", "mission wall-clock timeout")
        legs_ok = sum(l["result"] == "reached" for l in self.log.legs)
        return {"final_state": self.state, "waypoints": len(self.cfg.waypoints), "waypoints_reached": legs_ok,
                "legs": self.log.legs, "events": self.log.events,
                "min_truth_clearance": None if self.world is None else round(self.min_truth_clearance, 3),
                "min_clear_static": None if self.world is None else round(self.min_clear_static, 3),
                "min_clear_movers": None if self.min_clear_movers == float("inf") else round(self.min_clear_movers, 3)}

    def airborne_for_movers(self, z: float) -> bool:
        """Mover contacts count only while the vehicle is flying (a walker passing a parked drone is not a
        flight collision)."""
        return self.state not in ("PREFLIGHT", "ARM", "DONE", "ABORTED") and z > 0.3

    def _state_preflight(self) -> None:
        ok = self.px4.wait_for(lambda t: t.have_position, 60.0) and self.gz.wait_truth(30.0)
        end = time.monotonic() + 30.0
        while ok and self.gz.get_scan() is None and time.monotonic() < end:
            time.sleep(0.1)
        if not ok or self.gz.get_scan() is None:
            self._goto("ABORTED", "preflight: missing position estimate or LiDAR")
            return
        tel, pos = self.nav_state()
        self.home = pos.copy()
        self._goto("ARM", f"preflight ok, home={np.round(self.home, 2).tolist()}")

    def _state_arm(self) -> None:
        for _ in range(3):
            if self.px4.start_offboard(timeout=15.0) and self.px4.arm(timeout=15.0):
                self._goto("TAKEOFF", "armed in OFFBOARD")
                return
            time.sleep(2.0)
        self._goto("ABORTED", "PX4 refused OFFBOARD/arming: " + " | ".join(self.px4.snapshot().status_texts[-3:]))

    def _state_takeoff(self) -> None:
        end = time.monotonic() + 40.0
        while time.monotonic() < end:
            tel, pos = self.nav_state()
            if pos[2] >= self.cfg.takeoff_height - 0.1:
                self.px4.set_velocity()
                self._goto("NAVIGATE", f"takeoff complete at {pos[2]:.2f} m")
                self._start_leg()
                return
            self.px4.set_velocity(vz=min(1.0, max(0.3, self.cfg.takeoff_height - pos[2])))
            time.sleep(0.05)
        self._goto("LAND", "takeoff timeout")

    def _start_leg(self, target=None) -> None:
        goal = np.asarray(target if target is not None else self.cfg.waypoints[self.wp_index], dtype=np.float64)
        self.runtime = PolicyRuntime(self.model_path, self.obs_mode, goal, shield=self.shield_cfg, cbf=self.cbf)
        self.leg = {"index": self.wp_index if target is None else -1, "goal": goal.tolist(), "t0": self._t(),
                    "last_scan": -1.0, "stall_wall": time.monotonic(), "stall_sim": self._t(), "brakes": 0,
                    "shield_steps": 0, "steps": 0}

    def _fly_leg(self) -> Optional[str]:
        """One control tick toward the active goal. Returns 'reached', 'timeout', 'geofence', 'stall' or None."""
        leg = self.leg
        scan = self.gz.get_scan()
        if scan is None or scan.stamp <= leg["last_scan"]:
            # perception age on SIM time (PX4/Gazebo are lockstepped: a slow frame is not a stall)
            stalled = self._t() - leg["stall_sim"]
            if stalled > 0.3:
                self.px4.set_velocity()                          # fail-safe: stop and hold
            if stalled > 3.0 or time.monotonic() - leg["stall_wall"] > 30.0:
                return "stall"
            time.sleep(0.005)
            return None
        leg["stall_wall"], leg["stall_sim"] = time.monotonic(), self._t()
        leg["last_scan"] = scan.stamp
        tel, pos = self.nav_state()
        if self._t() - leg["t0"] > self.cfg.leg_timeout_s:
            return "timeout"
        if abs(pos[0]) > self.cfg.geofence_half or abs(pos[1]) > self.cfg.geofence_half or pos[2] > self.cfg.ceiling:
            return "geofence"
        step = self.runtime.step(pos, tel.vel_enu, tel.quat_enu_flu, tel.rates_flu, scan)
        leg["steps"] += 1
        leg["shield_steps"] += int(step.shield_active)
        cmd = step.cmd
        if self.obs_mode == "50d":
            cmd, braked = self._emergency_brake(cmd, step.obs)
            leg["brakes"] += int(braked)
        self.px4.set_velocity(*cmd)
        goal = np.asarray(leg["goal"])
        if np.linalg.norm(pos - goal) <= self.cfg.goal_tolerance and np.linalg.norm(tel.vel_enu) <= 0.6:
            return "reached"
        return None

    def _emergency_brake(self, cmd, obs):
        """Last-resort layer under the shield: for every LiDAR sector with a return inside hull radius +
        margin, cancel the commanded velocity component toward that sector (heading frame, sector k centred
        at 45k deg CCW from the nose); a close return in the down / up band also forbids descending /
        climbing. Motion away from the obstacle stays allowed, so the vehicle cannot get pinned."""
        limit = self.cfg.hull_radius + self.cfg.brake_margin
        v = np.array(cmd[:3], dtype=np.float64)
        braked = False
        for band, (lo, hi) in enumerate(((18, 26), (26, 34), (34, 42))):
            for k in np.flatnonzero(obs[lo:hi] < limit):
                u = np.array([math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)])
                toward = float(v[:2] @ u)
                if toward > 0.0:
                    v[:2] -= toward * u
                braked = True
            if band == 1 and np.any(obs[lo:hi] < limit):
                v[2] = max(v[2], 0.0)
            if band == 2 and np.any(obs[lo:hi] < limit):
                v[2] = min(v[2], 0.0)
        return (float(v[0]), float(v[1]), float(v[2]), float(cmd[3])), braked

    def _finish_leg(self, result: str) -> None:
        leg = self.leg
        tel, pos = self.nav_state()
        tr = self.gz.get_truth()
        self.log.legs.append({"index": leg["index"], "goal": leg["goal"], "result": result,
                              "duration_s": round(self._t() - leg["t0"], 2), "steps": leg["steps"],
                              "shield_rate": round(leg["shield_steps"] / max(leg["steps"], 1), 3),
                              "emergency_brakes": leg["brakes"],
                              "truth_error_m": round(float(np.linalg.norm(tr.pos - np.asarray(leg["goal"]))), 3)})

    def _state_navigate(self) -> None:
        result = self._fly_leg()
        if result is None:
            return
        self._finish_leg(result)
        if result == "reached":
            self._goto("HOVER", f"waypoint {self.wp_index} reached")
        elif result == "stall":
            self._goto("LAND", f"sensor stall on leg {self.wp_index}")
        else:
            self._goto("RETURN", f"leg {self.wp_index} failed ({result}); returning home")
            self._start_leg(target=[self.home[0], self.home[1], self.cfg.takeoff_height])

    def _state_hover(self) -> None:
        self.px4.set_velocity()
        time.sleep(self.cfg.hover_s)
        self.wp_index += 1
        if self.wp_index >= len(self.cfg.waypoints):
            self._goto("LAND", "all waypoints done")
        else:
            self._goto("NAVIGATE", f"leg {self.wp_index} start")
            self._start_leg()

    def _state_return(self) -> None:
        result = self._fly_leg()
        if result is None:
            return
        self._finish_leg("returned" if result == "reached" else f"return_{result}")
        self._goto("LAND", f"return leg ended: {result}")

    def _state_land(self) -> None:
        end = time.monotonic() + 40.0
        nudges = 0
        while time.monotonic() < end:
            tel, pos = self.nav_state()
            scan = self.gz.get_scan()
            vx = vy = 0.0
            down_clear = True
            if self.runtime is not None and scan is not None and self.obs_mode == "50d":
                obs = self.runtime.observe(pos, tel.vel_enu, tel.quat_enu_flu, tel.rates_flu, scan)
                down = obs[26:34]
                down_clear = float(np.min(down)) > 0.8 or pos[2] < 0.6
                if not down_clear:
                    # something below/ahead-below (bush, log): hold height and sidestep away from it
                    k = int(np.argmin(down))
                    vx, vy = -0.3 * math.cos(k * math.pi / 4), -0.3 * math.sin(k * math.pi / 4)
                    nudges += 1
            vz = -min(self.cfg.land_speed, max(0.2, pos[2])) if down_clear else 0.0
            self.px4.set_velocity(vx=vx, vy=vy, vz=vz)
            if pos[2] < 0.25 and abs(tel.vel_enu[2]) < 0.15:
                break
            time.sleep(0.05)
        self.px4.set_velocity()
        tel, pos = self.nav_state()
        # Touchdown and disarm are left to PX4's AUTO.LAND from wherever the offboard descent ended: its land
        # detector confirms ground contact (an offboard "vz = 0" setpoint on the ground keeps hover thrust,
        # so the detector never fires and PX4 refuses to disarm) and it disarms automatically. Motors are
        # never cut in the air; a forced disarm is used only if PX4 itself reports ON_GROUND.
        if pos[2] >= 0.4:
            self.log.event(self._t(), "LAND", f"offboard descent incomplete at {pos[2]:.2f} m")
        disarmed = self.px4.land(timeout=40.0)
        if not disarmed and self.px4.snapshot().landed_state == 1:
            self.px4.disarm(force=True)
            disarmed = self.px4.wait_for(lambda t: not t.armed, 5.0)
        tr = self.gz.get_truth()
        self.log.event(self._t(), "LAND", f"touchdown at truth z={tr.pos[2]:.2f}, disarmed={disarmed}, "
                                          f"sidestep ticks={nudges}")
        self._goto("DONE" if disarmed else "ABORTED", "landed" if disarmed else "could not disarm")
