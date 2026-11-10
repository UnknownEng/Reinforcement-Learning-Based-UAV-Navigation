"""
movers.py
---------
Moving obstacles in Gazebo, driven by the same kinematics as training
(rl_env.uav_world_env.Movers): walkers (1.8 m people-sized cylinders on the ground)
and flyers (spheres at 1.5-4 m). Each is a static Gazebo model repositioned through
the /world/<world>/set_pose service by a separate process (uav_sitl/movers_daemon.py), so the
control loop never shares an interpreter with Gazebo service calls (in-process updates cut the
policy rate from 10 Hz to 2-9 Hz). Positions are a deterministic function of absolute sim time
(fixed 0.05 s kinematic steps from the same seed): the daemon and the control process, which uses
them for ground-truth collision checks, agree exactly.
"""

import sys
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from rl_env.uav_world_env import Movers, UAVWorldEnvConfig
from rl_env.world3d import World3D
from uav_sitl.gz_link import GzLink


class SitlMovers:
    STEP = 0.05          # s of sim time per kinematic step: identical in every process -> identical positions

    def __init__(self, seed: int, n: int, avoid_xy=(0.0, 0.0), half_size: float = 20.0):
        cfg = UAVWorldEnvConfig(half_size=half_size)
        self.seed, self.n, self.avoid_xy = seed, n, tuple(float(a) for a in avoid_xy)
        self.movers = Movers(np.random.default_rng(seed), cfg, n, np.asarray(avoid_xy, dtype=float))
        self.t_sim = 0.0  # absolute sim time the kinematics have been stepped to

    def names(self):
        return [f"mover_{i}" for i in range(len(self.movers.pos))]

    def add_to_sdf(self, tree: ET.ElementTree) -> None:
        w = tree.getroot().find("world")
        for name, kind, p, r in zip(self.names(), self.movers.kind, self.movers.pos, self.movers.radius):
            m = ET.SubElement(w, "model", {"name": name})
            ET.SubElement(m, "static").text = "true"
            z = 0.9 if kind == "walker" else p[2]
            ET.SubElement(m, "pose").text = f"{p[0]:.3f} {p[1]:.3f} {z:.3f} 0 0 0"
            link = ET.SubElement(m, "link", {"name": "link"})
            geom = f"<cylinder><radius>{r:.3f}</radius><length>1.8</length></cylinder>" if kind == "walker"                 else f"<sphere><radius>{r:.3f}</radius></sphere>"
            for k in ("collision", "visual"):
                el = ET.SubElement(link, k, {"name": k})
                g = ET.SubElement(el, "geometry")
                g.append(ET.fromstring(geom))
                if k == "visual":
                    mat = ET.SubElement(el, "material")
                    ET.SubElement(mat, "ambient").text = "0.85 0.15 0.15 1"
                    ET.SubElement(mat, "diffuse").text = "0.85 0.15 0.15 1"

    def advance_to(self, sim_t: float) -> None:
        """Step the hazards in fixed increments up to absolute sim time `sim_t` (no Gazebo calls)."""
        while self.t_sim + self.STEP <= sim_t + 1e-9:
            self.movers.step(self.STEP)
            self.t_sim += self.STEP

    def update(self, gz: GzLink, sim_t: float) -> None:
        """Advance and push the poses to Gazebo (used by the daemon process)."""
        self.advance_to(sim_t)
        for name, kind, p in zip(self.names(), self.movers.kind, self.movers.pos):
            gz.set_pose(name, (p[0], p[1], 0.9 if kind == "walker" else p[2]))

    def daemon_argv(self, world: str, model: str) -> list:
        """Command line that moves these hazards in Gazebo from a separate process."""
        return [sys.executable, "-m", "uav_sitl.movers_daemon", "--world", world, "--model", model,
                "--seed", str(self.seed), "--n", str(self.n), "--avoid", str(self.avoid_xy[0]), str(self.avoid_xy[1])]

    def as_world(self, half_size: float = 20.0) -> World3D:
        cyl, sph = self.movers.primitives()
        return World3D(cyl, sph, np.zeros((0, 7)), half_size, "movers")
