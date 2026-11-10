"""
world_sdf.py
------------
Export an rl_env World3D (forest / urban / warehouse / poles) to a Gazebo Harmonic
world for PX4 SITL, plus a JSON copy of the exact primitives so trial runners can
check collisions against ground truth.

The world is built on PX4's own default.sdf (physics, sensors, sun, ground plane),
with one static model holding every obstacle as a collision + visual link.
"""

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

from rl_env.world3d import World3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "trunk": "0.36 0.25 0.15 1", "sapling": "0.42 0.30 0.18 1", "branch": "0.40 0.28 0.16 1",
    "log": "0.33 0.23 0.14 1", "crown": "0.16 0.42 0.14 1", "foliage": "0.22 0.52 0.18 1",
    "bush": "0.12 0.34 0.12 1", "building": "0.62 0.62 0.64 1", "vehicle": "0.20 0.30 0.65 1",
    "kiosk": "0.70 0.55 0.20 1", "pole": "0.35 0.35 0.38 1", "rack": "0.85 0.45 0.10 1",
    "crate": "0.72 0.58 0.38 1", "pillar": "0.55 0.55 0.58 1", "mover": "0.85 0.15 0.15 1",
}


def _link(link, name, pose, geometry_xml, tag, collisions):
    """Add one obstacle as visual (+ optional collision) elements of the shared static link."""
    for kind in (("collision", "visual") if collisions else ("visual",)):
        el = ET.SubElement(link, kind, {"name": f"{kind}_{name}"})
        ET.SubElement(el, "pose").text = " ".join(f"{v:.4f}" for v in pose)
        geom = ET.SubElement(el, "geometry")
        geom.append(ET.fromstring(geometry_xml))
        if kind == "visual":
            mat = ET.SubElement(el, "material")
            col = COLORS.get(str(tag), "0.5 0.5 0.5 1")
            ET.SubElement(mat, "ambient").text = col
            ET.SubElement(mat, "diffuse").text = col


def _natural_look(w: ET.Element) -> None:
    """Sky-blue background and a forest-floor ground instead of PX4's grey defaults, for camera footage.
    Colour only: the LiDAR and the rangefinder measure geometry, so sensing is unchanged."""
    bg = w.find("scene/background")
    if bg is not None:
        bg.text = "0.62 0.78 0.92 1"
    for m in w.findall("model"):
        if m.get("name") == "ground_plane":
            for mat in m.iter("material"):
                for key, val in (("ambient", "0.30 0.27 0.18 1"), ("diffuse", "0.34 0.30 0.20 1"),
                                 ("specular", "0.05 0.05 0.05 1")):
                    el = mat.find(key)
                    if el is not None:
                        el.text = val


def world_to_sdf(world: World3D, name: str, template: Optional[Path] = None, collisions: bool = False) -> ET.ElementTree:
    """
    All obstacles go into ONE static link (one entity) so Gazebo loads large forests quickly.
    By default only visual geometry is emitted: the gpu_lidar sees visuals, and trial runners
    judge collisions exactly from the JSON ground truth, so physics contact shapes are optional.
    """
    px4 = Path(os.environ.get("PX4_DIR", Path.home() / "projects/PX4-Autopilot"))
    template = template or px4 / "Tools/simulation/gz/worlds/default.sdf"
    tree = ET.parse(template)
    w = tree.getroot().find("world")
    w.set("name", name)
    _natural_look(w)
    model_el = ET.SubElement(w, "model", {"name": f"{name}_obstacles"})
    ET.SubElement(model_el, "static").text = "true"
    model = ET.SubElement(model_el, "link", {"name": "obstacles"})
    _l = _link
    _link_bound = lambda parent, nm, pose, geom, tag: _l(parent, nm, pose, geom, tag, collisions)
    tags = lambda t, n: t if t is not None and len(t) == n else np.array(["pole"] * n)
    for i, ((cx, cy, r, z0, z1), tag) in enumerate(zip(world.cyl, tags(world.cyl_tag, len(world.cyl)))):
        _link_bound(model, f"c{i}", (cx, cy, (z0 + z1) / 2, 0, 0, 0),
              f"<cylinder><radius>{r:.4f}</radius><length>{z1 - z0:.4f}</length></cylinder>", tag)
    for i, ((cx, cy, cz, r), tag) in enumerate(zip(world.sph, tags(world.sph_tag, len(world.sph)))):
        _link_bound(model, f"s{i}", (cx, cy, cz, 0, 0, 0), f"<sphere><radius>{r:.4f}</radius></sphere>", tag)
    for i, ((cx, cy, cz, hx, hy, hz, yaw), tag) in enumerate(zip(world.box, tags(world.box_tag, len(world.box)))):
        _link_bound(model, f"b{i}", (cx, cy, cz, 0, 0, yaw),
              f"<box><size>{2 * hx:.4f} {2 * hy:.4f} {2 * hz:.4f}</size></box>", tag)
    for i, ((ax, ay, az, bx, by, bz, r), tag) in enumerate(zip(world.cap, tags(world.cap_tag, len(world.cap)))):
        d = np.array([bx - ax, by - ay, bz - az])
        length = float(np.linalg.norm(d))
        yaw = math.atan2(d[1], d[0])
        pitch = math.acos(max(-1.0, min(1.0, d[2] / max(length, 1e-9))))
        _link_bound(model, f"k{i}", ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2, 0, pitch, yaw),
              f"<capsule><radius>{r:.4f}</radius><length>{length:.4f}</length></capsule>", tag)
    return tree


def world_to_dict(world: World3D) -> dict:
    f = lambda a: np.asarray(a).tolist()
    return {"kind": world.kind, "half_size": world.half_size, "meta": world.meta, "cyl": f(world.cyl),
            "sph": f(world.sph), "box": f(world.box), "cap": f(world.cap),
            "cyl_tag": f(world.cyl_tag) if world.cyl_tag is not None else [],
            "sph_tag": f(world.sph_tag) if world.sph_tag is not None else [],
            "box_tag": f(world.box_tag) if world.box_tag is not None else [],
            "cap_tag": f(world.cap_tag) if world.cap_tag is not None else []}


def world_from_dict(d: dict) -> World3D:
    a = lambda k, n: np.array(d.get(k, []), dtype=np.float64).reshape(-1, n)
    return World3D(a("cyl", 5), a("sph", 4), a("box", 7), float(d["half_size"]), d.get("kind", "?"),
                   meta=d.get("meta", {}), cyl_tag=np.array(d.get("cyl_tag", [])),
                   sph_tag=np.array(d.get("sph_tag", [])), box_tag=np.array(d.get("box_tag", [])),
                   cap=a("cap", 7), cap_tag=np.array(d.get("cap_tag", [])))


def export(world: World3D, name: str, start, goal, out_dir: Path = PROJECT_ROOT / "worlds", decorate=None) -> Path:
    """Write worlds/<name>.sdf and worlds/<name>.json (with the trial's start and goal).
    `decorate(tree)` may add extra models (e.g. moving obstacles) before the SDF is written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = world_to_sdf(world, name)
    if decorate is not None:
        decorate(tree)
    sdf_path = out_dir / f"{name}.sdf"
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    data = world_to_dict(world)
    data.update({"name": name, "start": list(map(float, start)), "goal": list(map(float, goal))})
    (out_dir / f"{name}.json").write_text(json.dumps(data))
    return sdf_path
