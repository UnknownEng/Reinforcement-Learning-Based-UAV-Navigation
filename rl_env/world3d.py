"""
world3d.py
----------
3-D obstacle worlds for the forest / urban / warehouse environments (M6-M10).

A world is a set of primitives, all axis-aligned in z:
  cyl  (N, 5)  vertical finite cylinders   [cx, cy, r, z0, z1]      trunks, stems, bushes, posts, people
  sph  (M, 4)  spheres                     [cx, cy, cz, r]          crowns, foliage, branches, logs (chains)
  box  (K, 7)  yawed boxes                 [cx, cy, cz, hx, hy, hz, yaw]  buildings, walls, shelves, crates
  cap  (L, 7)  capsules (segment + radius) [ax, ay, az, bx, by, bz, r]     branches, fallen logs

The same world drives the fast simulator (vectorized ray casting / distance
queries below) and the Gazebo SITL world (rl_env.world_sdf exports it), so the
policy is tested on exactly the geometry it was trained for.

Generators:
  generate_forest      realistic stand: lognormal trunk sizes, hard-core spacing at a
                       real stem density, saplings and dead branches at flight height,
                       undergrowth and fallen logs near the ground
  generate_urban       street canyon: building blocks, poles, parked vehicles
  generate_warehouse   shelving aisles, pillars, stacked crates
  generate_poles       the legacy M4/M5 style infinitely-tall pole field (for reference)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from rl_env import raycast_jit as _jit

BIG = 1e9


@dataclass
class World3D:
    cyl: np.ndarray
    sph: np.ndarray
    box: np.ndarray
    half_size: float                   # world spans [-half_size, half_size]^2
    kind: str = "forest"
    meta: Dict = field(default_factory=dict)
    # semantic tag per primitive (for rendering / SDF export)
    cyl_tag: Optional[np.ndarray] = None
    sph_tag: Optional[np.ndarray] = None
    box_tag: Optional[np.ndarray] = None
    cap: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    cap_tag: Optional[np.ndarray] = None

    @staticmethod
    def empty(half_size: float, kind: str = "empty") -> "World3D":
        return World3D(np.zeros((0, 5)), np.zeros((0, 4)), np.zeros((0, 7)), half_size, kind,
                       cyl_tag=np.zeros(0, dtype="<U8"), sph_tag=np.zeros(0, dtype="<U8"),
                       box_tag=np.zeros(0, dtype="<U8"), cap=np.zeros((0, 7)), cap_tag=np.zeros(0, dtype="<U8"))

    def counts(self) -> Dict[str, int]:
        return {"cylinders": len(self.cyl), "spheres": len(self.sph), "boxes": len(self.box),
                "capsules": len(self.cap)}

    def with_extra(self, cyl=None, sph=None) -> "World3D":
        """Copy with extra (e.g. moving) primitives appended; tags omitted."""
        c = self.cyl if cyl is None or len(cyl) == 0 else np.vstack([self.cyl, cyl])
        s = self.sph if sph is None or len(sph) == 0 else np.vstack([self.sph, sph])
        return World3D(c, s, self.box, self.half_size, self.kind, cap=self.cap)

    # ------------------------------------------------------------- queries
    def local(self, xy: np.ndarray, radius: float, z_window: Optional[Tuple[float, float]] = None) -> "World3D":
        """Primitives within `radius` horizontally of xy (and overlapping z_window, if given)."""
        c, sp, b, k = self.cyl, self.sph, self.box, self.cap
        c = c[np.hypot(c[:, 0] - xy[0], c[:, 1] - xy[1]) - c[:, 2] < radius]
        sp = sp[np.hypot(sp[:, 0] - xy[0], sp[:, 1] - xy[1]) - sp[:, 3] < radius]
        b = b[np.hypot(b[:, 0] - xy[0], b[:, 1] - xy[1]) - np.hypot(b[:, 3], b[:, 4]) < radius]
        if len(k):
            mx, my = (k[:, 0] + k[:, 3]) / 2, (k[:, 1] + k[:, 4]) / 2
            half_len = 0.5 * np.hypot(k[:, 3] - k[:, 0], k[:, 4] - k[:, 1])
            k = k[np.hypot(mx - xy[0], my - xy[1]) - half_len - k[:, 6] < radius]
        if z_window is not None:
            lo, hi = z_window
            c = c[(c[:, 4] >= lo) & (c[:, 3] <= hi)]
            sp = sp[(sp[:, 2] + sp[:, 3] >= lo) & (sp[:, 2] - sp[:, 3] <= hi)]
            b = b[(b[:, 2] + b[:, 5] >= lo) & (b[:, 2] - b[:, 5] <= hi)]
            if len(k):
                k = k[(np.maximum(k[:, 2], k[:, 5]) + k[:, 6] >= lo) & (np.minimum(k[:, 2], k[:, 5]) - k[:, 6] <= hi)]
        return World3D(c, sp, b, self.half_size, self.kind, cap=k)

    def _arrays(self):
        f = lambda a, n: np.ascontiguousarray(a, dtype=np.float64).reshape(-1, n)
        return f(self.cyl, 5), f(self.sph, 4), f(self.box, 7), f(self.cap, 7)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point (P, 3) to the nearest obstacle surface (negative inside)."""
        if _jit.HAVE_NUMBA:
            return _jit.distance(np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64), *self._arrays())
        return self.distance_numpy(points)

    def distance_numpy(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points).astype(np.float64)
        d = np.full(len(p), BIG)
        if len(self.cyl):
            c = self.cyl
            dxy = np.hypot(p[:, None, 0] - c[None, :, 0], p[:, None, 1] - c[None, :, 1]) - c[None, :, 2]
            dz = np.maximum(c[None, :, 3] - p[:, None, 2], p[:, None, 2] - c[None, :, 4])
            outside = np.hypot(np.maximum(dxy, 0.0), np.maximum(dz, 0.0))
            inside = np.minimum(np.maximum(dxy, dz), 0.0)
            d = np.minimum(d, (outside + inside).min(axis=1))
        if len(self.sph):
            s = self.sph
            ds = np.linalg.norm(p[:, None, :] - s[None, :, :3], axis=2) - s[None, :, 3]
            d = np.minimum(d, ds.min(axis=1))
        if len(self.box):
            b = self.box
            cy, sy = np.cos(b[:, 6]), np.sin(b[:, 6])
            rx, ry = p[:, None, 0] - b[None, :, 0], p[:, None, 1] - b[None, :, 1]
            lx, ly = cy * rx + sy * ry, -sy * rx + cy * ry
            lz = p[:, None, 2] - b[None, :, 2]
            q = np.stack([np.abs(lx) - b[None, :, 3], np.abs(ly) - b[None, :, 4], np.abs(lz) - b[None, :, 5]], axis=2)
            db = np.linalg.norm(np.maximum(q, 0.0), axis=2) + np.minimum(q.max(axis=2), 0.0)
            d = np.minimum(d, db.min(axis=1))
        if len(self.cap):
            k = self.cap
            a, ba = k[:, :3], k[:, 3:6] - k[:, :3]
            baba = np.maximum((ba * ba).sum(axis=1), 1e-12)
            pa = p[:, None, :] - a[None, :, :]
            h = np.clip((pa * ba[None]).sum(axis=2) / baba[None], 0.0, 1.0)
            dk = np.linalg.norm(pa - h[:, :, None] * ba[None], axis=2) - k[None, :, 6]
            d = np.minimum(d, dk.min(axis=1))
        return d

    def raycast(self, origin: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
        """Ranges (R,) from one origin along unit directions (R, 3); max_range where nothing is hit."""
        if _jit.HAVE_NUMBA:
            return _jit.raycast_binned(np.ascontiguousarray(origin, dtype=np.float64),
                                       np.ascontiguousarray(dirs, dtype=np.float64), *self._arrays(),
                                       float(max_range), 90)
        return self.raycast_numpy(origin, dirs, max_range)

    def raycast_numpy(self, origin: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(dirs, dtype=np.float64)
        t = np.full(len(d), max_range)
        if len(self.cyl):
            t = np.minimum(t, _ray_cylinders(o, d, self.cyl, max_range))
        if len(self.sph):
            t = np.minimum(t, _ray_spheres(o, d, self.sph, max_range))
        if len(self.box):
            t = np.minimum(t, _ray_boxes(o, d, self.box, max_range))
        if len(self.cap):
            t = np.minimum(t, _ray_capsules(o, d, self.cap, max_range))
        return t


def _ray_spheres(o, d, s, max_range):
    oc = o[None, :] - s[:, :3]                                  # (M, 3)
    b = d @ oc.T                                                # (R, M)
    c = (oc * oc).sum(axis=1) - s[:, 3] ** 2                    # (M,)
    disc = b * b - c[None, :]
    sq = np.sqrt(np.maximum(disc, 0.0))
    t1 = -b - sq
    t2 = -b + sq
    t = np.where(t1 > 0, t1, np.where(t2 > 0, 0.0, max_range))   # origin inside -> 0
    t = np.where(disc >= 0, t, max_range)
    return t.min(axis=1)


def _ray_capsules(o, d, k, max_range):
    """Ray vs capsule (segment a-b, radius r), vectorized."""
    a, b, r = k[:, :3], k[:, 3:6], k[:, 6]
    ba = b - a
    oa = o[None, :] - a
    baba = (ba * ba).sum(axis=1)
    baoa = (ba * oa).sum(axis=1)
    oaoa = (oa * oa).sum(axis=1)
    bard = d @ ba.T
    rdoa = d @ oa.T
    A = baba[None] - bard * bard
    B = baba[None] * rdoa - baoa[None] * bard
    C = baba[None] * oaoa[None] - baoa[None] ** 2 - (r * r * baba)[None]
    H = B * B - A * C
    A_safe = np.where(np.abs(A) < 1e-12, 1e-12, A)
    t_body = (-B - np.sqrt(np.maximum(H, 0.0))) / A_safe
    y = baoa[None] + t_body * bard
    body_ok = (H >= 0) & (np.abs(A) >= 1e-12) & (y > 0) & (y < baba[None]) & (t_body > 0)
    best = np.where(body_ok, t_body, max_range)
    for end in (a, b):
        oc = o[None, :] - end
        bb = d @ oc.T
        cc = (oc * oc).sum(axis=1) - r * r
        hh = bb * bb - cc[None]
        tt = -bb - np.sqrt(np.maximum(hh, 0.0))
        ok = (hh >= 0) & (tt > 0)
        best = np.where(ok & (tt < best), tt, best)
    seg = np.clip(baoa / np.maximum(baba, 1e-12), 0.0, 1.0)
    inside = np.linalg.norm(oa - seg[:, None] * ba, axis=1) < r
    best = np.where(inside[None, :], 0.0, best)
    return best.min(axis=1)


def _ray_cylinders(o, d, c, max_range):
    dx, dy, dz = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    ox, oy = o[0] - c[None, :, 0], o[1] - c[None, :, 1]
    r, z0, z1 = c[None, :, 2], c[None, :, 3], c[None, :, 4]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - r * r
    a_safe = np.where(a < 1e-12, 1e-12, a)
    disc = b * b - 4.0 * a * cc
    sq = np.sqrt(np.maximum(disc, 0.0))
    best = np.full(np.broadcast(dx, ox).shape, max_range)
    for sign in (-1.0, 1.0):
        tt = (-b + sign * sq) / (2.0 * a_safe)
        z = o[2] + tt * dz
        ok = (disc >= 0) & (a >= 1e-12) & (tt > 0) & (z >= z0) & (z <= z1)
        best = np.where(ok & (tt < best), tt, best)
    dz_safe = np.where(np.abs(dz) < 1e-12, 1e-12, dz)
    for zc in (z0, z1):                                          # end caps
        tt = (zc - o[2]) / dz_safe
        hx, hy = ox + tt * dx, oy + tt * dy
        ok = (np.abs(dz) >= 1e-12) & (tt > 0) & (hx * hx + hy * hy <= r * r)
        best = np.where(ok & (tt < best), tt, best)
    inside = (cc[0] <= 0) & (o[2] >= c[:, 3]) & (o[2] <= c[:, 4])
    best = np.where(inside[None, :], 0.0, best)
    return best.min(axis=1)


def _ray_boxes(o, d, b, max_range):
    cy, sy = np.cos(b[:, 6]), np.sin(b[:, 6])
    rx, ry, rz = o[0] - b[:, 0], o[1] - b[:, 1], o[2] - b[:, 2]
    olx, oly = cy * rx + sy * ry, -sy * rx + cy * ry           # (K,)
    dlx = d[:, 0:1] * cy[None, :] + d[:, 1:2] * sy[None, :]     # (R, K)
    dly = -d[:, 0:1] * sy[None, :] + d[:, 1:2] * cy[None, :]
    dlz = np.broadcast_to(d[:, 2:3], dlx.shape)
    tmin = np.full(dlx.shape, -np.inf)
    tmax = np.full(dlx.shape, np.inf)
    for ol, dl, h in ((olx, dlx, b[:, 3]), (oly, dly, b[:, 4]), (rz, dlz, b[:, 5])):
        inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        t1 = (-h[None, :] - ol[None, :]) * inv
        t2 = (h[None, :] - ol[None, :]) * inv
        tmin = np.maximum(tmin, np.minimum(t1, t2))
        tmax = np.minimum(tmax, np.maximum(t1, t2))
    hit = tmax >= np.maximum(tmin, 0.0)
    t = np.where(tmin > 0, tmin, 0.0)                           # origin inside -> 0
    return np.where(hit, t, max_range).min(axis=1)


# ---------------------------------------------------------------- generators
FOREST_PRESETS = {
    # stems/ha (trunks >= ~8 cm diameter), saplings fraction, undergrowth per m^2, logs per m^2,
    # fraction of mature trees with dead low branches
    "open": dict(stems_per_ha=350, sapling_frac=0.15, bush_density=0.02, log_density=0.002, branch_frac=0.25),
    "medium": dict(stems_per_ha=700, sapling_frac=0.25, bush_density=0.04, log_density=0.003, branch_frac=0.35),
    "dense": dict(stems_per_ha=1200, sapling_frac=0.35, bush_density=0.06, log_density=0.004, branch_frac=0.45),
    "very_dense": dict(stems_per_ha=1800, sapling_frac=0.40, bush_density=0.08, log_density=0.005, branch_frac=0.5),
}


def _hard_core_points(rng, n, half, min_dist, keep_out, max_tries=40):
    """Matern-style hard-core point process by dart throwing; keep_out = list of (x, y, radius)."""
    pts = []
    grid = {}
    cell = max(min_dist, 0.5)
    tries = 0
    while len(pts) < n and tries < n * max_tries:
        tries += 1
        x, y = rng.uniform(-half, half, size=2)
        if any((x - kx) ** 2 + (y - ky) ** 2 < kr ** 2 for kx, ky, kr in keep_out):
            continue
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        ok = True
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for (px, py) in grid.get((ix, iy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < min_dist ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            pts.append((x, y))
            grid.setdefault((gx, gy), []).append((x, y))
    return np.array(pts).reshape(-1, 2)


def generate_forest(rng: np.random.Generator, half_size: float = 20.0, preset: str = "medium",
                    keep_out=(), overrides: Optional[Dict] = None) -> World3D:
    """
    Procedural forest stand. Geometry in metres:
      mature trunks : radius lognormal (median 0.13 m, i.e. ~26 cm DBH), 12-24 m tall,
                      crowns start at 35-60 % of height (above the flight band)
      saplings      : radius 0.04-0.08 m, 2-6 m tall, leafy crown blobs from 1-5 m (in the flight band)
      dead branches : 1-3 per affected trunk at 1.5-5 m, 0.6-1.6 m long, 4-8 cm thick
      undergrowth   : bushes 0.3-1.4 m tall, 0.4-1.0 m radius
      fallen logs   : 3-8 m long, 0.12-0.30 m radius, lying on the ground
    """
    p = dict(FOREST_PRESETS[preset])
    p.update(overrides or {})
    area_ha = (2 * half_size) ** 2 / 1e4
    n_stems = int(rng.poisson(p["stems_per_ha"] * area_ha))
    n_saplings = int(n_stems * p["sapling_frac"])
    n_mature = n_stems - n_saplings
    keep = list(keep_out)

    cyl, cyl_tag, sph, sph_tag, cap, cap_tag = [], [], [], [], [], []
    trunks = _hard_core_points(rng, n_mature, half_size, min_dist=1.4, keep_out=keep)
    for (x, y) in trunks:
        r = float(np.clip(rng.lognormal(math.log(0.13), 0.40), 0.05, 0.45))
        h = float(rng.uniform(12.0, 24.0))
        cyl.append((x, y, r, 0.0, h)); cyl_tag.append("trunk")
        cbh = h * rng.uniform(0.35, 0.60)
        for _ in range(3):  # crown (well above the flight band)
            sph.append((x + rng.normal(0, 0.6), y + rng.normal(0, 0.6), cbh + rng.uniform(1.0, 4.0),
                        rng.uniform(1.5, 3.0))); sph_tag.append("crown")
        if rng.random() < p["branch_frac"]:
            for _ in range(int(rng.integers(1, 4))):
                z = rng.uniform(1.5, 5.0)
                ang = rng.uniform(-math.pi, math.pi)
                length = rng.uniform(0.6, 1.6)
                br = rng.uniform(0.04, 0.08)
                droop = rng.uniform(-0.3, 0.1)
                ax_, ay_ = x + r * math.cos(ang), y + r * math.sin(ang)
                cap.append((ax_, ay_, z, ax_ + length * math.cos(ang), ay_ + length * math.sin(ang),
                            z + droop * length, br)); cap_tag.append("branch")
    all_pts = trunks.tolist()
    saplings = _hard_core_points(rng, n_saplings, half_size, min_dist=0.8,
                                 keep_out=keep + [(x, y, 0.8) for x, y in all_pts])
    for (x, y) in saplings:
        h = float(rng.uniform(2.0, 6.0))
        cyl.append((x, y, float(rng.uniform(0.04, 0.08)), 0.0, h)); cyl_tag.append("sapling")
        for _ in range(int(rng.integers(1, 4))):
            sph.append((x + rng.normal(0, 0.25), y + rng.normal(0, 0.25), rng.uniform(max(1.0, h * 0.4), h),
                        rng.uniform(0.35, 0.9))); sph_tag.append("foliage")
    area = (2 * half_size) ** 2
    for _ in range(int(rng.poisson(p["bush_density"] * area))):
        x, y = rng.uniform(-half_size, half_size, size=2)
        if any((x - kx) ** 2 + (y - ky) ** 2 < (kr + 1.0) ** 2 for kx, ky, kr in keep):
            continue
        r = rng.uniform(0.4, 1.0)
        top = rng.uniform(0.3, 1.4)
        sph.append((x, y, max(top - r, r * 0.4), r)); sph_tag.append("bush")
    for _ in range(int(rng.poisson(p["log_density"] * area))):
        x, y = rng.uniform(-half_size, half_size, size=2)
        if any((x - kx) ** 2 + (y - ky) ** 2 < (kr + 4.0) ** 2 for kx, ky, kr in keep):
            continue
        ang = rng.uniform(0, math.pi)
        length = rng.uniform(3.0, 8.0)
        lr = rng.uniform(0.12, 0.30)
        dx, dy = 0.5 * length * math.cos(ang), 0.5 * length * math.sin(ang)
        cap.append((x - dx, y - dy, lr, x + dx, y + dy, lr, lr)); cap_tag.append("log")
    w = World3D(np.array(cyl, dtype=np.float64).reshape(-1, 5), np.array(sph, dtype=np.float64).reshape(-1, 4),
                np.zeros((0, 7)), half_size, "forest", cyl_tag=np.array(cyl_tag), sph_tag=np.array(sph_tag),
                box_tag=np.zeros(0, dtype="<U8"), cap=np.array(cap, dtype=np.float64).reshape(-1, 7),
                cap_tag=np.array(cap_tag))
    w.meta = {"preset": preset, "stems_per_ha": p["stems_per_ha"], "n_mature": len(trunks),
              "n_saplings": len(saplings), **w.counts()}
    return w


def generate_urban(rng: np.random.Generator, half_size: float = 20.0, keep_out=()) -> World3D:
    """Street canyon: two rows of building blocks along a street, lamp posts, parked vehicles, a kiosk."""
    box, tag, cyl, ctag = [], [], [], []
    street_w = rng.uniform(7.0, 12.0)
    yaw = rng.uniform(-math.pi, math.pi)
    c, s = math.cos(yaw), math.sin(yaw)
    def place(u, v):  # street frame -> world
        return u * c - v * s, u * s + v * c
    for side in (-1, 1):
        u = -half_size
        while u < half_size:
            length = rng.uniform(5.0, 14.0)
            depth = rng.uniform(6.0, 12.0)
            height = rng.uniform(6.0, 25.0)
            v = side * (street_w / 2 + depth / 2)
            x, y = place(u + length / 2, v)
            box.append((x, y, height / 2, length / 2, depth / 2, height / 2, yaw)); tag.append("building")
            u += length + rng.uniform(1.5, 5.0)                 # alleys between blocks
    for u in np.arange(-half_size, half_size, rng.uniform(8.0, 12.0)):
        for side in (-1, 1):
            x, y = place(u + rng.uniform(-1, 1), side * (street_w / 2 - 0.6))
            cyl.append((x, y, 0.12, 0.0, rng.uniform(4.0, 7.0))); ctag.append("pole")
    for _ in range(int(rng.integers(3, 9))):
        u = rng.uniform(-half_size, half_size)
        side = rng.choice([-1, 1])
        x, y = place(u, side * (street_w / 2 - 1.6))
        box.append((x, y, 0.75, 2.2, 0.9, 0.75, yaw)); tag.append("vehicle")
    for _ in range(int(rng.integers(0, 3))):
        x, y = place(rng.uniform(-half_size, half_size), rng.uniform(-street_w / 4, street_w / 4))
        box.append((x, y, 1.4, 1.0, 1.0, 1.4, yaw + rng.uniform(-0.3, 0.3))); tag.append("kiosk")
    w = World3D(np.array(cyl).reshape(-1, 5), np.zeros((0, 4)), np.array(box).reshape(-1, 7), half_size, "urban",
                cyl_tag=np.array(ctag), sph_tag=np.zeros(0, dtype="<U8"), box_tag=np.array(tag))
    w = _remove_keep_out(w, keep_out)
    w.meta = {"street_width": street_w, "street_yaw": yaw, **w.counts()}
    return w


def generate_warehouse(rng: np.random.Generator, half_size: float = 20.0, keep_out=()) -> World3D:
    """Racking aisles with cross-aisles, structural pillars and stacked crates."""
    box, tag, cyl, ctag = [], [], [], []
    aisle = rng.uniform(2.5, 4.0)
    rack_depth = rng.uniform(1.0, 1.4)
    rack_h = rng.uniform(4.0, 8.0)
    yaw = rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.05, 0.05)
    c, s = math.cos(yaw), math.sin(yaw)
    pitch = aisle + rack_depth
    v = -half_size + pitch / 2
    while v < half_size:
        u = -half_size
        while u < half_size:
            length = rng.uniform(8.0, 16.0)
            x, y = u + length / 2, v
            box.append((x * c - y * s, x * s + y * c, rack_h / 2, length / 2, rack_depth / 2, rack_h / 2, yaw))
            tag.append("rack")
            u += length + rng.uniform(2.5, 4.0)                 # cross-aisle
        v += pitch
    for x in np.arange(-half_size + 4, half_size, 10.0):
        for y in np.arange(-half_size + 4, half_size, 10.0):
            cyl.append((x + rng.uniform(-1, 1), y + rng.uniform(-1, 1), 0.25, 0.0, 10.0)); ctag.append("pillar")
    for _ in range(int(rng.integers(6, 16))):
        x, y = rng.uniform(-half_size, half_size, size=2)
        hz = rng.uniform(0.4, 1.2)
        box.append((x, y, hz, rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.8), hz, rng.uniform(0, math.pi)))
        tag.append("crate")
    w = World3D(np.array(cyl).reshape(-1, 5), np.zeros((0, 4)), np.array(box).reshape(-1, 7), half_size,
                "warehouse", cyl_tag=np.array(ctag), sph_tag=np.zeros(0, dtype="<U8"), box_tag=np.array(tag))
    w = _remove_keep_out(w, keep_out)
    w.meta = {"aisle_width": aisle, "rack_height": rack_h, **w.counts()}
    return w


def generate_poles(rng: np.random.Generator, half_size: float = 12.0, n: int = 8, keep_out=()) -> World3D:
    """Legacy M4/M5 style world: tall thin poles scattered over an open square."""
    pts = _hard_core_points(rng, n, half_size - 1.0, min_dist=2.0, keep_out=list(keep_out))
    cyl = [(x, y, rng.uniform(0.2, 0.5), 0.0, 30.0) for x, y in pts]
    return World3D(np.array(cyl).reshape(-1, 5), np.zeros((0, 4)), np.zeros((0, 7)), half_size, "poles",
                   cyl_tag=np.array(["pole"] * len(cyl)), sph_tag=np.zeros(0, dtype="<U8"),
                   box_tag=np.zeros(0, dtype="<U8"))


def _remove_keep_out(w: World3D, keep_out) -> World3D:
    if not keep_out:
        return w
    pts = np.array([(x, y, 0.0) for x, y, _ in keep_out])
    radii = np.array([r for _, _, r in keep_out])
    keep_c = np.ones(len(w.cyl), bool)
    keep_b = np.ones(len(w.box), bool)
    for i in range(len(w.cyl)):
        keep_c[i] = np.all(np.hypot(*(pts[:, :2] - w.cyl[i, :2]).T) - w.cyl[i, 2] > radii)
    for i in range(len(w.box)):
        sub = World3D(np.zeros((0, 5)), np.zeros((0, 4)), w.box[i:i + 1], w.half_size)
        for (x, y, _), r in zip(pts, radii):
            ground_d = sub.distance(np.array([[x, y, 1.0], [x, y, 3.0]])).min()
            if ground_d < r:
                keep_b[i] = False
    w.cyl, w.cyl_tag = w.cyl[keep_c], w.cyl_tag[keep_c]
    w.box, w.box_tag = w.box[keep_b], w.box_tag[keep_b]
    return w


# --------------------------------------------------------- route feasibility
def free_space_connected(world: World3D, start_xy, goal_xy, altitudes=(1.5, 2.25, 3.0, 3.75),
                         inflate: float = 0.45, res: float = 0.25, margin: float = 8.0) -> Tuple[bool, float]:
    """
    2.5-D route feasibility. Free space is rasterized at several altitudes (cells farther than
    `inflate` from every obstacle) inside the start-goal bounding box grown by `margin`; moving
    between adjacent layers is allowed where both are free. Returns (start and goal connected,
    free fraction of the window).
    """
    half = world.half_size
    lo = np.maximum(np.minimum(start_xy, goal_xy) - margin, -half)
    hi = np.minimum(np.maximum(start_xy, goal_xy) + margin, half)
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))
    sub = world.local(((lo + hi) / 2.0), float(np.hypot(*(hi - lo)) / 2.0 + 1.0))
    arrays = sub._arrays()
    layers = []
    for z in altitudes:
        if _jit.HAVE_NUMBA:
            occ = _jit.rasterize_occupancy(*arrays, float(z), float(inflate), float(lo[0]), float(lo[1]),
                                           float(res), nx, ny)
        else:
            xs = lo[0] + (np.arange(nx) + 0.5) * res
            ys = lo[1] + (np.arange(ny) + 0.5) * res
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
            occ = (sub.distance_numpy(pts) <= inflate).reshape(nx, ny)
        layers.append(~occ)
    free = np.stack(layers, axis=2)
    structure = np.zeros((3, 3, 3), bool)
    structure[:, :, 1] = True
    structure[1, 1, :] = True
    lab, _ = ndimage.label(free, structure=structure)

    def cell(xy):
        return (int(np.clip((xy[0] - lo[0]) / res, 0, nx - 1)), int(np.clip((xy[1] - lo[1]) / res, 0, ny - 1)))
    si, sj = cell(start_xy)
    gi, gj = cell(goal_xy)
    s_lab = set(lab[si, sj, :][lab[si, sj, :] > 0].tolist())
    g_lab = set(lab[gi, gj, :][lab[gi, gj, :] > 0].tolist())
    return bool(s_lab & g_lab), float(free.mean())
