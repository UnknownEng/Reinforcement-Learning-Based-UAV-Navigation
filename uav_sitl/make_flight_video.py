"""
make_flight_video.py
--------------------
Turns a recorded SITL flight (uav_sitl/run_trials.py --record DIR) into a 1280x720 video with four
synchronised panels, one frame per policy step (10 Hz sim time):

  chase camera  |  onboard FPV camera            (Gazebo renders; recording only)
  top-down map  |  LiDAR range image + status     (what the policy actually used)

The map shows the true obstacles near the route, the current 7-channel LiDAR returns projected into
the world, the moving hazards (same deterministic kinematics the SITL daemon uses), the path flown
and the goal. Also writes a few still frames for figures.

    python -m uav_sitl.make_flight_video logs/recordings/forest_movers_8309 --out logs/recordings/forest_movers_8309.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from rl_env.lidar3d import DEFAULT_SPEC  # noqa: E402
from rl_env.observation import quat_to_rotation_matrix  # noqa: E402
from uav_sitl.movers import SitlMovers  # noqa: E402
from uav_sitl.world_sdf import world_from_dict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
TAG_STYLE = {"trunk": ("#5c4026", 1.0), "sapling": ("#6b4d2e", 1.0), "bush": ("#1f5a1f", 0.85),
             "foliage": ("#3a8a2e", 0.55), "crown": ("#2a6b22", 0.15), "branch": ("#6b4a2a", 1.0),
             "log": ("#4f3a22", 1.0), "building": ("#9e9ea3", 1.0), "rack": ("#d9731a", 1.0),
             "crate": ("#b8945f", 1.0), "pillar": ("#8c8c94", 1.0), "pole": ("#59595e", 1.0),
             "vehicle": ("#334da6", 1.0), "kiosk": ("#b38c33", 1.0)}


def nearest_frame(folder: Path, stamps_ms: np.ndarray, files, t_ms: float):
    if not len(stamps_ms):
        return None
    i = int(np.argmin(np.abs(stamps_ms - t_ms)))
    if abs(stamps_ms[i] - t_ms) > 250:
        return None
    return cv2.imread(str(folder / files[i]))


def frame_index(folder: Path):
    files = sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []
    return files, np.array([int(f.split(".")[0]) for f in files], dtype=np.float64)


def lidar_points_world(ranges: np.ndarray, pos: np.ndarray, quat: np.ndarray, spec=DEFAULT_SPEC):
    R = quat_to_rotation_matrix(np.asarray(quat, dtype=np.float64)).astype(np.float64)
    d = spec.dirs_body().reshape(len(spec.elevations_deg), spec.n_azimuth, 3)
    origin = pos + R @ np.array(spec.mount_offset_body)
    hit = ranges < spec.max_range - 0.05
    pts = origin + (ranges[..., None] * d) @ R.T
    return pts[hit], np.nonzero(hit)[0], origin


def draw_world(ax, world, view):
    (x0, x1), (y0, y1) = view
    pad = 3.0
    inside = lambda x, y, r=0.0: (x0 - pad - r <= x <= x1 + pad + r) and (y0 - pad - r <= y <= y1 + pad + r)
    for i, (x, y, r, z0, z1) in enumerate(world.cyl):
        tag = str(world.cyl_tag[i]) if len(world.cyl_tag) > i else "trunk"
        if inside(x, y, r):
            col, a = TAG_STYLE.get(tag, ("#777777", 1.0))
            ax.add_patch(Circle((x, y), r, color=col, alpha=a, lw=0, zorder=3))
    for i, (x, y, z, r) in enumerate(world.sph):
        tag = str(world.sph_tag[i]) if len(world.sph_tag) > i else "foliage"
        if inside(x, y, r) and tag != "crown":
            col, a = TAG_STYLE.get(tag, ("#3a8a2e", 0.5))
            ax.add_patch(Circle((x, y), r, color=col, alpha=a, lw=0, zorder=2))
    for i, k in enumerate(world.cap):
        tag = str(world.cap_tag[i]) if len(world.cap_tag) > i else "branch"
        if inside(k[0], k[1]) or inside(k[3], k[4]):
            col, a = TAG_STYLE.get(tag, ("#6b4a2a", 1.0))
            ax.plot([k[0], k[3]], [k[1], k[4]], color=col, alpha=a, lw=max(1.0, 40 * k[6]), zorder=3,
                    solid_capstyle="round")
    for b in getattr(world, "box", np.zeros((0, 7))):
        x, y, _, hx, hy, _, yaw = b
        if inside(x, y, max(hx, hy)):
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy], [hx, hy]]) @ np.array([[c, s], [-s, c]])
            ax.fill(x + corners[:, 0], y + corners[:, 1], color="#999999", alpha=0.9, zorder=3)


def render_panels(fig, axes, world, view, i, d, meta, movers, goal, pts, bands):
    ax_map, ax_rng, ax_txt = axes
    for a in axes:
        a.clear()
    pos = d["truth_pos"][i]
    # --- top-down map
    draw_world(ax_map, world, view)
    path = d["truth_pos"][: i + 1]
    ax_map.plot(path[:, 0], path[:, 1], color="#1f77b4", lw=2.0, zorder=5)
    # LiDAR rows 0-1 look down, 2-4 level, 5-6 up; ground returns are dropped (as the policy's filter does)
    band_col = np.array(["#ff7f0e", "#ff7f0e", "#e377c2", "#e377c2", "#e377c2", "#17becf", "#17becf"])
    keep = pts[:, 2] > 0.15 if len(pts) else np.zeros(0, bool)
    if np.any(keep):
        ax_map.scatter(pts[keep, 0], pts[keep, 1], s=9, c=band_col[bands[keep]], zorder=6, linewidths=0)
    if movers is not None:
        movers.advance_to(float(d["stamp"][i]))
        for kind, p, r in zip(movers.movers.kind, movers.movers.pos, movers.movers.radius):
            ax_map.add_patch(Circle((p[0], p[1]), r, color="#d62728", alpha=0.85, zorder=7))
    R = quat_to_rotation_matrix(d["truth_quat"][i].astype(np.float64))
    fwd = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-6)
    ax_map.add_patch(Circle((pos[0], pos[1]), 0.35, fill=False, ec="k", lw=1.5, zorder=8))
    ax_map.arrow(pos[0], pos[1], 0.9 * fwd[0], 0.9 * fwd[1], width=0.06, color="k", zorder=8)
    ax_map.add_patch(Circle((pos[0], pos[1]), 10.0, fill=False, ec="#888888", ls=":", lw=0.8, zorder=4))
    ax_map.plot(goal[0], goal[1], marker="*", ms=16, color="gold", mec="k", zorder=9)
    ax_map.set_xlim(*view[0]); ax_map.set_ylim(*view[1]); ax_map.set_aspect("equal")
    ax_map.set_facecolor("#e9e4d8")
    ax_map.set_title("top-down: obstacles, LiDAR hits (orange down / pink level / cyan up), hazards red",
                     fontsize=7)
    ax_map.tick_params(labelsize=6)
    # --- LiDAR range image (what the sensor returned this step)
    rng = d["ranges_raw"][i]
    im = ax_rng.imshow(rng[::-1], aspect="auto", cmap="turbo_r", vmin=0.1, vmax=10.0,
                       extent=[-180, 180, -35, 35], interpolation="nearest")
    ax_rng.set_title("7-channel LiDAR range image (m), 90 azimuths x 7 elevations", fontsize=8)
    ax_rng.set_xlabel("azimuth (deg, 0 = nose, CCW +)", fontsize=7); ax_rng.set_ylabel("elev. (deg)", fontsize=7)
    ax_rng.set_yticks([-30, -10, 10, 30]); ax_rng.tick_params(labelsize=6)
    if not hasattr(fig, "_cbar"):
        fig._cbar = fig.colorbar(im, ax=ax_rng, fraction=0.035, pad=0.01)
        fig._cbar.ax.tick_params(labelsize=6)
    # --- status
    t = d["stamp"][i] - meta["t0"]
    v = d["vel"][i]
    dist = float(np.linalg.norm(pos - goal))
    raw, act, cmd = d["raw_action"][i], d["action"][i], d["cmd"][i]
    safe = bool(d["safety_active"][i])
    ax_txt.axis("off")
    lines = [f"t = {t:5.1f} s   speed {np.linalg.norm(v):4.2f} m/s   to goal {dist:5.2f} m",
             f"policy action   vx {raw[0]:+.2f}  vy {raw[1]:+.2f}  vz {raw[2]:+.2f}  yaw {raw[3]:+.2f}",
             f"after safety    vx {act[0]:+.2f}  vy {act[1]:+.2f}  vz {act[2]:+.2f}  yaw {act[3]:+.2f}",
             f"PX4 setpoint    {cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f} m/s, {cmd[3]:+.2f} rad/s"]
    for k, line in enumerate(lines):
        ax_txt.text(0.0, 1.0 - 0.21 * k, line, fontsize=8, family="monospace", va="top", transform=ax_txt.transAxes)
    ax_txt.text(0.0, -0.08, "SAFETY LAYER ACTIVE" if safe else "policy command passed unchanged", fontsize=9,
                color="#c0392b" if safe else "#2e7d32", weight="bold", va="bottom", transform=ax_txt.transAxes)


def fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def label(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 8 + 10, 20), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stills", type=int, default=4, help="also save N evenly spaced composite frames as PNG")
    args = ap.parse_args()
    rdir = Path(args.recording)
    d = dict(np.load(rdir / "flight.npz"))
    meta = json.load(open(rdir / "flight.json"))
    wd = json.load(open(PROJECT_ROOT / "worlds" / f"{meta['world']}.json"))
    world = world_from_dict(wd["world"] if "world" in wd else wd)
    goal = np.array(meta["goal"])
    movers = SitlMovers(meta["mover_seed"], meta["n_movers"]) if meta.get("n_movers") else None
    xy = np.vstack([d["truth_pos"][:, :2], goal[None, :2]])
    lo, hi = xy.min(0) - 4.0, xy.max(0) + 4.0
    span = max(hi - lo)
    c = (lo + hi) / 2
    view = ((c[0] - span * 0.80, c[0] + span * 0.80), (c[1] - span / 2, c[1] + span / 2))
    chase_files, chase_ms = frame_index(rdir / "chase")
    fpv_files, fpv_ms = frame_index(rdir / "fpv")

    fig = plt.figure(figsize=(W / 100, H / 2 / 100), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.6, 1.0], left=0.04, right=0.98,
                          bottom=0.12, top=0.83, wspace=0.12, hspace=0.55)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rng = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), 10.0, (W, H))
        if writer.isOpened():
            break
    n = len(d["stamp"])
    still_idx = set(np.linspace(0, n - 1, args.stills).astype(int).tolist()) if args.stills else set()
    blank = np.zeros((H // 2, W // 2, 3), np.uint8)
    for i in range(n):
        t_ms = d["stamp"][i] * 1000.0
        chase = nearest_frame(rdir / "chase", chase_ms, chase_files, t_ms)
        fpv = nearest_frame(rdir / "fpv", fpv_ms, fpv_files, t_ms)
        chase = label(cv2.resize(chase, (W // 2, H // 2)) if chase is not None else blank.copy(),
                      "chase camera (Gazebo render, recording only)")
        fpv = label(cv2.resize(fpv, (W // 2, H // 2)) if fpv is not None else blank.copy(),
                    "onboard forward camera (recording only)")
        pts, rows, _ = lidar_points_world(d["ranges_raw"][i].astype(np.float64), d["truth_pos"][i].astype(np.float64),
                                          d["truth_quat"][i])
        render_panels(fig, (ax_map, ax_rng, ax_txt), world, view, i, d, meta, movers, goal, pts, rows)
        cbf = meta.get("cbf")
        stack = ("rule shield + moving-obstacle CBF" if cbf == "moving" else "rule shield + CBF" if cbf
                 else "rule shield" if meta.get("shield") else "no safety layer")
        fig.suptitle(f"PX4 SITL  |  {meta['suite']} seed {meta['seed']}  |  learned policy + {stack}  |  "
                     f"outcome: {meta['outcome']}", fontsize=8, y=0.985)
        bottom = cv2.resize(fig_to_bgr(fig), (W, H // 2))
        frame = np.vstack([np.hstack([chase, fpv]), bottom])
        writer.write(frame)
        if i in still_idx:
            cv2.imwrite(str(out.with_name(f"{out.stem}_frame{i:03d}.png")), frame)
    writer.release()
    print(f"wrote {out} ({n} frames, {n / 10:.1f} s at 10 fps)")


if __name__ == "__main__":
    main()
