"""
sim_launcher.py
---------------
Start / stop PX4 SITL + Gazebo (scripts/run_px4_sitl.sh) as a process group so
each trial can run on a fresh simulator. Must run inside WSL with the project
environment sourced (scripts/wsl_exec.sh does this).
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimLauncher:
    def __init__(self, world: str = "default", model: str = "x500_lidar", autostart: int = 4012,
                 headless: bool = True, log_path: Optional[Path] = None, world_sdf: Optional[str] = None,
                 extra_env: Optional[dict] = None):
        self.world, self.model, self.autostart = world, model, autostart
        self.headless, self.world_sdf = headless, world_sdf
        self.log_path = Path(log_path) if log_path else PROJECT_ROOT / "logs" / f"sitl_{world}.log"
        self.extra_env = extra_env or {}
        self.proc: Optional[subprocess.Popen] = None
        self._log = None

    def start(self) -> None:
        self.stop_stale()
        px4_dir = Path(os.environ.get("PX4_DIR", Path.home() / "projects/PX4-Autopilot"))
        rootfs = px4_dir / "build/px4_sitl_default/rootfs"
        # Fresh parameters / mission storage each run, so trials are independent.
        for stale in list(rootfs.glob("parameters*.bson")) + [rootfs / "dataman"]:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        env = dict(os.environ)
        env.update({"PX4_GZ_WORLD": self.world, "PX4_SIM_MODEL": self.model,
                    "PX4_SYS_AUTOSTART": str(self.autostart), "HEADLESS": "1" if self.headless else "0"})
        if self.world_sdf:
            env["PX4_GZ_WORLD_SDF"] = str(self.world_sdf)
        env.update(self.extra_env)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(["bash", str(PROJECT_ROOT / "scripts/run_px4_sitl.sh")], env=env,
                                     stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, cwd=str(PROJECT_ROOT))

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Wait until PX4 reports it is ready for takeoff (EKF converged)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                return False
            try:
                if "Ready for takeoff" in self.log_path.read_text(errors="ignore"):
                    return True
            except FileNotFoundError:
                pass
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None
        self.stop_stale()

    @staticmethod
    def stop_stale() -> None:
        subprocess.run(["bash", str(PROJECT_ROOT / "scripts/wsl_procs.sh"), "stop"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
