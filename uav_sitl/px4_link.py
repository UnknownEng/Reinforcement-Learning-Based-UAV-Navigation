"""
px4_link.py
-----------
Minimal, synchronous-friendly MAVLink client for PX4 SITL.

A background thread reads telemetry and streams the current velocity setpoint
at 50 Hz (PX4 needs a continuous stream to stay in OFFBOARD). Everything the
caller sees is in the project's conventions: world ENU, body FLU, velocity
commands in the heading frame (MAV_FRAME_BODY_NED, i.e. yaw-only rotation).
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pymavlink import mavutil

PX4_MAIN_MODE_AUTO = 4
PX4_MAIN_MODE_OFFBOARD = 6
PX4_SUB_MODE_AUTO_LAND = 6
PX4_SUB_MODE_AUTO_LOITER = 3
VELOCITY_YAWRATE_MASK = 0x05C7  # use vx, vy, vz, yaw_rate


@dataclass
class Telemetry:
    t: float = 0.0                       # host monotonic time of last update
    boot_ms: int = 0
    pos_enu: tuple = (0.0, 0.0, 0.0)
    vel_enu: tuple = (0.0, 0.0, 0.0)
    roll: float = 0.0                    # FLU, rad (right side down positive)
    pitch: float = 0.0                   # FLU, rad (nose down positive)
    yaw: float = 0.0                     # ENU, rad (CCW from east)
    rates_flu: tuple = (0.0, 0.0, 0.0)
    quat_enu_flu: tuple = (1.0, 0.0, 0.0, 0.0)
    armed: bool = False
    main_mode: int = 0
    sub_mode: int = 0
    landed_state: int = 0                # MAV_LANDED_STATE (1 on ground, 2 in air, ...)
    have_position: bool = False
    status_texts: List[str] = field(default_factory=list)


def ned_to_enu(x, y, z):
    return (y, x, -z)


def frd_euler_to_flu(roll, pitch, yaw_ned):
    """PX4 ATTITUDE (FRD body, NED world) -> project convention (FLU body, ENU world)."""
    yaw_enu = math.atan2(math.sin(math.pi / 2.0 - yaw_ned), math.cos(math.pi / 2.0 - yaw_ned))
    return roll, -pitch, yaw_enu


def euler_flu_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return q if q[0] >= 0 else tuple(-c for c in q)


class PX4Link:
    def __init__(self, url: str = "udpin:0.0.0.0:14540", log: bool = False):
        self.url = url
        self.mav = None
        self.tel = Telemetry()
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0, 0.0, 0.0)          # vx, vy, vz (heading-frame FLU), yaw_rate (FLU)
        self.streaming = False
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.log_enabled = log
        self.samples: List[dict] = []

    # ------------------------------------------------------------------ setup
    def connect(self, timeout: float = 60.0) -> None:
        self.mav = mavutil.mavlink_connection(self.url, source_system=245, source_component=191)
        hb = self.mav.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError(f"No PX4 heartbeat on {self.url} within {timeout}s")
        for msg_id, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 50),
                           (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50),
                           (mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 5)):
            self.command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
        self._stop.clear()
        for target in (self._reader, self._streamer, self._heartbeat):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)

    def close(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        if self.mav is not None:
            self.mav.close()

    def command(self, cmd_id, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0) -> None:
        self.mav.mav.command_long_send(self.mav.target_system, self.mav.target_component,
                                       cmd_id, 0, p1, p2, p3, p4, p5, p6, p7)

    # ---------------------------------------------------------------- threads
    def _reader(self) -> None:
        while not self._stop.is_set():
            msg = self.mav.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue
            mtype = msg.get_type()
            now = time.monotonic()
            with self.lock:
                tel = self.tel
                if mtype == "LOCAL_POSITION_NED":
                    tel.pos_enu = ned_to_enu(msg.x, msg.y, msg.z)
                    tel.vel_enu = ned_to_enu(msg.vx, msg.vy, msg.vz)
                    tel.boot_ms = msg.time_boot_ms
                    tel.have_position = True
                    tel.t = now
                    if self.log_enabled:
                        self.samples.append({"t": now, "boot_ms": msg.time_boot_ms, "pos": tel.pos_enu,
                                             "vel": tel.vel_enu, "roll": tel.roll, "pitch": tel.pitch,
                                             "yaw": tel.yaw, "cmd": self.cmd})
                elif mtype == "ATTITUDE":
                    tel.roll, tel.pitch, tel.yaw = frd_euler_to_flu(msg.roll, msg.pitch, msg.yaw)
                    tel.rates_flu = (msg.rollspeed, -msg.pitchspeed, -msg.yawspeed)
                    tel.quat_enu_flu = euler_flu_to_quat(tel.roll, tel.pitch, tel.yaw)
                elif mtype == "HEARTBEAT" and msg.get_srcSystem() == self.mav.target_system \
                        and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4:
                    tel.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    tel.main_mode = (msg.custom_mode >> 16) & 0xFF
                    tel.sub_mode = (msg.custom_mode >> 24) & 0xFF
                elif mtype == "EXTENDED_SYS_STATE":
                    tel.landed_state = msg.landed_state
                elif mtype == "STATUSTEXT":
                    tel.status_texts.append(msg.text)
                    del tel.status_texts[:-50]

    def _streamer(self) -> None:
        while not self._stop.is_set():
            if self.streaming:
                vx, vy, vz, yr = self.cmd
                # Heading-frame FLU -> MAV_FRAME_BODY_NED (FRD, yaw-only rotation in PX4)
                self.mav.mav.set_position_target_local_ned_send(
                    int(time.monotonic() * 1000) & 0xFFFFFFFF, self.mav.target_system,
                    self.mav.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, VELOCITY_YAWRATE_MASK,
                    0.0, 0.0, 0.0, vx, -vy, -vz, 0.0, 0.0, 0.0, 0.0, -yr)
            time.sleep(0.02)

    def _heartbeat(self) -> None:
        while not self._stop.is_set():
            self.mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            time.sleep(0.5)

    # ------------------------------------------------------------- telemetry
    def snapshot(self) -> Telemetry:
        with self.lock:
            t = self.tel
            return Telemetry(t.t, t.boot_ms, t.pos_enu, t.vel_enu, t.roll, t.pitch, t.yaw, t.rates_flu,
                             t.quat_enu_flu, t.armed, t.main_mode, t.sub_mode, t.landed_state,
                             t.have_position, list(t.status_texts))

    def wait_for(self, predicate, timeout: float, period: float = 0.05) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate(self.snapshot()):
                return True
            time.sleep(period)
        return False

    # --------------------------------------------------------------- actions
    def set_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0) -> None:
        """Heading-frame FLU velocity (m/s) and yaw rate (rad/s, CCW positive)."""
        self.cmd = (float(vx), float(vy), float(vz), float(yaw_rate))

    def start_offboard(self, timeout: float = 10.0) -> bool:
        self.set_velocity()
        self.streaming = True
        time.sleep(0.5)  # PX4 requires setpoints before accepting OFFBOARD
        for _ in range(int(timeout / 0.5)):
            self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                         mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_MAIN_MODE_OFFBOARD)
            if self.wait_for(lambda t: t.main_mode == PX4_MAIN_MODE_OFFBOARD, 0.5):
                return True
        return False

    def arm(self, timeout: float = 15.0) -> bool:
        for _ in range(int(timeout / 1.0)):
            self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
            if self.wait_for(lambda t: t.armed, 1.0):
                return True
        return False

    def offboard_takeoff(self, altitude: float, climb_rate: float = 1.0, timeout: float = 40.0) -> bool:
        """Arm in OFFBOARD and climb with velocity setpoints to `altitude` (m above home)."""
        if not self.wait_for(lambda t: t.have_position, 30.0):
            raise TimeoutError("No LOCAL_POSITION_NED from PX4 (EKF not ready?)")
        if not self.start_offboard():
            raise RuntimeError("PX4 refused OFFBOARD: " + " | ".join(self.snapshot().status_texts[-5:]))
        if not self.arm():
            raise RuntimeError("PX4 refused to arm: " + " | ".join(self.snapshot().status_texts[-5:]))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            z = self.snapshot().pos_enu[2]
            if z >= altitude - 0.1:
                self.set_velocity()
                return True
            self.set_velocity(vz=min(climb_rate, max(0.3, altitude - z)))
            time.sleep(0.05)
        self.set_velocity()
        return False

    def land(self, timeout: float = 40.0) -> bool:
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                     PX4_MAIN_MODE_AUTO, PX4_SUB_MODE_AUTO_LAND)
        ok = self.wait_for(lambda t: not t.armed, timeout)
        self.streaming = False
        return ok

    def disarm(self, force: bool = False) -> None:
        self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0 if force else 0.0)
