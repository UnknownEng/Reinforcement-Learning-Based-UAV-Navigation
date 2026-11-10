"""
camera_recorder.py
------------------
Saves the recording-only Gazebo cameras of x500_lidar3d_cam (onboard FPV and third-person chase view)
as JPEG frames named by sim time, from its own process so image handling never competes with the
vehicle's control loop for the Python interpreter.

    python -m uav_sitl.camera_recorder --out logs/recordings/<run> [--topics /uav/camera/fpv /uav/camera/chase]

Frames go to <out>/<camera>/<sim_time_ms>.jpg; stops when killed.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

# gz.msgs.PixelFormatType values used by gz-sensors cameras
_RGB_INT8, _BGR_INT8 = 3, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--topics", nargs="+", default=["/uav/camera/fpv", "/uav/camera/chase"])
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()
    out = Path(args.out)
    lock = threading.Lock()
    counts = {}

    def make_cb(topic):
        folder = out / topic.rstrip("/").split("/")[-1]
        folder.mkdir(parents=True, exist_ok=True)

        def cb(msg: Image):
            if not msg.width or not msg.height:
                return
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if img.size < msg.width * msg.height * 3:
                return
            img = img[: msg.height * msg.step].reshape(msg.height, msg.step)[:, : msg.width * 3]
            img = img.reshape(msg.height, msg.width, 3)
            if msg.pixel_format_type != _BGR_INT8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            stamp_ms = int(round((msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9) * 1000))
            cv2.imwrite(str(folder / f"{stamp_ms:09d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            with lock:
                counts[topic] = counts.get(topic, 0) + 1
        return cb

    node = Node()
    for t in args.topics:
        node.subscribe(Image, t, make_cb(t))
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
