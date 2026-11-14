"""One headset for several robots: share the operator's motion over the network.

Normally each teleop_hand_and_arm.py serves its own headset page (TeleVuerWrapper) and reads the
operator's wrists, hands and controllers from it. To let several robots copy one operator:

  source     `python tele_share.py --input-mode hand` on one machine every robot can reach (a PC).
             It serves the headset page (port 8012) and publishes each new TeleData on a ZeroMQ
             PUB socket (default tcp://*:8013). It controls no robot.
  followers  `teleop_hand_and_arm.py --tele-source tcp://<source>:8013 ...` on every robot. A
             RemoteTeleSource replaces the TeleVuerWrapper: same get_tele_data(), fed from the
             network. Each robot then solves its own IK for its own arms from the same wrist poses.

Messages are JSON (Python float repr round-trips exactly), so a pose that did not change on the
source is bit-identical on the followers too. That keeps the tracking-loss rule working end to
end: when the headset stops sending, or the source or the network goes away, a follower's wrist
poses stop changing and it holds its robot (TrackingWatch, XR-101). Nothing is unpickled from
the network.
"""
import argparse
import json
import os
import threading
import time
from dataclasses import fields

import numpy as np

DEFAULT_PORT = 8013
PUBLISH_HZ = 60.0


def encode(tele_data, seq):
    """TeleData -> bytes. Arrays as nested lists, everything else as is."""
    out = {"seq": seq, "t": time.time()}
    for f in fields(tele_data):
        value = getattr(tele_data, f.name)
        out[f.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return json.dumps(out).encode()


def decode(payload, tele_data_class):
    """bytes -> (TeleData, seq, source time). Unknown fields are ignored, missing ones keep their default."""
    raw = json.loads(payload)
    known = {f.name for f in fields(tele_data_class)}
    values = {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    return tele_data_class(**values), raw.get("seq"), raw.get("t")


class RemoteTeleSource:
    """Stands in for TeleVuerWrapper on a follower: the operator's motion comes from a tele_share source."""

    def __init__(self, address):
        from televuer.tv_wrapper import TeleData
        self._tele_data_class = TeleData
        self.address = address
        self._lock = threading.Lock()
        # Until the first message: nothing tracked (motion_data_ready False), like a headset not yet connected.
        self._data = TeleData(head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4))
        self._running = True
        self.received = 0
        self._thread = threading.Thread(target=self._receive, name="tele_source", daemon=True)
        self._thread.start()

    def _receive(self):
        # The socket lives in this thread only (ZeroMQ sockets must not be shared between threads).
        import zmq
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)                  # only the newest message matters
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)
        try:
            while self._running:
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue                                # nothing new: the last data stays (and goes stale)
                except zmq.ZMQError:
                    break
                try:
                    data, _seq, _t = decode(payload, self._tele_data_class)
                except (ValueError, TypeError):
                    continue
                with self._lock:
                    self._data = data
                    self.received += 1
        finally:
            socket.close()

    def get_tele_data(self):
        with self._lock:
            return self._data

    def render_to_xr(self, img):
        pass                                                # the headset page is served by the source

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)                      # the thread closes its socket itself

def main():
    parser = argparse.ArgumentParser(description="Serve the headset page and share the operator's motion with several robots.")
    parser.add_argument("--input-mode", choices=["hand", "controller"], default="hand", help="what the headset tracks; every follower must use the same")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port the motion is published on")
    parser.add_argument("--bind", default="0.0.0.0", help="address to publish on (default: all)")
    args = parser.parse_args()

    import zmq
    from televuer import TeleVuerWrapper

    # Pass-through only: there is no single robot view to show. Certificates as teleop_hand_and_arm.py finds them.
    wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", binocular=False, img_shape=(480, 640),
                              display_mode="pass-through", zmq=False, webrtc=False, arm_reference_mode="head_yaw")
    socket = zmq.Context.instance().socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(f"tcp://{args.bind}:{args.port}")
    print(json.dumps({"ready": True, "port": args.port, "input_mode": args.input_mode}), flush=True)

    period = 1.0 / PUBLISH_HZ
    seq, last_report = 0, time.monotonic()
    last_pose, changed_at = None, None
    try:
        while True:
            start = time.monotonic()
            data = wrapper.get_tele_data()
            seq += 1
            socket.send(encode(data, seq))
            # Status for the operator UI: is the headset sending fresh poses? (same rule as TrackingWatch)
            pose = np.asarray(data.left_wrist_pose, dtype=float).tobytes() + np.asarray(data.right_wrist_pose, dtype=float).tobytes()
            if data.motion_data_ready and pose != last_pose:
                last_pose, changed_at = pose, start
            if start - last_report >= 0.5:
                last_report = start
                age = None if changed_at is None else int((start - changed_at) * 1000)
                print(json.dumps({"connected": bool(data.motion_data_ready), "age_ms": age, "seq": seq}), flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            wrapper.close()
        except Exception:
            pass
        socket.close(linger=0)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
