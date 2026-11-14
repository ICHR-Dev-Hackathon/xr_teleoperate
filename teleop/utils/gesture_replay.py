"""Replay of arm gestures stored in the navigation stack's file format.

Format and playback rules: testbedCIIRC/unitree_nav_stack (scripts/gesture_library.py and
gesture_server.py, commit b378a64). One JSON file per gesture, 14 arm joint angles of the
29-DoF G1 in the order of G1_29_JointArmIndex (motors 15..28):

  continuous  {"kind": "continuous", "rate": 50, "samples": [[14], ...], "crop": [s, e], "speed": 1.0}
  keyframe    {"kind": "keyframe", "poses": [{"arm_pose": [14], "move_time", "hold_time"}, ...]}

GesturePlayer produces the commanded joint angles tick by tick, like gesture_server.py:

  continuous  a far-away start is approached slowly (APPROACH_VEL), then the recording is
              followed with a joint speed limit (max_vel); the replay clock waits while the
              command lags behind (Controller.follow()).
  keyframe    each pose is reached by linear interpolation in max(move_time, distance /
              KEYFRAME_MAX_VEL), then held for hold_time (Controller.play_steps()).
              Difference: the move to the FIRST pose is limited to APPROACH_VEL, because here
              it starts from wherever teleoperation left the arms, not from the neutral pose.

The gesture's own gains (arm_kp, arm_kd) and "return_neutral" are not used: the teleoperation
arm controller keeps its gains, and the arms stay in the last pose until the operator resumes.

Other robots: a file may carry "robot": "<arm type>" (the --arm value it was recorded with,
e.g. "H2"); a file without it is a navigation-stack gesture for the G1 ("G1_29"). A gesture is
only valid on the robot type it was recorded on (other geometry, other joint limits), so
check_robot() refuses a mismatch, and the angles are clipped to the limits of the robot at hand.
"""
import json

import numpy as np

NUM_ARM = 14
DEFAULT_ROBOT = "G1_29"       # files without "robot": the navigation stack's gestures
GESTURE_ARMS = ("G1_29", "H2")    # --arm values whose arms are these 14 joints (motors 15..28) and were checked
APPROACH_VEL = 0.4            # rad/s, navigation stack setting "approach_vel" (default)
MAX_VEL = 2.0                 # rad/s, navigation stack setting "max_vel" (default)
MAX_VEL_RANGE = (0.2, 4.0)    # its allowed range
SPEED_RANGE = (0.1, 3.0)
KEYFRAME_MAX_VEL = 3.0        # rad/s, its MAX_JOINT_VEL cap on keyframe moves
SYNC_TOL = 0.04               # rad: the replay is "synced" when every joint is this close
FAR_TOL = 0.5                 # rad: further away -> slow approach again
LAG_TOL = 0.12                # rad: the replay clock waits while the command lags more than this
MAX_SAMPLES = 50 * 180
# Joint limits of ARM_JOINTS in the navigation stack's gestures.py (rad).
ARM_LIMITS = np.array([
    (-3.08, 2.67), (-1.58, 2.25), (-2.61, 2.61), (-1.04, 2.09), (-1.97, 1.97), (-1.57, 1.57), (-1.57, 1.57),
    (-3.08, 2.67), (-2.25, 1.58), (-2.61, 2.61), (-1.04, 2.09), (-1.97, 1.97), (-1.57, 1.57), (-1.57, 1.57),
])


class GestureFileError(ValueError):
    pass


def _poses(values, what, limits):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise GestureFileError(f"{what}: not numbers") from None
    if arr.ndim != 2 or arr.shape[1] != NUM_ARM or not np.all(np.isfinite(arr)):
        raise GestureFileError(f"{what}: every pose needs {NUM_ARM} finite joint angles")
    return np.clip(arr, limits[:, 0], limits[:, 1])


def check_robot(gesture, arm):
    """Raise unless the gesture was recorded on this arm type (the --arm value)."""
    if arm not in GESTURE_ARMS:
        raise GestureFileError(f"gesture replay is not available for {arm} arms")
    if gesture["robot"] != arm:
        raise GestureFileError(f"this gesture was recorded on {gesture['robot']} arms, not {arm}")


def load_gesture(path, limits=None):
    """Read and check a gesture file. Joint angles are clipped to `limits` ((14, 2) lower/upper,
    the joint limits of the robot at hand); by default to the G1 limits of the navigation stack."""
    limits = ARM_LIMITS if limits is None else np.asarray(limits, dtype=float).reshape(NUM_ARM, 2)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise GestureFileError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise GestureFileError("not a gesture file")
    kind = data.get("kind")
    robot = str(data.get("robot") or DEFAULT_ROBOT)
    if kind == "continuous":
        samples = _poses(data.get("samples") or [[]], "samples", limits)
        if not 2 <= len(samples) <= MAX_SAMPLES:
            raise GestureFileError("a continuous gesture needs 2 .. 9000 samples")
        rate = float(data.get("rate") or 0)
        if not 1.0 <= rate <= 1000.0:
            raise GestureFileError("invalid sample rate")
        total = (len(samples) - 1) / rate
        crop = data.get("crop") if isinstance(data.get("crop"), list) and len(data.get("crop")) == 2 else [0.0, total]
        start = min(max(float(crop[0]), 0.0), total)
        end = min(max(float(crop[1]), start), total)
        i0, i1 = int(round(start * rate)), int(round(end * rate))
        if i1 - i0 < 1:
            raise GestureFileError("the cropped part is empty")
        speed = float(data.get("speed") or 1.0)
        return {"kind": kind, "robot": robot, "samples": samples[i0:i1 + 1], "rate": rate,
                "speed": min(max(speed, SPEED_RANGE[0]), SPEED_RANGE[1])}
    if kind == "keyframe":
        poses = data.get("poses") or []
        if not poses or not all(isinstance(p, dict) for p in poses):
            raise GestureFileError("a keyframe gesture needs poses")
        arm = _poses([p.get("arm_pose") for p in poses], "poses", limits)
        steps = []
        for p, q in zip(poses, arm):
            move, hold = float(p.get("move_time", 1.0)), float(p.get("hold_time", 0.5))
            if not (0.0 <= move <= 60.0 and 0.0 <= hold <= 60.0):
                raise GestureFileError("move and hold times must be 0 .. 60 s")
            steps.append({"arm_pose": q, "move_time": move, "hold_time": hold})
        return {"kind": kind, "robot": robot, "steps": steps}
    raise GestureFileError(f"unknown gesture kind {kind!r}")


class GesturePlayer:
    """Commanded arm joint angles for one replay; call step(dt) every control tick."""

    def __init__(self, gesture, q_now, speed=None, max_vel=MAX_VEL):
        self.kind = gesture["kind"]
        self.cmd = np.array(q_now, dtype=float).reshape(NUM_ARM)
        self.max_vel = float(min(max(max_vel or MAX_VEL, MAX_VEL_RANGE[0]), MAX_VEL_RANGE[1]))
        self.done = False
        self.phase = "approach"            # "approach" -> "play" -> "done"
        if self.kind == "continuous":
            self.samples, self.rate = gesture["samples"], gesture["rate"]
            self.speed = float(min(max(speed or gesture["speed"], SPEED_RANGE[0]), SPEED_RANGE[1]))
            self.end = (len(self.samples) - 1) / self.rate
            self.t = 0.0
            self.tracking = False
            self.waiting = True
        else:
            self.steps = gesture["steps"]
            self.index = -1
            self._next_pose()

    # -- continuous: gesture_server.py Controller.follow()
    def _ref(self):
        f = min(max(self.t * self.rate, 0.0), len(self.samples) - 1.0)
        i = min(int(f), len(self.samples) - 2)
        u = f - i
        return self.samples[i] * (1.0 - u) + self.samples[i + 1] * u

    def _step_continuous(self, dt):
        if not self.waiting:
            self.t = min(self.end, self.t + dt * self.speed)
        ref = self._ref()
        err = ref - self.cmd
        m = float(np.max(np.abs(err)))
        if self.tracking and m > FAR_TOL:
            self.tracking = False
        elif not self.tracking and m < SYNC_TOL:
            self.tracking = True
        lim = (self.max_vel if self.tracking else APPROACH_VEL) * dt
        if m > lim:
            self.cmd = self.cmd + err * (lim / m)
            m -= lim
        else:
            self.cmd = ref.copy()
            m = 0.0
        self.waiting = not self.tracking or m > LAG_TOL
        self.phase = "play" if self.tracking else "approach"
        if self.t >= self.end and m < SYNC_TOL:
            self.done = True

    # -- keyframe: gesture_server.py Controller.play_steps()
    def _next_pose(self):
        self.index += 1
        if self.index >= len(self.steps):
            self.done = True
            return
        step = self.steps[self.index]
        self._from = self.cmd.copy()
        delta = float(np.max(np.abs(step["arm_pose"] - self._from)))
        limit = APPROACH_VEL if self.index == 0 else KEYFRAME_MAX_VEL
        self._move = max(step["move_time"], delta / limit, 1e-3)
        self._elapsed = 0.0
        self._holding = False

    def _step_keyframe(self, dt):
        step = self.steps[self.index]
        self._elapsed += dt
        if not self._holding:
            frac = min(1.0, self._elapsed / self._move)
            self.cmd = self._from + (step["arm_pose"] - self._from) * frac
            self.phase = "approach" if self.index == 0 else "play"
            if frac >= 1.0:
                self._holding, self._elapsed = True, 0.0
        elif self._elapsed >= step["hold_time"]:
            self._next_pose()
        else:
            self.phase = "play"

    def step(self, dt):
        """Advance by dt seconds; returns the commanded arm joint angles (14)."""
        if not self.done:
            if self.kind == "continuous":
                self._step_continuous(dt)
            else:
                self._step_keyframe(dt)
        if self.done:
            self.phase = "done"
        return self.cmd.copy()

    def info(self):
        """Progress for the operator UI."""
        if self.kind == "continuous":
            return {"phase": self.phase, "t": round(self.t / self.speed, 2), "duration": round(self.end / self.speed, 2)}
        return {"phase": self.phase, "t": min(self.index + 1, len(self.steps)), "duration": len(self.steps)}
