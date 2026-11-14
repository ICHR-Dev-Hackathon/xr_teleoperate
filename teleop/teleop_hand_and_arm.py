import time
import json
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController, R1_A5_ArmController, R1_A7_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK, R1_A5_ArmIK, R1_A7_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.gesture_replay import GesturePlayer, GestureFileError, load_gesture, check_robot
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper, G1DAgvClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from play_beep import play_beep, play_arpegio  # pribavoj
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
PAUSED         = False  # Hold robot commands while keeping the teleop process alive
PAUSE_TOGGLE   = False  # Request a transition between tracking and paused states
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECORD_DISCARD = False  # Discard current recording and return to READY state pribavoj
HOLD_REQUEST   = False  # Stop from the operator UI: hold the robot where it is (also while recording)
HOLD           = False  # Holding after a stop or lost tracking, until the operator resumes
PROBLEM        = None   # Problem code for the operator UI, e.g. "XR-101" (tracking lost)
XR_CONNECTED   = False  # The headset has sent tracking data at least once
TRACKING_AGE_MS = None  # Time since the tracked hand/controller poses last changed
REPLAY_REQUEST = None   # Operator UI: play this gesture file on the arms ({"path", "speed", "max_vel"})
REPLAY_CANCEL  = False  # Operator UI: end the running replay, keep holding
REPLAY         = None   # Progress of the running gesture replay for the operator UI, else None
REPLAY_HZ      = 100.0  # Replay tick rate (no IK needed, so faster than --frequency)
HOLD_RELEASE   = False  # Operator UI: continue after a stop that came before tracking started
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

CONTROLLER_BUTTON_ACTIONS = {
    # Pico/Quest-style controller labels: left A/B are usually shown as X/Y.
    "left_ctrl_aButton": ("r", "left X"),
    "left_ctrl_bButton": ("s", "left Y"),
    "right_ctrl_aButton": ("q", "right A"),
    "right_ctrl_bButton": ("d", "right B"),  # pribavoj
}

def handle_control_key(key, source="keyboard"):
    global STOP, START, PAUSED, PAUSE_TOGGLE, RECORD_TOGGLE, RECORD_DISCARD, HOLD_REQUEST
    global REPLAY_REQUEST, REPLAY_CANCEL, HOLD_RELEASE
    if key.startswith('replay:'):
        try:
            request = json.loads(key[len('replay:'):])
            REPLAY_REQUEST = {"path": str(request["path"]), "speed": request.get("speed"),
                              "max_vel": request.get("max_vel")}
            logger_mp.info(f"[{source}] gesture replay requested: {REPLAY_REQUEST['path']}")
        except (ValueError, KeyError, TypeError):
            logger_mp.warning(f"[{source}] gesture replay ignored: no gesture file given.")
        return
    if key == 'replay_stop':
        REPLAY_CANCEL = True
        return
    if key == 'hold':
        HOLD_REQUEST = True
        logger_mp.info(f"[{source}] stop requested: holding the robot where it is.")
        return
    if key == 'resume':
        if START and PAUSED:
            PAUSE_TOGGLE = True
            logger_mp.info(f"[{source}] resume requested.")
        elif not START:
            HOLD_RELEASE = True      # a stop before tracking: nothing resumes, the stop is acknowledged
        else:
            logger_mp.info(f"[{source}] resume ignored: teleop is not paused.")
        return
    if key == 'pause_on':
        if START and not PAUSED:
            PAUSE_TOGGLE = True
            logger_mp.info(f"[{source}] pause requested.")
        return
    if key in ('r', 'start'):
        if not START:
            START = True
            logger_mp.info(f"[{source}] start teleop requested.")
        elif key == 'r':
            PAUSE_TOGGLE = True
            logger_mp.info(f"[{source}] {'resume' if PAUSED else 'pause'} teleop requested.")
        else:
            logger_mp.info(f"[{source}] teleop is already started.")
    elif key == 'p' and START:
        PAUSE_TOGGLE = True
        logger_mp.info(f"[{source}] {'resume' if PAUSED else 'pause'} teleop requested.")
    elif key == 'p':
        logger_mp.warning(f"[{source}] pause toggle ignored because teleop has not started.")
    elif key == 'q':
        START = False
        STOP = True
        logger_mp.info(f"[{source}] stop teleop requested.")
    elif key == 's' and START and not PAUSED:
        RECORD_TOGGLE = True
        logger_mp.info(f"[{source}] recording toggle requested.")
    elif key == 's' and PAUSED:
        logger_mp.warning(f"[{source}] recording toggle ignored while teleop is paused.")
    elif key == 's':
        logger_mp.warning(f"[{source}] recording toggle ignored because teleop has not started.")
    elif key == 'd' and RECORD_RUNNING:  # pribavoj
        RECORD_DISCARD = True  # pribavoj
        logger_mp.info(f"[{source}] discard recording requested.")  # pribavoj
    else:
        logger_mp.warning(f"[{source}] {key} was pressed, but no action is defined for this key.")

def on_press(key):
    handle_control_key(key)

class ControllerButtonMapper:
    def __init__(self):
        self.previous_buttons = {button_name: False for button_name in CONTROLLER_BUTTON_ACTIONS}

    def update(self, tele_data):
        for button_name, (key, label) in CONTROLLER_BUTTON_ACTIONS.items():
            is_pressed = bool(getattr(tele_data, button_name, False))
            was_pressed = self.previous_buttons[button_name]
            if is_pressed and not was_pressed:
                handle_control_key(key, source=f"Pico {label}")
            self.previous_buttons[button_name] = is_pressed

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, PAUSED, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "PAUSED": PAUSED,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "HOLD": HOLD,
        "PROBLEM": PROBLEM,
        "XR_CONNECTED": XR_CONNECTED,
        "TRACKING_AGE_MS": TRACKING_AGE_MS,
        "REPLAY": REPLAY,
    }

def clip_unit(value):
    return max(-1.0, min(1.0, float(value)))

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))

def apply_deadband(value, deadband):
    value = clip_unit(value)
    deadband = clamp(deadband, 0.0, 1.0)
    if abs(value) <= deadband:
        return 0.0
    return value

def controller_analog_button_value(is_pressed, value):
    value = clamp(value, 0.0, 1.0)
    if value == 0.0 and is_pressed:
        return 1.0
    return value

def make_pose_clutch(held_target_pose, resume_input_pose):
    """Create offsets that make the current XR pose equal the held robot target."""
    held_target_pose = np.asarray(held_target_pose)
    resume_input_pose = np.asarray(resume_input_pose)
    rotation_offset = held_target_pose[:3, :3] @ resume_input_pose[:3, :3].T
    position_offset = held_target_pose[:3, 3] - resume_input_pose[:3, 3]
    return rotation_offset, position_offset

def apply_pose_clutch(input_pose, clutch):
    """Apply a pause/resume clutch while preserving subsequent relative motion."""
    target_pose = np.array(input_pose, copy=True)
    if clutch is None:
        return target_pose
    rotation_offset, position_offset = clutch
    target_pose[:3, :3] = rotation_offset @ target_pose[:3, :3]
    target_pose[:3, 3] += position_offset
    return target_pose

def controller_motion_is_neutral(tele_data, deadband, include_side_triggers=False):
    """Return True when resuming cannot immediately command base/column motion."""
    axes = [
        *tele_data.left_ctrl_thumbstickValue,
        *tele_data.right_ctrl_thumbstickValue,
    ]
    if include_side_triggers:
        axes.extend([
            controller_analog_button_value(tele_data.left_ctrl_squeeze, tele_data.left_ctrl_squeezeValue),
            controller_analog_button_value(tele_data.right_ctrl_squeeze, tele_data.right_ctrl_squeezeValue),
        ])
    return all(abs(float(value)) <= deadband for value in axes)

def dex1_trigger_position(value):
    """Map the XR trigger value to the Dex1 position used by its controller."""
    return float(np.interp(clamp(value, 0.0, 10.0), [5.0, 7.0], [0.0, 5.4]))

def apply_dex1_soft_takeover(input_value, held_input_value, previous_input_value, tolerance=0.18):
    """Hold a Dex1 command until the physical trigger reaches that command."""
    input_value = float(input_value)
    if held_input_value is None:
        return input_value, None

    held_position = dex1_trigger_position(held_input_value)
    input_position = dex1_trigger_position(input_value)
    reached = abs(input_position - held_position) <= tolerance
    if previous_input_value is not None:
        previous_position = dex1_trigger_position(previous_input_value)
        reached = reached or (
            (previous_position - held_position) * (input_position - held_position) <= 0.0
        )

    if reached:
        return input_value, None
    return float(held_input_value), float(held_input_value)

ARM_VELOCITY_START = 20.0   # rad/s before Start and right after a "match" resume
ARM_VELOCITY_MAX = 30.0     # rad/s, upstream's default arm velocity limit
ARM_VELOCITY_RAMP_S = 5.0

class ArmVelocityRamp:
    """Hold the arm joint velocity limit at 20 rad/s until Start, then raise it to 30 rad/s over 5 s.

    The g1d-thesis branch did this inside the arm controllers (speed_gradual_max). Upstream v1.6
    removed that method and starts at 30 rad/s, so the ramp is kept here on top of upstream's
    set_arm_velocity_limit(), which every arm controller provides.
    """
    def __init__(self, arm_ctrl):
        self.arm_ctrl = arm_ctrl
        self.start_time = None
        self.arm_ctrl.set_arm_velocity_limit(ARM_VELOCITY_START)

    def restart(self, now=None):
        self.start_time = time.time() if now is None else now
        self.arm_ctrl.set_arm_velocity_limit(ARM_VELOCITY_START)

    def update(self, now=None):
        if self.start_time is None:
            return
        now = time.time() if now is None else now
        fraction = min(1.0, max(0.0, (now - self.start_time) / ARM_VELOCITY_RAMP_S))
        self.arm_ctrl.set_arm_velocity_limit(
            ARM_VELOCITY_START + (ARM_VELOCITY_MAX - ARM_VELOCITY_START) * fraction)
        if fraction >= 1.0:
            self.start_time = None

TRACKING_STALE_S = 0.5      # tracked poses frozen this long -> stop (XR-101)

class TrackingWatch:
    """Notice when the headset stops sending fresh hand/controller poses.

    televuer only reports whether tracking data arrived at least once (motion_data_ready).
    While the headset tracks, each wrist (hand or controller) pose changes with every update,
    because real tracking always jitters a little. A pose that stays bit-identical means no
    updates arrive for that side: the headset disconnected, or the hand/controller is lost.
    Stopping then also prevents a frozen stick value from keeping the robot walking.
    """
    def __init__(self):
        self._last = [None, None]
        self._changed = [None, None]

    def age(self, tele_data, now):
        """Seconds since the stalest side last changed, or None before any tracking data."""
        if not tele_data.motion_data_ready:
            return None
        ages = []
        for i, pose in enumerate((tele_data.left_wrist_pose, tele_data.right_wrist_pose)):
            sig = np.asarray(pose, dtype=np.float64).tobytes()
            if sig != self._last[i]:
                self._last[i], self._changed[i] = sig, now
            ages.append(now - self._changed[i])
        return max(ages)

def arm_gravity_torque(arm_ik, q):
    """Feed-forward torque that holds the arms at q against gravity (as solve_ik() returns it)."""
    import pinocchio as pin
    model = arm_ik.reduced_robot.model
    return pin.rnea(model, arm_ik.reduced_robot.data, np.asarray(q, dtype=float), np.zeros(model.nv), np.zeros(model.nv))

def load_gesture_for(arm, arm_ik, path):
    """A gesture file for replay on this robot: recorded on this arm type, clipped to its joint limits."""
    model = arm_ik.reduced_robot.model
    gesture = load_gesture(path, limits=np.column_stack([model.lowerPositionLimit, model.upperPositionLimit]))
    check_robot(gesture, arm)
    return gesture

def frame_bgr(frame):
    """Return a decoded camera image, or None while a stream has no frame."""
    return None if frame is None else getattr(frame, "bgr", None)

def build_recording_colors(camera_config, head_img, left_wrist_img, right_wrist_img):
    """Build a complete set of recording images without using missing frames."""
    camera_frames = (
        ("head", camera_config['head_camera'], head_img),
        ("left wrist", camera_config['left_wrist_camera'], left_wrist_img),
        ("right wrist", camera_config['right_wrist_camera'], right_wrist_img),
    )
    decoded_frames = {
        name: frame_bgr(frame)
        for name, config, frame in camera_frames
        if config['enable_zmq']
    }
    missing_cameras = [name for name, image in decoded_frames.items() if image is None]
    if missing_cameras:
        return {}, missing_cameras

    colors = {}
    if camera_config['head_camera']['enable_zmq']:
        head_bgr = decoded_frames['head']
        if camera_config['head_camera']['binocular']:
            eye_width = camera_config['head_camera']['image_shape'][1] // 2
            colors["color_0"] = head_bgr[:, :eye_width]
            colors["color_1"] = head_bgr[:, eye_width:]
        else:
            colors["color_0"] = head_bgr

    wrist_color_index = 2 if camera_config['head_camera']['binocular'] else 1
    if camera_config['left_wrist_camera']['enable_zmq']:
        colors[f"color_{wrist_color_index}"] = decoded_frames['left wrist']
    if camera_config['right_wrist_camera']['enable_zmq']:
        colors[f"color_{wrist_color_index + 1}"] = decoded_frames['right wrist']
    return colors, []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--show-wrist-cameras', action='store_true', help='Show the left and right wrist WebRTC streams as picture-in-picture panels in XR')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2', 'R1_A5', 'R1_A7'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='Image-server IP used by the laptop for camera config and ZMQ frames')
    parser.add_argument('--webrtc-server-ip', type=str, default=None, help='Optional image-server IP reachable by the XR headset; defaults to --img-server-ip')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--motion-base', type=str, choices=['loco', 'g1d_agv'], default='loco', help='Select motion-control backend')
    parser.add_argument('--motion-no-walk', action='store_true', help='With --motion and controller input: never command the locomotion controller from the controllers (no stick walking, no damping on both sticks pressed). The robot keeps balancing by itself; only the arms and hands follow.')
    parser.add_argument('--motion-max-vx', type=float, default=0.2, help='Max forward velocity from controller stick, m/s')
    parser.add_argument('--motion-max-vy', type=float, default=0.3, help='Max lateral velocity from controller stick, m/s')
    parser.add_argument('--motion-max-vyaw', type=float, default=0.3, help='Max yaw velocity from controller stick, rad/s')
    parser.add_argument('--g1d-column-scale', type=float, default=0.5, help='Max normalized G1-D column velocity from right stick Y')
    parser.add_argument('--g1d-left-stick-x-mode', type=str, choices=['base_yaw', 'torso_yaw'], default='base_yaw', help='For G1-D AGV: map left stick X to base yaw or torso yaw')
    parser.add_argument('--g1d-torso-yaw-input', type=str, choices=['none', 'left_stick_x', 'side_triggers'], default=None, help='For G1-D AGV: optional torso yaw input source')
    parser.add_argument('--torso-yaw-max-rate', type=float, default=0.9, help='Max torso yaw target rate, rad/s')
    parser.add_argument('--torso-yaw-limit', type=float, default=2.0, help='Symmetric torso yaw software limit for joystick control, rad')
    parser.add_argument('--torso-yaw-deadband', type=float, default=0.08, help='Controller deadband for torso yaw input')
    parser.add_argument('--pause-resume-mode', type=str, choices=['clutch', 'match'], default='match', help='On resume, preserve the held arm/gripper pose (clutch) or move toward the current XR pose (match)')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--controller-alignment', action='store_true', help='Use packet-matched G1 simulation target limiting')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--record-torso-yaw', action='store_true', help='Also store measured torso yaw and commanded torso yaw target in body recording data')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    if args.ee == "dex1_internal" and args.motion:
        parser.error("--ee dex1_internal does not currently support --motion.")

    try:
        if args.motion and args.motion_base == "g1d_agv" and args.input_mode != "controller":
            raise ValueError("--motion-base g1d_agv requires --input-mode controller.")
        if args.torso_yaw_max_rate is not None and args.torso_yaw_max_rate < 0.0:
            raise ValueError("--torso-yaw-max-rate must be non-negative.")
        if args.torso_yaw_limit < 0.0:
            raise ValueError("--torso-yaw-limit must be non-negative.")
        if args.torso_yaw_deadband < 0.0:
            raise ValueError("--torso-yaw-deadband must be non-negative.")
        if args.g1d_left_stick_x_mode == "torso_yaw":
            if args.g1d_torso_yaw_input is not None and args.g1d_torso_yaw_input != "left_stick_x":
                raise ValueError("--g1d-left-stick-x-mode=torso_yaw conflicts with --g1d-torso-yaw-input.")
            g1d_torso_yaw_input = "left_stick_x"
        else:
            g1d_torso_yaw_input = args.g1d_torso_yaw_input or "none"
        arm_motion_mode = args.motion and args.motion_base != "g1d_agv"
        webrtc_server_ip = args.webrtc_server_ip or args.img_server_ip

        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        logger_mp.info(
            f"Camera data server: {args.img_server_ip}; "
            f"XR WebRTC server: {webrtc_server_ip}."
        )
        left_wrist_webrtc_url = None
        right_wrist_webrtc_url = None
        left_wrist_aspect = 4.0 / 3.0
        right_wrist_aspect = 4.0 / 3.0
        if args.show_wrist_cameras:
            wrist_configs = {
                "left": camera_config['left_wrist_camera'],
                "right": camera_config['right_wrist_camera'],
            }
            missing_webrtc = [
                side for side, config in wrist_configs.items()
                if not config.get('enable_webrtc', False)
            ]
            if missing_webrtc:
                raise ValueError(
                    "--show-wrist-cameras requires WebRTC for both wrist cameras; "
                    f"disabled for: {', '.join(missing_webrtc)}."
                )

            left_wrist_webrtc_url = (
                f"https://{webrtc_server_ip}:"
                f"{wrist_configs['left']['webrtc_port']}/offer"
            )
            right_wrist_webrtc_url = (
                f"https://{webrtc_server_ip}:"
                f"{wrist_configs['right']['webrtc_port']}/offer"
            )
            left_shape = wrist_configs['left']['image_shape']
            right_shape = wrist_configs['right']['image_shape']
            left_wrist_aspect = left_shape[1] / left_shape[0]
            right_wrist_aspect = right_shape[1] / right_shape[0]
            logger_mp.info(
                "XR wrist-camera panels enabled: "
                f"left={left_wrist_webrtc_url}, right={right_wrist_webrtc_url}."
            )
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{webrtc_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw",
                                     left_wrist_webrtc_url=left_wrist_webrtc_url,
                                     right_wrist_webrtc_url=right_wrist_webrtc_url,
                                     left_wrist_aspect=left_wrist_aspect,
                                     right_wrist_aspect=right_wrist_aspect,
                                     )
        controller_button_mapper = ControllerButtonMapper() if args.input_mode == "controller" else None
        
        # Controllers command the robot's locomotion (sticks walk, both sticks pressed = damping)
        # only in motion mode and unless --motion-no-walk asks for arms only.
        stick_walking = args.motion and args.input_mode == "controller" and not args.motion_no_walk
        body_action_scale = 0.0 if args.motion_no_walk else 0.3     # recorded stick "body action"; nothing is commanded without walking
        if args.motion and args.input_mode == "controller" and args.motion_no_walk:
            logger_mp.info("Motion mode without stick walking: the controllers move the arms only; "
                           "the robot's own controller keeps the balance.")

        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if stick_walking:
                if args.motion_base == "g1d_agv":
                    loco_wrapper = G1DAgvClientWrapper()
                    logger_mp.info("Motion backend: G1-D AGV; arm commands stay on rt/lowcmd.")
                else:
                    loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived

        if args.ee == "dex1_internal":
            if args.arm != "G1_29":
                raise ValueError("dex1_internal is only supported with --arm G1_29.")
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array,
                                                              dual_gripper_action_array, motion_mode=arm_motion_mode, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
            else:
                arm_ctrl = G1_29_ArmController(
                    motion_mode=arm_motion_mode,
                    simulation_mode=args.sim,
                    controller_alignment=args.controller_alignment,
                )
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=arm_motion_mode, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=arm_motion_mode, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=arm_motion_mode, simulation_mode=args.sim)
        elif args.arm == "R1_A5":
            arm_ik = R1_A5_ArmIK()
            arm_ctrl = R1_A5_ArmController(motion_mode=arm_motion_mode, simulation_mode=args.sim)
        elif args.arm == "R1_A7":
            arm_ik = R1_A7_ArmIK()
            arm_ctrl = R1_A7_ArmController(motion_mode=arm_motion_mode, simulation_mode=args.sim)
        # Start slower and ramp up after Start (see ArmVelocityRamp).
        arm_velocity_ramp = ArmVelocityRamp(arm_ctrl)
        if args.record_torso_yaw and not hasattr(arm_ctrl, "get_current_waist_yaw"):
            raise ValueError("--record-torso-yaw is currently supported only with --arm=G1_29.")

        torso_yaw_target = 0.0
        torso_yaw_min = -abs(args.torso_yaw_limit)
        torso_yaw_max = abs(args.torso_yaw_limit)
        torso_yaw_max_rate = args.torso_yaw_max_rate if args.torso_yaw_max_rate is not None else args.motion_max_vyaw
        if args.motion and args.motion_base == "g1d_agv":
            if g1d_torso_yaw_input != "none":
                if not hasattr(arm_ctrl, "ctrl_waist_yaw"):
                    raise ValueError("--g1d-torso-yaw-input is currently supported only with --arm=G1_29.")
                waist_yaw_min, waist_yaw_max = arm_ctrl.get_waist_yaw_limits()
                torso_yaw_center = arm_ctrl.get_current_waist_yaw()
                torso_yaw_min = max(torso_yaw_center - abs(args.torso_yaw_limit), waist_yaw_min)
                torso_yaw_max = min(torso_yaw_center + abs(args.torso_yaw_limit), waist_yaw_max)
                torso_yaw_target = clamp(torso_yaw_center, torso_yaw_min, torso_yaw_max)
                arm_ctrl.ctrl_waist_yaw(torso_yaw_target)
                logger_mp.info(
                    f"G1-D mapping: {g1d_torso_yaw_input} controls torso yaw "
                    f"({torso_yaw_min:.2f} to {torso_yaw_max:.2f} rad from startup yaw {torso_yaw_center:.2f}); "
                    f"{'right' if g1d_torso_yaw_input == 'left_stick_x' else 'left'} stick X controls base yaw."
                )
            else:
                logger_mp.info("G1-D mapping: left stick X controls base yaw.")

        # end-effector
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] or Pico left X to START, PAUSE, or RESUME teleoperation.")
        logger_mp.info(f"Pause resume mode: {args.pause_resume_mode}.")
        if args.record:
            logger_mp.info("🟡  Press [s] or Pico left Y to START or SAVE recording (toggle cycle).")
            logger_mp.info("🟠  Press Pico right B to DISCARD the current recording.")  # pribavoj
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] or Pico right A to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        tracking_watch = TrackingWatch()
        gesture_player = None          # a gesture being replayed on the arms
        replay_last_time = 0.0
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(1.0 / REPLAY_HZ if gesture_player is not None else 0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            tele_data = tv_wrapper.get_tele_data()
            tracking_age = tracking_watch.age(tele_data, time.time())
            XR_CONNECTED = bool(tele_data.motion_data_ready)
            TRACKING_AGE_MS = None if tracking_age is None else int(tracking_age * 1000)
            if controller_button_mapper is not None:
                controller_button_mapper.update(tele_data)
            if HOLD_RELEASE:
                HOLD_RELEASE = False
                if HOLD:
                    HOLD = False
                    logger_mp.info("Stop acknowledged; the arms keep their position.")
            if REPLAY_CANCEL:
                REPLAY_CANCEL = False
                if gesture_player is not None:
                    gesture_player, REPLAY = None, None
                    logger_mp.info("Gesture replay ended by the operator; holding the arms.")
            if REPLAY_REQUEST is not None:
                # A gesture can be replayed before tracking starts: no headset is needed for it.
                request, REPLAY_REQUEST = REPLAY_REQUEST, None
                if HOLD:
                    logger_mp.warning("Gesture replay ignored: the robot is stopped. Continue first.")
                else:
                    try:
                        gesture_player = GesturePlayer(load_gesture_for(args.arm, arm_ik, request["path"]), arm_ctrl.get_current_dual_arm_q(),
                                                       speed=request.get("speed"), max_vel=request.get("max_vel"))
                        replay_last_time = time.time()
                        REPLAY = gesture_player.info()
                        logger_mp.info(f"▶️  Gesture replay started: {request['path']} "
                                       f"(max {gesture_player.max_vel:g} rad/s), before tracking.")
                    except (GestureFileError, TypeError, ValueError) as e:
                        logger_mp.warning(f"Gesture replay ignored: {e}")
            if HOLD_REQUEST:
                # Stop before tracking started: keep the arms where they are now (this also
                # halts the move to the initial pose that starts when the controller connects).
                HOLD_REQUEST = False
                HOLD = True
                if gesture_player is not None:
                    gesture_player, REPLAY = None, None
                    logger_mp.info("Gesture replay ended by the stop.")
                current_q = arm_ctrl.get_current_dual_arm_q()
                arm_ctrl.ctrl_dual_arm(current_q, np.zeros_like(current_q))
                logger_mp.info("Stop before tracking: holding the arms at their current position.")
            if gesture_player is not None:
                now = time.time()
                replay_q = gesture_player.step(min(0.05, max(0.0, now - replay_last_time)))
                replay_last_time = now
                arm_ctrl.ctrl_dual_arm(replay_q, arm_gravity_torque(arm_ik, replay_q))
                REPLAY = gesture_player.info()
                if gesture_player.done:
                    gesture_player, REPLAY = None, None
                    logger_mp.info("⏹️  Gesture replay finished; holding the last pose.")

        if STOP:
            logger_mp.info("Stop requested before tracking started.")
            raise KeyboardInterrupt

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        HOLD = False
        PROBLEM = None
        gesture_player, REPLAY, REPLAY_REQUEST = None, None, None     # tracking takes the arms over
        arm_velocity_ramp.restart()
        last_motion_time = time.time()

        left_wrist_clutch = None
        right_wrist_clutch = None
        last_left_wrist_target = None
        last_right_wrist_target = None
        last_left_gripper_input = None
        last_right_gripper_input = None
        left_gripper_takeover = None
        right_gripper_takeover = None
        previous_left_gripper_input = None
        previous_right_gripper_input = None
        last_missing_camera_warning_time = 0.0

        head_img = None
        left_wrist_img = None
        right_wrist_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            motion_dt = max(0.0, min(0.1, start_time - last_motion_time))
            last_motion_time = start_time
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # get xr's tele data and controller button actions
            tele_data = tv_wrapper.get_tele_data()
            if controller_button_mapper is not None:
                controller_button_mapper.update(tele_data)
            if STOP:
                break
            torso_yaw_input_value = 0.0
            tracking_age = tracking_watch.age(tele_data, start_time)
            XR_CONNECTED = bool(tele_data.motion_data_ready)
            TRACKING_AGE_MS = None if tracking_age is None else int(tracking_age * 1000)
            tracking_lost = (not PAUSED and tracking_age is not None and tracking_age > TRACKING_STALE_S)

            if HOLD_REQUEST or tracking_lost:
                # Stop (operator UI) or lost tracking: hold the robot where it is. Unlike a pause
                # this also works while recording; the unfinished episode is discarded.
                stop_requested = HOLD_REQUEST
                HOLD_REQUEST = False
                if gesture_player is not None:      # a replay ends here; the arms keep the last command
                    gesture_player, REPLAY, REPLAY_REQUEST = None, None, None
                    logger_mp.info("Gesture replay ended by the stop.")
                if args.record and RECORD_RUNNING:
                    RECORD_RUNNING = False
                    recorder.discard_episode()
                    logger_mp.warning("Recording discarded because the robot was stopped.")
                if not PAUSED:
                    PAUSED = True
                    RECORD_TOGGLE = False
                    if stick_walking:
                        loco_wrapper.Stop()
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                HOLD = True
                if stop_requested:
                    logger_mp.info("⏹️  Stop: holding the robot. Resume from the operator UI or with left X.")
                else:
                    PROBLEM = "XR-101"
                    logger_mp.warning(f"⏹️  Tracking lost ({tracking_age:.2f} s without new poses): holding the robot. XR-101")

            if REPLAY_CANCEL:
                REPLAY_CANCEL = False
                if gesture_player is not None:
                    gesture_player, REPLAY = None, None
                    logger_mp.info("Gesture replay ended by the operator; holding the arms.")

            if REPLAY_REQUEST is not None:
                # Play a gesture file on the arms. Teleoperation is paused for it (the robot no longer
                # follows the operator) and stays paused afterwards, until the operator resumes.
                request, REPLAY_REQUEST = REPLAY_REQUEST, None
                if HOLD:
                    logger_mp.warning("Gesture replay ignored: the robot is stopped. Resume first.")
                elif args.record and RECORD_RUNNING:
                    logger_mp.warning("Gesture replay ignored while recording. Save or discard the episode first.")
                else:
                    try:
                        gesture = load_gesture_for(args.arm, arm_ik, request["path"])
                        if not PAUSED:
                            PAUSED = True
                            RECORD_TOGGLE = False
                            if stick_walking:
                                loco_wrapper.Stop()
                            with xr_motion_data_ready.get_lock():
                                xr_motion_data_ready.value = False
                        gesture_player = GesturePlayer(gesture, arm_ctrl.get_current_dual_arm_q(),
                                                       speed=request.get("speed"), max_vel=request.get("max_vel"))
                        replay_last_time = time.time()
                        REPLAY = gesture_player.info()
                        PAUSE_TOGGLE = False
                        logger_mp.info(f"▶️  Gesture replay started: {request['path']} "
                                       f"(max {gesture_player.max_vel:g} rad/s). Teleoperation is paused.")
                    except (GestureFileError, TypeError, ValueError) as e:
                        logger_mp.warning(f"Gesture replay ignored: {e}")

            if PAUSE_TOGGLE and gesture_player is not None:
                # left X / resume during a replay only ends the replay; a second press resumes.
                PAUSE_TOGGLE = False
                gesture_player, REPLAY = None, None
                logger_mp.info("Gesture replay ended; teleoperation stays paused.")

            if PAUSE_TOGGLE:
                PAUSE_TOGGLE = False
                if PAUSED:
                    controls_neutral = (
                        not stick_walking
                        or controller_motion_is_neutral(
                            tele_data,
                            args.torso_yaw_deadband,
                            include_side_triggers=(
                                args.motion_base == "g1d_agv"
                                and g1d_torso_yaw_input == "side_triggers"
                            ),
                        )
                    )
                    if not tele_data.motion_data_ready or tracking_age is None or tracking_age > TRACKING_STALE_S:
                        logger_mp.warning("Cannot resume: XR controller/hand tracking is not ready.")
                    elif not controls_neutral:
                        logger_mp.warning("Cannot resume: center both sticks and release the side triggers first.")
                    else:
                        if args.pause_resume_mode == "clutch":
                            held_left_pose = (
                                last_left_wrist_target
                                if last_left_wrist_target is not None
                                else tele_data.left_wrist_pose
                            )
                            held_right_pose = (
                                last_right_wrist_target
                                if last_right_wrist_target is not None
                                else tele_data.right_wrist_pose
                            )
                            left_wrist_clutch = make_pose_clutch(held_left_pose, tele_data.left_wrist_pose)
                            right_wrist_clutch = make_pose_clutch(held_right_pose, tele_data.right_wrist_pose)
                            if args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                                left_gripper_takeover = last_left_gripper_input
                                right_gripper_takeover = last_right_gripper_input
                                previous_left_gripper_input = tele_data.left_ctrl_triggerValue
                                previous_right_gripper_input = tele_data.right_ctrl_triggerValue
                            resume_message = "controller poses re-clutched"
                        else:
                            left_wrist_clutch = None
                            right_wrist_clutch = None
                            left_gripper_takeover = None
                            right_gripper_takeover = None
                            arm_velocity_ramp.restart()
                            resume_message = "matching the current controller poses"
                        PAUSED = False
                        HOLD = False
                        PROBLEM = None
                        last_motion_time = start_time
                        logger_mp.info(f"🟢 Teleoperation RESUMED; {resume_message}.")
                elif args.record and RECORD_RUNNING:
                    logger_mp.warning("Cannot pause while recording. Save with [s]/left Y or discard with right B first.")
                else:
                    PAUSED = True
                    RECORD_TOGGLE = False
                    if stick_walking:
                        loco_wrapper.Stop()
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                    logger_mp.info("⏸️  Teleoperation PAUSED; robot commands are being held. Press [r]/left X to resume.")

            if PAUSED:
                # Arm control keeps publishing its last target. Disabling XR readiness
                # makes the end-effector controller hold its measured position.
                with xr_motion_data_ready.get_lock():
                    xr_motion_data_ready.value = False
                if args.record:
                    READY = recorder.is_ready()
                if gesture_player is not None:
                    now = time.time()
                    replay_q = gesture_player.step(min(0.05, max(0.0, now - replay_last_time)))
                    replay_last_time = now
                    arm_ctrl.ctrl_dual_arm(replay_q, arm_gravity_torque(arm_ik, replay_q))
                    REPLAY = gesture_player.info()
                    if gesture_player.done:
                        gesture_player, REPLAY = None, None
                        logger_mp.info("⏹️  Gesture replay finished; holding the last pose. Resume to follow the operator again.")
                time_elapsed = time.time() - start_time
                loop_hz = REPLAY_HZ if gesture_player is not None else args.frequency
                sleep_time = max(0, (1 / loop_hz) - time_elapsed)
                time.sleep(sleep_time)
                continue

            if args.record and RECORD_DISCARD:  # pribavoj
                RECORD_DISCARD = False  # pribavoj
                play_beep(volume=0.001, duration=0.5)  # wake up audio, pribavoj
                play_arpegio(arpeggio = [392.00, 261.63])  # pribavoj

                if RECORD_RUNNING:  # pribavoj
                    RECORD_RUNNING = False  # pribavoj
                    recorder.discard_episode()  # pribavoj

                    if args.sim:  # pribavoj
                        publish_reset_category(1, reset_pose_publisher)  # pribavoj

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    _, missing_cameras = build_recording_colors(
                        camera_config,
                        head_img,
                        left_wrist_img,
                        right_wrist_img,
                    )
                    if missing_cameras:
                        logger_mp.error(
                            "Recording not started: no decoded image from "
                            f"{', '.join(missing_cameras)} camera(s). Check teleimager-server and --img-server-ip."
                        )
                    elif recorder.create_episode():
                        RECORD_RUNNING = True
                        play_beep(volume=0.001, duration=0.5)  # wake up audio, pribavoj
                        play_arpegio(arpeggio = [261.63, 329.63, 392.00])  # pribavoj
    
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    play_beep(volume=0.001, duration=0.5)  # wake up audio, pribavoj
                    play_arpegio(arpeggio = [392.00, 329.63, 261.63])  # pribavoj
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                had_left_takeover = left_gripper_takeover is not None
                had_right_takeover = right_gripper_takeover is not None
                left_gripper_input, left_gripper_takeover = apply_dex1_soft_takeover(
                    tele_data.left_ctrl_triggerValue,
                    left_gripper_takeover,
                    previous_left_gripper_input,
                )
                right_gripper_input, right_gripper_takeover = apply_dex1_soft_takeover(
                    tele_data.right_ctrl_triggerValue,
                    right_gripper_takeover,
                    previous_right_gripper_input,
                )
                with left_gripper_value.get_lock():
                    left_gripper_value.value = left_gripper_input
                with right_gripper_value.get_lock():
                    right_gripper_value.value = right_gripper_input
                last_left_gripper_input = left_gripper_input
                last_right_gripper_input = right_gripper_input
                previous_left_gripper_input = tele_data.left_ctrl_triggerValue
                previous_right_gripper_input = tele_data.right_ctrl_triggerValue
                if had_left_takeover and left_gripper_takeover is None:
                    logger_mp.info("Left Dex1 trigger synchronized after resume.")
                if had_right_takeover and right_gripper_takeover is None:
                    logger_mp.info("Right Dex1 trigger synchronized after resume.")
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # high level control
            if stick_walking:
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                if args.motion_base == "g1d_agv":
                    if not (tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick):
                        vx = clip_unit(-tele_data.left_ctrl_thumbstickValue[1]) * args.motion_max_vx
                        if g1d_torso_yaw_input != "none":
                            if g1d_torso_yaw_input == "left_stick_x":
                                torso_yaw_axis = -tele_data.left_ctrl_thumbstickValue[0]
                                base_yaw_axis = -tele_data.right_ctrl_thumbstickValue[0]
                            else:
                                left_side = controller_analog_button_value(tele_data.left_ctrl_squeeze, tele_data.left_ctrl_squeezeValue)
                                right_side = controller_analog_button_value(tele_data.right_ctrl_squeeze, tele_data.right_ctrl_squeezeValue)
                                torso_yaw_axis = left_side - right_side
                                base_yaw_axis = -tele_data.left_ctrl_thumbstickValue[0]
                            torso_yaw_axis = apply_deadband(torso_yaw_axis, args.torso_yaw_deadband)
                            torso_yaw_input_value = torso_yaw_axis
                            torso_yaw_target = clamp(
                                torso_yaw_target + torso_yaw_axis * torso_yaw_max_rate * motion_dt,
                                torso_yaw_min,
                                torso_yaw_max,
                            )
                            arm_ctrl.ctrl_waist_yaw(torso_yaw_target)
                            vyaw = clip_unit(base_yaw_axis) * args.motion_max_vyaw
                        else:
                            vyaw = clip_unit(-tele_data.left_ctrl_thumbstickValue[0]) * args.motion_max_vyaw
                        column_vz = clip_unit(-tele_data.right_ctrl_thumbstickValue[1]) * args.g1d_column_scale
                        loco_wrapper.Move(vx, 0.0, vyaw)
                        loco_wrapper.HeightAdjust(column_vz)
                else:
                    loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * args.motion_max_vx,
                                      -tele_data.left_ctrl_thumbstickValue[0] * args.motion_max_vy,
                                      -tele_data.right_ctrl_thumbstickValue[0] * args.motion_max_vyaw)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            left_wrist_target = apply_pose_clutch(tele_data.left_wrist_pose, left_wrist_clutch)
            right_wrist_target = apply_pose_clutch(tele_data.right_wrist_pose, right_wrist_clutch)
            sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist_target, right_wrist_target, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_velocity_ramp.update()
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            last_left_wrist_target = left_wrist_target.copy()
            last_right_wrist_target = right_wrist_target.copy()

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * body_action_scale,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * body_action_scale,
                                               -tele_data.right_ctrl_thumbstickValue[0] * body_action_scale]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * body_action_scale,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * body_action_scale,
                                               -tele_data.right_ctrl_thumbstickValue[0] * body_action_scale]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action (split into left/right halves by the arm's own DOF, so it works for any variant: H1/G1_23/R1_A5 = 4/5 per arm, G1_29/R1_A7 = 7)
                half = len(current_lr_arm_q) // 2
                left_arm_state,  right_arm_state  = current_lr_arm_q[:half], current_lr_arm_q[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                if RECORD_RUNNING:
                    colors, missing_cameras = build_recording_colors(
                        camera_config,
                        head_img,
                        left_wrist_img,
                        right_wrist_img,
                    )
                    depths = {}
                    if missing_cameras and start_time - last_missing_camera_warning_time >= 1.0:
                        logger_mp.warning(
                            "Skipping recording frames: no decoded image from "
                            f"{', '.join(missing_cameras)} camera(s)."
                        )
                        last_missing_camera_warning_time = start_time
                    body_state = {
                        "qpos": current_body_state,
                    }
                    body_action = {
                        "qpos": current_body_action,
                    }
                    if args.record_torso_yaw:
                        measured_torso_yaw = arm_ctrl.get_current_waist_yaw()
                        body_state["torso_yaw"] = measured_torso_yaw
                        body_action["torso_yaw_target"] = (
                            torso_yaw_target if g1d_torso_yaw_input != "none" else measured_torso_yaw
                        )
                        body_action["torso_yaw_input"] = torso_yaw_input_value
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": body_state, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": body_action, 
                    }
                    if not missing_cameras:
                        if args.sim:
                            sim_state = sim_state_subscriber.read_data()
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                        else:
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        try:
            if args.motion and 'loco_wrapper' in locals() and hasattr(loco_wrapper, "Stop"):
                loco_wrapper.Stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop motion wrapper: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
