"""DDS messages of Unitree's interface for Modbus TCP ("FTP") Inspire hands, without inspire_sdkpy.

`--ee inspire_ftp` talks to the hand driver on rt/inspire_hand/{ctrl,state}/{l,r} with the types
of module `inspire` (hand_idl/inspire_hand_{ctrl,state}.idl of Unitree's inspire_hand_ws). The
official Python package for them, inspire_sdkpy, also imports a Qt GUI and a Modbus client when
it is loaded (PyQt5, pyqtgraph, pymodbus), none of which teleoperation uses and which are not
available everywhere (no PyQt5 wheel for arm64). This module holds only what the hand controller
needs: the two message types, written from those IDL files (same type names, field order and
annotations, so both sides see each other), and the default command of
inspire_sdkpy.inspire_hand_defaut.get_inspire_hand_ctrl().

robot_hand_inspire.py uses inspire_sdkpy when it is installed and this module otherwise.
"""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_ctrl(idl.IdlStruct, typename="inspire.inspire_hand_ctrl"):
    pos_set: types.sequence[types.int16, 6]
    angle_set: types.sequence[types.int16, 6]
    force_set: types.sequence[types.int16, 6]
    speed_set: types.sequence[types.int16, 6]
    mode: types.int8


@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_state(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act: types.sequence[types.int16, 6]
    angle_act: types.sequence[types.int16, 6]
    force_act: types.sequence[types.int16, 6]
    current: types.sequence[types.int16, 6]
    err: types.sequence[types.uint8, 6]
    status: types.sequence[types.uint8, 6]
    temperature: types.sequence[types.uint8, 6]


def get_inspire_hand_ctrl():
    """An empty command: every set-point 0 and no mode bit set (nothing is written to the hand)."""
    return inspire_hand_ctrl(pos_set=[0] * 6, angle_set=[0] * 6, force_set=[0] * 6, speed_set=[0] * 6, mode=0b0000)
