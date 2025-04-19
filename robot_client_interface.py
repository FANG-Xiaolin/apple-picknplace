#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : robot_client_interface.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""
Minimal client interface for Franka robot in real world.
"""
import zmq
import zlib
import pickle
import numpy as np
from typing import Optional, Union

# Deoxys EE to gripper_camera
EE2CAM = np.array([
    [0.01019998, -0.99989995, 0.01290367, 0.03649885],
    [0.9999, 0.0103, 0.0057, -0.034889],
    [-0.00580004, 0.01280367, 0.99989995, -0.04260014],
    [0.0, 0.0, 0.0, 1.0],
])

class FrankaRealworldController:
    def __init__(self, robot_ip):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(robot_ip)
        self.socket = socket
        self.camera_intrinsics = None
        self.image_dim = None
        self.capture_rs = None

    def capture_image(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "capture_realsense"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))

        return (
            message["rgb"],
            message["depth"],
            message["intrinsics"],
        )  # dep in m (not mm, no need to /1000)

    def get_gripper_camera_extrinsics(self):
        robot_state_cur = self.get_current_joint_states()
        extrinsic = robot_state_cur["ee_pose"].dot(EE2CAM)
        return extrinsic

    def execute_cartesian_impedance_path(self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3):
        """
        End-effector poses in world frame.
        """
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "execute_posesmat4_osc",
            "ee_poses": poses,
            "gripper_isclose": gripper_isclose,
            "use_smoothing": True,
            "speed_factor": speed_factor,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_joint_impedance_path(self, poses, gripper_isclose: Optional[Union[np.ndarray, bool]] = None, speed_factor=3):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "execute_joint_impedance_path",
            "joint_confs": poses,
            "gripper_isclose": gripper_isclose.astype(bool) if isinstance(gripper_isclose, np.ndarray) else gripper_isclose,
            "speed_factor": speed_factor,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def open_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "open_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def close_gripper(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "close_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def get_current_joint_confs(self):
        return self.get_current_joint_states()["qpos"]

    def get_current_joint_states(self):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "get_joint_states"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def go_to_home(self, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "go_to_home", "gripper_open": gripper_open})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]

    def free_motion(self, gripper_open=False, timeout=3.0):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "free_motion_control",
            "gripper_open": gripper_open,
            "timeout": timeout,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]

    def reset_joint_to(self, qpos, gripper_open=False):
        self.socket.send(zlib.compress(pickle.dumps({
            "message_name": "reset_joint_to",
            "gripper_open": gripper_open,
            "qpos": qpos,
        })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["success"]


def initialize_robot_interface(robot_ip):
    robot_interface = FrankaRealworldController(robot_ip=robot_ip)
    return robot_interface
