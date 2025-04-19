import numpy as np
import os.path as osp

from deoxys import config_root
from deoxys.utils import YamlConfig, transform_utils
from deoxys.franka_interface import FrankaInterface
from functools import lru_cache


@lru_cache(maxsize=1)
def get_global_config():
    return YamlConfig(osp.join(config_root, 'GLOBAL_ENV.yml')).as_easydict()


def get_robot_by_index(robot_index):
    config = get_global_config().robots
    for robot in config:
        if robot.index == robot_index:
            return robot

    raise ValueError(f"Robot with index {robot_index} not found")


def get_franka_interface(robot_index: int = 1, wait_for_state: bool = True, auto_close: bool = True, auto_gripper_reset: bool = False):
    config_file = get_robot_by_index(robot_index).config_file
    robot_interface = FrankaInterface(
        osp.join(config_root, config_file), use_visualizer=False,
        auto_close=auto_close, automatic_gripper_reset=auto_gripper_reset
    )

    if wait_for_state:
        robot_interface.wait_for_state()

    return robot_interface


def get_xyzrpy_action(target_pose_mat4):
    target_pos = target_pose_mat4[:3, 3]
    target_rot = target_pose_mat4[:3, :3]
    target_rot_quat = transform_utils.mat2quat(target_rot)
    target_rot_rpy = transform_utils.quat2axisangle(target_rot_quat)
    pose_action = np.concatenate([target_pos, target_rot_rpy])
    return pose_action


def osc_move_to_target_absolute_pose_controller(robot_interface, controller_cfg, target_pose_mat4, gripper_open=None, gripper_close=None):
    target_pose_xyzrpy = get_xyzrpy_action(target_pose_mat4)
    if gripper_open is not None:
        gripper_cmd = -1. if gripper_open else 1.
    elif gripper_close is not None:
        gripper_cmd = 1. if gripper_close else -1.
    else:
        gripper_cmd = 0.
    action = np.concatenate([target_pose_xyzrpy, [gripper_cmd]]) # (8, )
    print(f'action osc is {action}')
    robot_interface.control(
        controller_type='OSC_POSE',
        action=action,
        controller_cfg=controller_cfg,
    )
    return
