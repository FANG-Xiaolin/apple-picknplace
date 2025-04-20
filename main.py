#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""

"""
from robot_client_interface import initialize_robot_interface
from execution_manager import initialize_execution_manager
from perception import initialize_perception_interface, RGBDObservation
from planner import initialize_planning_interface
from utils import read_yaml, show_image_with_mask

GRASPABLE_WIDTH = 0.08

def main_pick_place_planned_franka():
    config = read_yaml('./config.yml')

    robot_interface = initialize_robot_interface(config.FR3_CONTROLLER_ADDR)
    perception_interface = initialize_perception_interface()
    planning_interface = initialize_planning_interface(config)
    execution_manager = initialize_execution_manager(config, robot_interface)

    current_qpos = robot_interface.get_current_joint_confs()
    print(f'Retrieved qpos from the server: {current_qpos}')

    object_name = 'peach'
    place_xrange = [0.2, 0.5]
    place_yrange = [-0.6, -0.1]

    rgb_im, dep_im, intrinsics = robot_interface.capture_image()
    extrinsic = robot_interface.get_gripper_camera_extrinsics()
    rgbd_observation = RGBDObservation(rgb_im, dep_im, intrinsics, extrinsic)

    target_object_mask = perception_interface.get_object_mask(rgbd_observation, object_name)
    show_image_with_mask(rgbd_observation.rgb_im, target_object_mask)
    picking_command_sequence = planning_interface.plan_picking(current_qpos, rgbd_observation.pcd_cameraframe, rgbd_observation.pcd_worldframe, rgbd_observation.rgb_im, target_object_mask)

    execution_manager.execute_commands(picking_command_sequence)
    robot_interface.go_to_home(gripper_open=False)

    current_qpos = robot_interface.get_current_joint_confs()
    print(f'Retrieved qpos from the server: {current_qpos}')

    placing_command_sequence = planning_interface.plan_placing(current_qpos, place_xrange, place_yrange)
    execution_manager.execute_commands(placing_command_sequence)

    robot_interface.go_to_home(gripper_open=True)


if __name__ == '__main__':
    main_pick_place_planned_franka()
