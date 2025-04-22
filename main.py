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
import time
from applebot.robot_client_interface import initialize_robot_interface
from applebot.execution_manager import initialize_execution_manager
from applebot.perception import initialize_perception_interface, RGBDObservation
from applebot.planner import initialize_planning_interface
from applebot.utils.common_utils import read_yaml, show_image_with_mask

def main_pick_place_planned_franka():
    config = read_yaml('./config.yml')

    robot_interface = initialize_robot_interface(config)
    perception_interface = initialize_perception_interface()
    planning_interface = initialize_planning_interface(config)
    execution_manager = initialize_execution_manager(config, robot_interface)

    is_capturing = True
    trial_num = 3
    object_name = 'apple'
    place_xrange = [0.45, 0.7]
    place_yrange = [-0.6, -0.1]

    for _ in range(trial_num):
        rgb_im, dep_im, intrinsics = robot_interface.capture_image()
        extrinsic = robot_interface.get_gripper_camera_extrinsics()
        rgbd_observation = RGBDObservation(rgb_im, dep_im, intrinsics, extrinsic)

        target_object_mask = perception_interface.get_object_mask(rgbd_observation, object_name)
        if config.vis:
            show_image_with_mask(rgbd_observation.rgb_im, target_object_mask)

        current_qpos = robot_interface.get_current_joint_confs()
        picking_command_sequence = planning_interface.plan_picking(current_qpos, rgbd_observation.pcd_cameraframe, rgbd_observation.pcd_worldframe, rgbd_observation.rgb_im, target_object_mask)
        print('Executing picking command sequence...')
        is_success = execution_manager.execute_commands(
            picking_command_sequence, is_capturing=is_capturing, capture_save_name=f'saved/pick_{object_name}_{time.strftime("%Y%m%d-%H%M%S")}.pkl',
            success_checker_hook=lambda: robot_interface.get_gripper_state() > 0.01
        )
        print('Finish picking command sequence.')

        if not is_success:
            print('Grasp failed.')
            robot_interface.go_to_home(gripper_open=True)
            continue

        robot_interface.go_to_home(gripper_open=False)

        current_qpos = robot_interface.get_current_joint_confs()
        placing_command_sequence = planning_interface.plan_placing(current_qpos, place_xrange, place_yrange)
        print('Executing placing command sequence...')
        execution_manager.execute_commands(placing_command_sequence, is_capturing=is_capturing, capture_save_name=f'saved/place_{object_name}_{time.strftime("%Y%m%d-%H%M%S")}.pkl')
        print('Finish placing command sequence.')

        robot_interface.go_to_home(gripper_open=True)


if __name__ == '__main__':
    main_pick_place_planned_franka()
