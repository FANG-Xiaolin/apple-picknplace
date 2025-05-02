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
import tap
from beepp.robot_client_interface import initialize_robot_interface
from beepp.execution_manager import initialize_execution_manager
from beepp.perception import initialize_perception_interface, RGBDObservation
from beepp.planner import initialize_planning_interface
from beepp.utils.common_utils import read_yaml, show_image_with_mask

def main_pick_place_planned_franka():
    config = BeeppConfig()
    sys_configs = read_yaml('./config.yml')

    robot_interface = initialize_robot_interface(sys_configs, config.run_in_simulation)
    perception_interface = initialize_perception_interface()
    planning_interface = initialize_planning_interface(sys_configs)
    execution_manager = initialize_execution_manager(sys_configs, robot_interface)

    for run_i in range(config.num_runs):
        print(f'>>>> Running {run_i + 1} / {config.num_runs} <<<')
        try:
            rgb_im, dep_im, intrinsics = robot_interface.capture_image()
            extrinsic = robot_interface.get_gripper_camera_extrinsics()
            rgbd_observation = RGBDObservation(rgb_im, dep_im, intrinsics, extrinsic)

            target_object_mask = perception_interface.get_object_mask(rgbd_observation, config.object_name)
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
            placing_command_sequence = planning_interface.plan_placing(current_qpos, config.place_xrange, config.place_yrange)
            print('Executing placing command sequence...')
            execution_manager.execute_commands(
                placing_command_sequence, is_capturing=config.is_capturing,
                capture_save_name=f'{config.save_dir}/place_{config.object_name}_{time.strftime("%Y%m%d-%H%M%S")}.pkl'
            )
            print('Finish placing command sequence.')

            robot_interface.go_to_home(gripper_open=True)
        except:
            pass


class BeeppConfig(tap.Tap):
    vis: bool = False
    run_in_simulation: bool = False

    num_runs: int = 10
    num_trial: int = 4 # number of trials for each run

    # Data collection Settings
    is_capturing: bool = True
    save_dir: str = 'default'
    object_name: str = 'apple'
    place_xrange: list[float] = [0.45, 0.7]
    place_yrange: list[float] = [-0.6, -0.1]


if __name__ == '__main__':
    main_pick_place_planned_franka()
