#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : planner.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""
Wrapper for basic motion planning utils.
"""

from abc import ABC
from typing import Any, Optional, Literal, Iterator
from dataclasses import dataclass
import numpy as np
from utils import EvalConfig
from rotation_utils import euler2mat

Mat4 = np.ndarray #np.typing.NDArray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]
JointConf = np.ndarray
JointPath = list[JointConf]
CartesianGoal = Mat4
GripperMotion = Literal['open', 'close']


class Command(ABC):
    def __init__(self, name, *args):
        self.name = name
        self.args = args


class JointPathCommand(Command):
    def __init__(self, action: JointPath):
        super().__init__('joint_position_path', action)


class CartesianGoalCommand(Command):
    def __init__(self, action: Mat4):
        super().__init__('ee_cartesian_goal', action)


class GripperMotionCommand(Command):
    def __init__(self, action: GripperMotion):
        super().__init__('gripper_motion', action)


@dataclass
class GraspParameter:
    pregrasp_pose: Mat4 # mat 4
    grasp_pose: Mat4 # mat 4

CommandSequence = list[Command]

class MotionPlanner:
    def __init__(self, config):
        # Initialize the motion planner
        self.grasp_sampler = TopDownGraspSampler()
        self.iksolver = self.create_tracik_solver(config.iksolver_config.urdf_path, config.iksolver_config.tool_link_name)

    def create_tracik_solver(self, urdf_path, tool_link_name, max_time=0.0025, error=1e-2):
        from tracikpy import TracIKSolver
        tracik_solver = TracIKSolver(
            urdf_file=urdf_path,
            base_link='panda_link0',
            tip_link=tool_link_name,
            timeout=max_time,
            epsilon=error,
            solve_type="Distance",
        )  # Speed | Distance | Manipulation1 | Manipulation2
        assert tracik_solver.joint_names
        tracik_solver.urdf_file = urdf_path
        return tracik_solver

    def plan_arm_path(self, start_conf: JointConf, goal: Mat4) -> list[JointConf]:
        # Implement the motion planning algorithm
        end_conf = self.iksolver.ik(goal, qinit=start_conf)
        if end_conf is None:
            return None
        n_step = 100
        arm_path = [start_conf + (end_conf - start_conf) * i / n_step for i in range(n_step)] + [end_conf]
        return arm_path

    def sample_grasp_pose(self, pcd_cameraframe, pcd_worldframe, rgb_im, target_object_mask) -> Iterator[GraspParameter]:
        # Execute the planned motion
        target_object_pcd_worldframe = pcd_worldframe[target_object_mask]
        target_object_pcd_worldframe = target_object_pcd_worldframe[(target_object_pcd_worldframe!=0).any(axis=1)]
        for rv in self.grasp_sampler.sample_grasp(target_object_pcd_worldframe):
            yield rv

    def generate_grasp_path(self, start_qpos: JointConf, grasp_param: GraspParameter) -> CommandSequence:
        # Generate the path for the robot to grasp the object
        path_to_pregrasp = self.plan_arm_path(start_qpos, grasp_param.pregrasp_pose)
        if path_to_pregrasp is None:
            return None
        return CommandSequence([
            GripperMotionCommand('open'),
            JointPathCommand(path_to_pregrasp), CartesianGoalCommand(grasp_param.grasp_pose),
            GripperMotionCommand('close'), CartesianGoalCommand(grasp_param.pregrasp_pose)
        ])

    def plan_picking(self, start_qpos: JointConf, pcd_cameraframe, pcd_worldframe, rgb_im, target_object_mask) -> CommandSequence:
        for grasp_param in self.sample_grasp_pose(pcd_cameraframe, pcd_worldframe, rgb_im, target_object_mask):
            picking_command = self.generate_grasp_path(start_qpos, grasp_param)
            if picking_command is not None:
                return picking_command

    def plan_placing(self, qpos: JointConf, x_range, y_range, max_iter=10) -> CommandSequence:
        for _ in range(max_iter):
            placement_param = self.sample_placement_pose(x_range, y_range)
            placing_command = self.generate_placement_path(qpos, placement_param)
            if placing_command is not None:
                return placing_command

    def sample_placement_pose(self, x_range, y_range) -> GraspParameter:
        # Sample a placement pose for the object
        x_loc = np.random.uniform(x_range[0], x_range[1])
        y_loc = np.random.uniform(y_range[0], y_range[1])
        z = 0.05 # Fixed height for placement
        place_pose = np.eye(4)
        place_pose[:3, 3] = [x_loc, y_loc, z]
        preplace_pose = place_pose.copy()
        preplace_pose[2, 3] = z + 0.05
        return GraspParameter(world2ee(preplace_pose), world2ee(place_pose))

    def generate_placement_path(self, start_qpos: JointConf, grasp_param: GraspParameter) -> CommandSequence:
        # Generate the path for the robot to place the object
        path_to_pregrasp = self.plan_arm_path(start_qpos, grasp_param.pregrasp_pose)
        if path_to_pregrasp is None:
            return None
        return CommandSequence([
            JointPathCommand(path_to_pregrasp), CartesianGoalCommand(grasp_param.grasp_pose),
            GripperMotionCommand('open'), CartesianGoalCommand(grasp_param.pregrasp_pose)
        ])


def get_xyz_extend_from_mask(point_cloud, mask):
    """
    Get the XYZ extent from the point cloud and mask.
    Args:
        point_cloud (np.ndarray): The point cloud data.
        mask (np.ndarray): The mask for the object.
    Returns:
        tuple: The limits of the point cloud.
    """
    point_valid = point_cloud[mask]
    min_x, min_y, min_z = point_valid.min(axis=0)
    max_x, max_y, max_z = point_valid.max(axis=0)
    return min_x, max_x, min_y, max_y, min_z, max_z


class GraspSampler(ABC):
    def __init__(self):
        # Initialize the grasp sampler
        pass

    def sample_grasp(self, pcd, target_object_mask):
        # Sample a grasp for the object
        pass

    def generate_grasp_path(self, grasp_pose, qpos):
        # Generate the path for the robot to grasp the object
        pass


class TopDownGraspSampler(GraspSampler):
    def __init__(self):
        super().__init__()

    def sample_grasp(
        self,
        pointcloud: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rgb_im: Optional[np.ndarray] = None,
        grasp_depth: float = 0.05,
        max_trial: int = 1000,
    ) -> Iterator[GraspParameter]:
        top_center = np.array([*pointcloud[..., :2].mean(axis=0), pointcloud[..., 2].max()])
        grasp_center = top_center
        grasp_center[2] -= grasp_depth

        for _ in range(max_trial):
            yaw_angle = 2 * np.pi * np.random.rand()
            grasp_pose = np.eye(4)
            grasp_pose[:3, :3] = euler2mat([0, 0, yaw_angle])
            grasp_pose[:3, 3] = grasp_center
            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose[2, 3] += 0.03 + grasp_depth
            yield GraspParameter(world2ee(pregrasp_pose), world2ee(grasp_pose))


def world2ee(pose: Mat4) -> Mat4:
    """
    Convert the pose from world frame to end-effector frame.
    Args:
        pose (Mat4): The pose in world frame.
    Returns:
        Mat4: The pose in end-effector frame.
    """
    world2ee_tform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return pose.dot(world2ee_tform)


class AnyGraspSampler(GraspSampler):
    def __init__(self, config):
        # Initialize the any grasp sampler
        super().__init__()
        from gsnet import AnyGrasp
        from graspnetAPI import GraspGroup

        cfgs = EvalConfig({
            'checkpoint_path': config.anygrasp_ckpt_path,
            'max_gripper_width': 0.1,
            'gripper_height': 0.03,
            'top_down_grasp': True,
            'debug': False
        })

        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()

        self.anygrasp = anygrasp

    def sample_grasp(
        self, point_cloud: np.ndarray, mask: Optional[np.ndarray] = None, rgb_im: Optional[np.ndarray] = None,
    ) -> Iterator[dict[str, Any]]:
        point_cloud = point_cloud.astype(np.float32)
        if rgb_im is not None:
            rgb_im = rgb_im.astype(np.float32)
            if rgb_im.max() > 100:
                rgb_im = rgb_im / 255.
        lims = get_xyz_extend_from_mask(point_cloud, mask)
        grasp_poses, _ = self.anygrasp.get_grasp(
            point_cloud, rgb_im, lims=lims, apply_object_mask=True,
            dense_grasp=False, collision_detection=True
        )
        if len(grasp_poses) == 0:
            return None

        grasps = grasp_poses.nms().sort_by_score()
        if len(grasps) == 0:
            return list()

        for g in grasps[:20]:
            grasp_pose = np.eye(4)
            grasp_pose[:3, :3] = g.rotation_matrix
            grasp_pose[:3, 3] = g.translation
            yield {
                'score': g.score,
                'c2g_translation': g.translation,
                'c2g_rotation_matrix': g.rotation_matrix,
                'c2g_mat4': grasp_pose
            }


def initialize_planning_interface(config):
    """
    Initialize the planning interface.
    Returns:
        MotionPlanner: The initialized planning interface.
    """
    planning_interface = MotionPlanner(config)
    return planning_interface
