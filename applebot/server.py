import zmq
import zlib
import pickle
import time
import argparse

import numpy as np
import os.path as osp

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to, follow_joint_traj
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils import YamlConfig, transform_utils
from beta_scripts.osc_traj_following import interpolate_pose_trajectory

from applebot.utils.rs_capture import get_realsense_capturer_dict
from applebot.utils.deoxys_utils import osc_move_to_target_absolute_pose_controller, get_franka_interface


def reset_joint_to(message):
    reset_joints_to(robot_interface, message['qpos'], gripper_open=message['gripper_open'])
    return_message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(return_message)))


def capture_realsense(message):
    (rgb, depth), intrinsics = cameras[target_camera_name].capture(), cameras[target_camera_name].intrinsics
    message = {"rgb": rgb,
			   "depth": depth/1000.,
			   "intrinsics": intrinsics[0]
               }
    socket.send(zlib.compress(pickle.dumps(message)))


def get_fixed_camera_extrinsic(message):
    with open(osp.join(osp.dirname(__file__), 'calibrations', 'camera_configs_latest.pkl'), 'rb') as f:
        camera_configs = pickle.load(f)
    message = {
        message['camera_name']: camera_configs[message['camera_name']]['extrinsics']
    }
    socket.send(zlib.compress(pickle.dumps(message)))


def get_joint_states(message):
    last_state = robot_interface._state_buffer[-1] #.copy()
    last_gripper_state = robot_interface._gripper_state_buffer[-1].width if len(
        robot_interface._gripper_state_buffer) > 0 else 0.0
    message = {
        'ee_pose': np.array(last_state.O_T_EE).reshape(4,4).T,
        'qpos': np.array(last_state.q),
        'gripper_state': np.array(last_gripper_state)
    }
    socket.send(zlib.compress(pickle.dumps(message)))


def open_gripper(message):
    last_state = robot_interface._state_buffer[-1] #.copy()
    last_eepose_mat4 = np.array(last_state.O_T_EE).reshape(4,4).T
    osc_move_to_target_absolute_pose_controller(robot_interface, controller_cfg_osc,
                                                last_eepose_mat4, 
                                                gripper_open=True)
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def close_gripper(message):
    last_state = robot_interface._state_buffer[-1]  # .copy()
    last_eepose_mat4 = np.array(last_state.O_T_EE).reshape(4, 4).T
    osc_move_to_target_absolute_pose_controller(robot_interface, controller_cfg_osc,
                                                last_eepose_mat4, 
                                                gripper_close=True)
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def go_to_home(message):
    reset_joints_to(robot_interface, robot_interface.init_q, gripper_open=message['gripper_open'])
    return_message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(return_message)))


def free_motion_control_for_time(
    gripper_open: bool,
    timeout: float
):
    st = time.time()

    current_ee_pose = robot_interface.last_eef_pose.copy()
    current_pos = current_ee_pose[:3, 3:]
    current_rot = current_ee_pose[:3, :3]
    current_quat = transform_utils.mat2quat(current_rot)
    current_axis_angle = transform_utils.quat2axisangle(current_quat)
    target_pose = current_pos.flatten().tolist() + current_axis_angle.flatten().tolist()

    while True:
        action = target_pose + ([-1.0] if gripper_open else [+1.0])
        robot_interface.control(
            controller_type=controller_type_free,
            action=action,
            controller_cfg=controller_cfg_free,
        )

        new_ee_pose = robot_interface.last_eef_pose.copy()
        new_pos = new_ee_pose[:3, 3:]
        new_rot = new_ee_pose[:3, :3]
        new_quat = transform_utils.mat2quat(current_rot)
        new_pos = new_pos.flatten().tolist()
        new_quat = new_quat.flatten().tolist()
        if time.time() - st > timeout:
            return


def free_motion_control(message):
    free_motion_control_for_time(gripper_open=message['gripper_open'], timeout=message['timeout'])
    return_message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(return_message)))


def execute_posesmat4_osc(message):
    pred_traj = message['ee_poses']
    speed_factor = message['speed_factor']
    # gripper_isopen = 1 - message['gripper_isclose'] # TODO
    use_smoothing = message.get('use_smoothing', True)
    print(f'move {len(pred_traj)} steps.')

    if use_smoothing:
        if speed_factor > 1:
            pred_traj, indices = interpolate_pose_trajectory(pred_traj, speed_factor)
        else:
            indices = range(len(pred_traj))

    for i, step in enumerate(pred_traj):
        if use_smoothing:
            osc_move_to_target_absolute_pose_controller(
                robot_interface, controller_cfg_osc,
                step, 
            )
        else:
            for _ in range(speed_factor):
                osc_move_to_target_absolute_pose_controller(
                    robot_interface, controller_cfg_osc,
                    step,                 
                )
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def execute_joint_impedance_path(message):
    joint_confs = message['joint_confs']
    gripper_isclose = message['gripper_isclose']
    print(f'move {len(joint_confs)} steps.')
    follow_joint_traj(
        robot_interface, joint_confs, controller_cfg=controller_cfg_imp,
        gripper_close=gripper_isclose,
    )
    socket.send(zlib.compress(pickle.dumps(message)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument('--open', action='store_true', help="open the gripper")
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    global controller_type_osc, controller_cfg_osc
    global controller_type_imp, controller_cfg_imp
    global controller_type_free, controller_cfg_free
    global context, socket

    controller_type_osc = 'OSC_POSE'
    controller_cfg_osc = YamlConfig(config_root + '/osc-pose-controller-absolute.yml').as_easydict()
    controller_type_imp = 'JOINT_IMPEDANCE'
    controller_cfg_imp = YamlConfig(config_root + '/joint-impedance-controller.yml').as_easydict()
    controller_type_free = 'OSC_POSE'
    controller_cfg_free = YamlConfig(config_root + '/osc-free-motion-controller.yml').as_easydict()

    if args.reset:
        print(f'reset')
        reset_joints_to(robot_interface, robot_interface.init_q, gripper_open=args.open)
    else:
        print(f'no reset')

    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5560")
    try:
        while True:
            #  Wait for next request from client
            print("Waiting for request...")
            message = pickle.loads(zlib.decompress(socket.recv()))

            print("Received request: {}".format(message))

            #  Send reply back to client
            globals()[message['message_name']](message)
    except Exception as e:
        print('error')
        print(e)
        import traceback; traceback.print_exc()
    finally:
        print('exit')
        socket.close()
        try:
            context.term()
        except:
            pass


if __name__ == '__main__':
    logger = get_deoxys_example_logger()
    robot_interface = get_franka_interface(robot_index=1, wait_for_state=True, auto_close=True)
    # robot_interface.set_open_gripper_width(0.06) # default gripper width

    target_camera_name = 'robot1_hand' # 'mount2'  # 
    cameras = get_realsense_capturer_dict([target_camera_name], auto_close=True, skip_frames=35)

    while True:
        try:
            print('(Re) starting the server...')
            main()
        except:
            pass
        print('Server errored... Waiting for 2 seconds before restarting...')
        time.sleep(2)

