import os
import sys

# 添加父目录到 Python 路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from estimater import *
from datareader import *


import numpy as np
from scipy.spatial.transform import Rotation as R  # 四元数，旋转矩阵互相转化


# 使用 last.py 中已适配的 MMK2RealRobot
from last import MMK2RealRobot

def get_full_pose(base_point, new_quat): #  输入 目标在底座下的位置 和 目标的姿态，返回机械臂的位置和姿态组合
    # 移动到目标位置上方
    new_pos = [base_point[0], base_point[1], base_point[2]]
    waypoint = [
            new_pos[0],
            new_pos[1],
            new_pos[2],
            new_quat[0],
            new_quat[1],
            new_quat[2],
            new_quat[3] 
            # 0.5805051369025039, -0.5749042886531817, 0.40565205637421475, -0.4098112420094832
    ]
    return waypoint

import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_quaternion(quat, T):
    """
    将输入的四元数（姿态）从局部坐标系转换到另一个坐标系。

    参数:
        quat: list 或 np.ndarray, 四元数 [x, y, z, w]
        T: np.ndarray, 4x4 齐次变换矩阵 (例如 base_to_head)

    返回:
        new_quat: np.ndarray, 转换后的四元数 [x, y, z, w]
    """
    # 提取旋转矩阵部分
    R_base_to_head = T[:3, :3]

    # 原四元数对应的旋转矩阵
    R_head = R.from_quat(quat).as_matrix()

    # 转换后的旋转矩阵
    R_base = R_base_to_head @ R_head

    # 转为四元数
    new_quat = R.from_matrix(R_base).as_quat()
    return new_quat

def transform_point(point_camera, transform_matrix):
    """
    将 head_camera_link 下的点转换到 base_link 下
    :param point_camera: [x, y, z] in camera frame
    :param T_base_to_camera: 4x4 matrix (from tf2_echo base_link head_camera_link)
    :return: [x, y, z] in base_link frame
    """
    # 转成齐次坐标
    p_cam = np.array([*point_camera, 1.0])

    # 计算变换
    p_base = transform_matrix @ p_cam

    # 返回前三个分量
    return p_base[:3]


def matrix_to_pose(matrix):
    """
    将4×4变换矩阵转换为7元素位姿表示 [x, y, z, qx, qy, qz, qw]
    
    参数:
        matrix: 4×4 numpy数组或列表，包含旋转和平移信息
    
    返回:
        7元素列表，前3个为位置，后4个为四元组姿态
    """
    # 确保输入是 numpy 数组
    matrix = np.array(matrix)
    
    # 提取平移分量 (x, y, z)
    tx = matrix[0, 3]
    ty = matrix[1, 3]
    tz = matrix[2, 3]
    
    # 提取旋转矩阵部分 (3×3)
    rotation_matrix = matrix[:3, :3]
    
    # 转换旋转矩阵为四元组 (x, y, z, w)
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    
    # 组合位置和姿态
    return [tx, ty, tz, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]


if __name__ == '__main__':

    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    (head, lift, 
    left_arm, right_arm,
    left_arm_eef, right_arm_eef,
    left_gripper, right_gripper, 
    base_pose) = mmk2.get_all_joint_states()     # 3. 获取机械臂所有关节/末端状态（头部、升降、左右臂、夹爪等）
    print(head, lift, 
            left_arm, right_arm,
            left_arm_eef, right_arm_eef,
            left_gripper, right_gripper, 
            base_pose)
    print("left_arm_eef: ", left_arm_eef)


   # left_arm_eef1:  [[0.5856411606810898, 0.05454282735765471, 1.102595391982941], [0.8094673107705215, -0.5862670547983784, 0.029102726052135798, 0.014375137853587716]]
   # left_arm_eef2:  [[0.603514545046778, -0.18193094582205668, 0.9929337517509562], [0.7698759191802366, -0.6380571150084569, 0.013196712631442429, 0.0001839271429610308]]


    init_pose = [0.4798196384291117, 0.050344892205700036, 1.3300944085789266,0.004922124603778732, -0.003476176971870248, -0.6913608163682425, 0.7224845399547885]
    mmk2.control_arm_pose('left_arm', init_pose) # 初始点的位置+位姿

    # mmk2.set_robot_eef('left_arm', 1)

    # grasp_rot = [0.09538087446650391, 0.019704521010289917, -0.5310926713809747, 0.8416975674452077] # 抓取点的四元数位姿

    # # foundaationpose下的坐标
    # point_fdp_camera_xyz = [0.0508, 0.0562, 0.5299]     # 提取目标在相机下的位置  正确[0.0509, 0.1018, 0.5209]    m

    # # ros2 run tf2_ros tf2_echo base_link head_camera_link
    # T_head_to_base = np.array([
    #     [-0.001 ,-0.749 , 0.662  ,0.365],
    #     [-1.000 , 0.001, -0.001 , 0.036],
    #     [-0.000, -0.662, -0.749 , 1.516],
    #     [0.000,  0.000 , 0.000,  1.000],
    # ])
    # point_prepare_base_xyz = transform_point(point_fdp_camera_xyz, T_head_to_base) #转换：相机下的位置 → 底座下的位置（调用transform_point函数）


    # # ros2 run tf2_ros tf2_echo base_link left_arm_end_link
    # T_left_arm_to_base = np.array([
    #     [0.042 , 0.999 ,-0.012 , 0.480],
    #     [-0.999 , 0.042 , 0.004,  0.050],
    #     [0.005 , 0.012  ,1.000 , 1.330],
    #     [0.000 , 0.000 , 0.000 , 1.000],
    # ])
    # point_fdp_left_arm_xyz = transform_point(point_prepare_base_xyz, np.linalg.inv(T_left_arm_to_base))# 这里需要底座到左臂，所以用逆矩阵
    # # point_left_arm:在左臂坐标系下
    # point_fdp_left_arm_xyz[2] += 0.15 # 让左臂末端向上移动 z方向
    # point_fdp_left_arm_xyz[1] += 0.02 
    # point_prepare_base_xyz = transform_point(point_fdp_left_arm_xyz, T_left_arm_to_base)
    # print('point_prepare_base_xyz: ', point_prepare_base_xyz)   

    # mmk2.set_robot_eef('left_arm', 1)
    # point_prepare_base = list(point_prepare_base_xyz) + list(grasp_rot)
    # print(point_prepare_base)
    # mmk2.control_arm_pose('left_arm', point_prepare_base)

    # # 这里到抓取位置
    # point_prepare_left_arm_xyz = transform_point(point_prepare_base_xyz, np.linalg.inv(T_left_arm_to_base))# 这里需要底座到左臂，所以用逆矩阵
    # # point_left_arm:在左臂坐标系下
    # point_prepare_left_arm_xyz[2] -= 0.09 # 让左臂末端向上移动 z方向

    # point_grasp_base_xyz = transform_point(point_prepare_left_arm_xyz, T_left_arm_to_base)
    # print('point_prepare_base_xyz: ', point_grasp_base_xyz)   
    # point_grasp_base = list(point_grasp_base_xyz) + list(grasp_rot)
    # mmk2.control_arm_pose('left_arm', point_grasp_base)
    # mmk2.set_robot_eef('left_arm', 0)


    # mmk2.control_arm_pose('left_arm', point_prepare_base)
    # mmk2.control_arm_pose('left_arm', init_pose)

