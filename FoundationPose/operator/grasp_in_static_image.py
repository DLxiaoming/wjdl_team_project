# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
2

from estimater import *
from datareader import *
import argparse


import numpy as np
from scipy.spatial.transform import Rotation as R
from time import sleep

from robot.mmk2_robot_sdk import MMK2RealRobot

def tf_head_camera_to_base(head_pose, T_base_to_head):
    """
    将 head_camera_link 系下的位姿转换为 base_link 系下的位姿
    
    参数:
        head_pose (list/tuple): head_camera_link 系下的位姿，格式为 [x, y, z, qx, qy, qz, qw]
                                其中 (x,y,z) 是平移（单位：米），(qx,qy,qz,qw) 是四元数（旋转）
        T_base_to_head (np.ndarray): base_link 到 head_camera_link 的 4×4 齐次变换矩阵
                                     （即 tf2_echo base_link head_camera_link 输出的 Matrix）
    
    返回:
        base_pose (list): base_link 系下的位姿，格式为 [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]
    """
    # 1. 解析 head_pose 的平移和旋转
    x_head, y_head, z_head = head_pose[0], head_pose[1], head_pose[2]
    q_head = head_pose[3:7]  # [qx, qy, qz, qw]

    # 2. 将 head 系位姿转换为 4×4 齐次变换矩阵 T_head_pose
    # 2.1 旋转部分：四元数 → 3×3 旋转矩阵（scipy 默认四元数格式为 (x,y,z,w)）
    R_head = R.from_quat(q_head).as_matrix()  # 3×3 旋转矩阵
    # 2.2 平移部分：构造 4×4 齐次矩阵
    T_head_pose = np.eye(4)  # 初始化 4×4 单位矩阵
    T_head_pose[:3, :3] = R_head  # 填充旋转矩阵
    T_head_pose[:3, 3] = np.array([x_head, y_head, z_head])  # 填充平移向量

    # 3. 计算 base 系下的变换矩阵：T_base_pose = T_base→head × T_head_pose
    T_base_pose = np.dot(T_base_to_head, T_head_pose)

    # 4. 从 T_base_pose 中提取 base 系位姿
    # 4.1 提取平移（x_base, y_base, z_base）
    x_base, y_base, z_base = T_base_pose[:3, 3]
    # 4.2 提取旋转（3×3 旋转矩阵 → 四元数）
    R_base = T_base_pose[:3, :3]
    q_base = R.from_matrix(R_base).as_quat()  # 四元数格式：[qx, qy, qz, qw]

    # 5. 整理输出：[x, y, z, qx, qy, qz, qw]
    base_pose = [x_base, y_base, z_base, q_base[0], q_base[1], q_base[2], q_base[3]]
    return base_pose


def trans_tf_format(T_base_pose):
    x_base, y_base, z_base = T_base_pose[:3, 3]
    R_base = T_base_pose[:3, :3]
    q_base = R.from_matrix(R_base).as_quat()  # 四元数格式：[qx, qy, qz, qw]
    base_pose = [x_base, y_base, z_base, q_base[0], q_base[1], q_base[2], q_base[3]]
    return base_pose

import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_pose(matrix):
    """
    将4×4变换矩阵转换为7元素位姿表示 [x, y, z, qx, qy, qz, qw]
    
    参数:
        matrix: 4×4 numpy数组或列表，包含旋转和平移信息
    
    返回:
        7元素列表，前3个为位置，后4个为四元组姿态
    """
    # 提取平移分量 (x, y, z)
    tx = matrix[0, 3]
    ty = matrix[1, 3]
    tz = matrix[2, 3]
    
    # 提取旋转矩阵部分 (3×3)
    rotation_matrix = np.array([
        [matrix[0, 0], matrix[0, 1], matrix[0, 2]],
        [matrix[1, 0], matrix[1, 1], matrix[1, 2]],
        [matrix[2, 0], matrix[2, 1], matrix[2, 2]]
    ])
    
    # 转换旋转矩阵为四元组 (x, y, z, w)
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    
    # 组合位置和姿态
    return [tx, ty, tz, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/grasp_nearby/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/grasp_nearby')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, 
                          model_normals=mesh.vertex_normals, 
                          mesh=mesh, 
                          scorer=scorer, 
                          refiner=refiner, 
                          debug_dir=debug_dir, 
                          debug=debug, 
                          glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    i = 0
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(0).astype(bool)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    print("pose: ", pose)
    pose_hm = matrix_to_pose(pose)
    print("pose_hm: ", pose_hm)


    mmk2 = MMK2RealRobot(ip="192.168.11.200")

#  -0.000 -0.857  0.515  0.385
#  -1.000  0.000 -0.000  0.036
#  -0.000 -0.515 -0.857  1.185
#   0.000  0.000  0.000  1.000

    T_base_to_head = np.array([
    [-0.000 ,-0.857 , 0.515 , 0.385],
    [-1.000,  0.000 ,-0.000 , 0.036],
    [-0.000 ,-0.515 ,-0.857 , 1.185],
    [  0.000 , 0.000  ,0.000  ,1.000],
    ])

    # pose_in_head_camera_link = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    # print("pose_in_head_camera_link1: ", pose_in_head_camera_link)
    pose_in_head_camera_link = pose_hm
    print("pose_in_head_camera_link2: ", pose_in_head_camera_link)

    pose_in_base_link = tf_head_camera_to_base(pose_in_head_camera_link, T_base_to_head)
    print("pose_in_base_link: ", pose_in_base_link)


    # 目标位置 真值姿态
    # pose_in_base_link = [pose_in_base_link[0], pose_in_base_link[1], pose_in_base_link[2],
    #                         0.12455700933529998, 
    #                         0.10708387712504314, 
    #                         -0.7291753532268325, 
    #                         0.6643206296148259]

    left_arm_before = mmk2.get_arm_ee_pose('left_arm')
    print("left_arm_before: ", left_arm_before)

    mmk2.control_arm_pose('left_arm', pose_in_base_link)

    left_arm_after = mmk2.get_arm_ee_pose('left_arm')
    print("left_arm_after: ", left_arm_after)

    # left_arm:  [[0.5767181702710851, -0.07468125484927444, 0.8771478614212382], 
    # [0.12455700933529998, 0.10708387712504314, -0.7291753532268325, 0.6643206296148259]]

    # head, lift, left_arm, right_arm, left_arm_eef, right_arm_eef, left_gripper, right_gripper, base_pose = mmk2.get_all_joint_states()
    # print("head: ", head)
    # print("lift: ", lift)
    # print("left_arm: ", left_arm)
    # print("right_arm: ", right_arm)
    # print("left_arm_eef: ", left_arm_eef)
    # print("right_arm_eef: ", right_arm_eef)
    # print("left_gripper: ", left_gripper)
    # print("right_gripper: ", right_gripper)
    
    # base_pose = mmk2.get_base_pose()
    # print("base_pose: ", base_pose)



