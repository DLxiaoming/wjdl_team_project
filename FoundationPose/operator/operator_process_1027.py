import os
import sys

# 添加父目录到 Python 路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from estimater import *
from datareader import *
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R  # 四元数，旋转矩阵互相转化
from time import sleep

import cv2


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

    # parser = argparse.ArgumentParser()
    # code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/grasp_7/mesh/textured_simple.obj')
    # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/grasp_7')
    # parser.add_argument('--est_refine_iter', type=int, default=5)
    # parser.add_argument('--track_refine_iter', type=int, default=2)
    # parser.add_argument('--debug', type=int, default=1)
    # parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    # args = parser.parse_args()

    # set_logging_format()
    # set_seed(0)

    # mesh = trimesh.load(args.mesh_file)

    # debug = args.debug
    # debug_dir = args.debug_dir
    # os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # scorer = ScorePredictor()
    # refiner = PoseRefinePredictor()
    # glctx = dr.RasterizeCudaContext()
    # est = FoundationPose(model_pts=mesh.vertices, 
    #                       model_normals=mesh.vertex_normals, 
    #                       mesh=mesh, 
    #                       scorer=scorer, 
    #                       refiner=refiner, 
    #                       debug_dir=debug_dir, 
    #                       debug=debug, 
    #                       glctx=glctx)
    # logging.info("estimator initialization done")

    # reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # i = 0
    # color = reader.get_color(i)
    # depth = reader.get_depth(i)
    # mask = reader.get_mask(0).astype(bool)
    # pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    
    # foundaationpose下的坐标
    pose_result = np.array([[    0.71772  , 0.0040997  ,  -0.69632  ,-0.0048338],
                             [    0.69633 , -0.0027474 ,    0.71772 ,  -0.073031],
                             [  0.0010294 ,   -0.99999 , -0.0048265 ,    0.37994],
                             [          0  ,         0 ,          0   ,        1]])
    print("pose: ", pose_result)
    pose_head = matrix_to_pose(pose_result)# pose_head是相机坐标系下的位姿
    print("pose_head: ", pose_head)
    pose_arm = [0.051514246, 0.09068671, 0.5053671, 0.8962801876980515, -0.026652786048024537, 0.05711456504334031, -0.43898676585488755]
    pose_head = pose_head[0:3] + pose_arm[3:7] # 相当于获取到了目标在底座下的位置和姿态（七个数，fdp的三个位姿+机械臂原始姿态）
    # ------------------------------------------------

    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    left_arm = mmk2.get_arm_ee_pose('left_arm') # 获取左臂末端在base坐标系下的位姿
    print("left_arm: ", left_arm)
    # init = [0.5902837695659998, -0.03620627453000008, 1.21398344875, 0.14581857475746277, 0.18977384261801228, -0.5684452694497939, 0.7871421774710381]
    # mmk2.control_arm_pose('left_arm', init)

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
# [0.009346150793135166, -0.0078202486038208] 
# -0.00061425 
# [0.20084688067436218, -1.730945348739624, 1.8514915704727173, 2.0986876487731934, -0.6162737607955933, -0.3263523280620575] 
# [0.1516365259885788, -0.6738765835762024, 0.9187838435173035, -1.7622263431549072, 1.037422776222229, 0.14553292095661163] 
# [[0.6896952455901185, -0.1368310337099575, 1.1137532576055147], [0.14664337706177522, 0.18399240188463753, -0.5706684248776973, 0.786752861361335]] 
# [[0.4598274399645455, -0.046575647017603194, 1.3203971970561306], [-0.04112684579159852, -0.00023148028489630934, 0.7025685842696603, 0.7104265714128638]] 
# [0.9712129086256027] 
# [0.41616596281528473] 
# [0.0, 0.0, 0.0]
    # joint_action = [0.20084688067436218, -1.730945348739624, 1.8514915704727173, 2.0986876487731934, -0.6162737607955933, -0.3263523280620575] 
    # mmk2.control_arm_joints('left_arm', joint_action)
    # ros2 run tf2_ros tf2_echo base_link head_camera_link
#   0.004 -0.900  0.435  0.392
#  -1.000 -0.004  0.002  0.037
#  -0.000 -0.435 -0.900  1.346
#   0.000  0.000  0.000  1.000
    T_head_to_base = np.array([ # 相机坐标系到base坐标系的变换矩阵
        [0.004, -0.900, 0.435, 0.392],
        [-1.000, -0.004, 0.002, 0.037],
        [-0.000, -0.435, -0.900, 1.346],
        [0.000, 0.000, 0.000, 1.000],
    ])
    # point_camera = [0, 0, 0.222]
    point_camera = pose_head[0:3]  # 提取目标在相机下的位置
    point_base = transform_point(point_camera, T_head_to_base) #转换：相机下的位置 → 底座下的位置（调用transform_point函数）
    print('point_base1: ', point_base)   


# ros2 run tf2_ros tf2_echo base_link left_arm_end_link
#   0.035  0.999 -0.019  0.478
#  -0.999  0.035 -0.015  0.052
#  -0.014  0.020  1.000  1.207
#   0.000  0.000  0.000  1.000
# ros2 run tf2_ros tf2_echo base_link left_arm_base_link
    T_left_arm_to_base = np.array([ # 左臂坐标系到base坐标系的变换矩阵
        [0.035, 0.999, -0.019, 0.478],
        [-0.999, 0.035, -0.015, 0.052],
        [-0.014, 0.020, 1.000, 1.207],
        [0.000, 0.000, 0.000, 1.000],
    ])
    point_left_arm = transform_point(point_base, np.linalg.inv(T_left_arm_to_base))# 这里需要底座到左臂，所以用逆矩阵
    point_left_arm[0] -= 0.01 # 让左臂末端向左移动0.01米
    point_left_arm[2] += 0.07 # 让左臂末端向上移动0.07米  

    point_base = transform_point(point_left_arm, T_left_arm_to_base)
    print('point_base2: ', point_base)   


    # gt = [0.5875903774804702, 0.04060842535182542, 1.10696762318877]
    # gt_camera = transform_camera_to_base(gt, np.linalg.inv(T_base_to_head))
    # print('gt_camera: ', gt_camera)
    # gt_camera:  [-0.00461075 -0.00232876  0.41868768]     

    # quat_camera = pose_head[3:7]
    # quat_camera = Rotation.from_euler('zyx', [0, 0, 0]).as_quat()
    # quat_base = transform_quaternion(quat_camera, T_base_to_head)
    quat_base = [0.14581857475746277, 0.18977384261801228, -0.5684452694497939, 0.7871421774710381]


    # ------------------------------------------
    full_pose_base = get_full_pose(point_base, quat_base)
    # full_pose_base[2] = full_pose_base[2] + 0.05

    left_arm_eef = [0.6125017600737456, 0.05068043188840441, 1.1034711546538305, 0.17910768107670647, 0.03399635589282703, -0.5913176453641912, 0.7855623009324477]
    
    print('full_pose_base_0: ', full_pose_base)

    mmk2.set_robot_eef('left_arm', 1)
    full_pose_base = full_pose_base[0:3] + left_arm_eef[3:7]
    mmk2.control_arm_pose('left_arm', full_pose_base)
    mmk2.set_robot_eef('left_arm', 0)


    point_left_arm = transform_point(full_pose_base[0:3], np.linalg.inv(T_left_arm_to_base))
    point_left_arm[2] += 0.10
    point_base = transform_point(point_left_arm, T_left_arm_to_base)
    print('point_base2: ', point_base) 
    full_pose_base = list(point_base[0:3]) + list(left_arm_eef[3:7])
    mmk2.control_arm_pose('left_arm', full_pose_base)


#     full_pose_base[2] -= 0.1
#     mmk2.control_arm_pose('left_arm', full_pose_base)
#     mmk2.set_robot_eef('left_arm', 0)
#     print('full_pose_base_1: ', full_pose_base)

#     # full_pose_base[2] += 0.1
#     # mmk2.control_arm_pose('left_arm', full_pose_base)

#     # full_pose_base[0] += 0.2
#     # full_pose_base[1] += 0.2
#     pose_base = [0.8868162429374099, 0.21029169690800675, 1.2578051635026589, -0.016453529876487794, 0.16611498795436896, -0.1737661554229103, 0.9705361484051299]
#     mmk2.control_arm_pose('left_arm', pose_base)
    
#     pose_base = [0.8868162429374099, 0.21029169690800675, 1.2498051635026589, -0.016453529876487794, 0.16611498795436896, -0.1737661554229103, 0.9705361484051299]
#     mmk2.control_arm_pose('left_arm', pose_base)
#     time.sleep(0.3)
#     mmk2.set_robot_eef('left_arm', 1)

#     # full_pose_base[2] -= 0.1
#     # mmk2.control_arm_pose('left_arm', full_pose_base)
#     # mmk2.set_robot_eef('left_arm', 1)





# #     root@orangepi5plus:/# ros2 run tf2_ros tf2_echo base_link left_arm_base_link
# # [INFO] [1757265801.349815255] [tf2_echo]: Waiting for transform base_link ->  left_arm_base_link: Invalid frame ID "base_link" passed to canTransform argument target_frame - frame does not exist
# # At time 1757265802.299674225
# # - Translation: [0.057, 0.101, 1.312]
# # - Rotation: in Quaternion [-0.653, 0.271, -0.271, 0.653]
# # - Rotation: in RPY (radian) [-1.571, 0.000, -0.785]
# # - Rotation: in RPY (degree) [-90.000, 0.000, -45.000]
# # - Matrix:
# #   0.707 -0.000  0.707  0.057
# #  -0.707  0.000  0.707  0.101
# #  -0.000 -1.000 -0.000  1.312
# #   0.000  0.000  0.000  1.000
# # At time 1757265803.304687368
# # - Translation: [0.057, 0.101, 1.312]
# # - Rotation: in Quaternion [-0.653, 0.271, -0.271, 0.653]
# # - Rotation: in RPY (radian) [-1.571, 0.000, -0.785]
# # - Rotation: in RPY (degree) [-90.000, 0.000, -45.000]
# # - Matrix:
# #   0.707 -0.000  0.707  0.057
# #  -0.707  0.000  0.707  0.101
# #  -0.000 -1.000 -0.000  1.312
# #   0.000  0.000  0.000  1.000
# # ^C[INFO] [1757265803.752641038] [rclcpp]: signal_handler(signum=2)
# # root@orangepi5plus:/# 

