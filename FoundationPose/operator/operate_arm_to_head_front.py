

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


# ------------------- 示例：使用你的 tf2_echo 输出进行测试 -------------------
if __name__ == "__main__":

    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    # mmk2.set_move_base_zero()
    # base_pose = mmk2.get_base_pose()
    # print("base_pose1: ", base_pose)

    # print("waiting for 10s")
    # sleep(10)
    # print("---------------------------------------------------------")

    # base_pose = mmk2.get_base_pose()
    # print("base_pose2: ", base_pose)

#  -0.000 -0.857  0.515  0.385
#  -1.000  0.000 -0.000  0.036
#  -0.000 -0.515 -0.857  1.185
#   0.000  0.000  0.000  1.000


    T_base_to_head = np.array([
            [0.374, -0.534 , 0.758 , 0.309],
            [-0.927 ,-0.215 , 0.306 , 0.080],
            [-0.000 ,-0.817 ,-0.576 , 1.549],
            [0.000  ,0.000,  0.000  ,1.000],
    ])

    pose_in_head_camera_link = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]

    pose_in_base_link = tf_head_camera_to_base(pose_in_head_camera_link, T_base_to_head)
    print("pose_in_base_link: ", pose_in_base_link)

    left_arm_before = mmk2.get_arm_ee_pose('left_arm')
    print("left_arm_before: ", left_arm_before)
    mmk2.control_arm_pose('left_arm', pose_in_base_link)
    mmk2.set_robot_eef('left_arm', 1.0)

    left_arm_after = mmk2.get_arm_ee_pose('left_arm')
    print("left_arm_after: ", left_arm_after)


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