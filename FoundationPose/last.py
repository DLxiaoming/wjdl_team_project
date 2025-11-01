

     
from mmk2_types.types import RobotComponents as MMK2Components, ImageTypes
from mmk2_types.config import JointNames
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    GoalStatus,
    BaseControlParams,
    BuildMapParams,
    Pose3D,
    Twist3D,
    Pose,
    Position,
    Orientation,
    TrackingParams,
    ForwardPositionParams,
)
from airbot_py.airbot_mmk2 import AirbotMMK2
import logging
import time
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


START_ALL_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
}

START_LEFT_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
}

START_RIGHT_JOINT_ACTION = {
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    # MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.0, 0.0, -1.57, 0.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
}

STOP_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[1.52, -2.1, 2.0, 1.4, 0.1, -0.62]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

TEST_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[1.197, -1.082,  0.71,   2.595,  1.415,  0.499]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.197, -1.082,  0.71,   -2.595,  -1.415,  -0.499]),
    # MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, -0.8]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

class MMK2RealRobot():
    def __init__(self, ip="192.168.11.200"):
        self.mmk2 = AirbotMMK2(ip)
        self._joint_order_cache = None
        logger.info("开始创建相机实例...")
        try:
            self.camera = Camera(self.mmk2)  # 相机实例化
            logger.info("✓ 相机实例创建成功")
        except Exception as e:
            logger.error(f"相机实例创建失败: {e}")
            import traceback
            traceback.print_exc()
            self.camera = None

    def reset_start(self, arm_type):
        if arm_type == 'left_arm':
            self.control_trajectory_full(START_LEFT_JOINT_ACTION)
        elif arm_type == 'right_arm':
            self.control_trajectory_full(START_RIGHT_JOINT_ACTION)
        elif arm_type == "all_arm":
            self.control_trajectory_full(START_ALL_JOINT_ACTION)
        else:
            logger.error(f"Invalid arm type: {arm_type}")
        print(f"回到{arm_type}起始姿态，当前状态为：")
        self.printMessage()

    def reset_stop(self):
        self.control_trajectory_full(STOP_JOINT_ACTION)
        # self.set_move_base_zero()
        print("机器人重置，回到双臂停止姿态，当前状态为：")
        self.printMessage()

    @property
    def head_pose(self):
        return self.get_all_joint_states()[0]

    @property
    def base_pose(self):
        return self.get_all_joint_states()[-1]

    @property
    def spine_position(self):
        return self.get_all_joint_states()[1]

    @property
    def left_arm_joints(self):
        return self.get_all_joint_states()[2]

    @property
    def right_arm_joints(self):
        return self.get_all_joint_states()[3]

    @property
    def left_arm_pose(self):
        """返回的是一个列表  [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
        return self.get_all_joint_states()[4]

    @property
    def right_arm_pose(self):
        """返回的是一个列表  [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
        return self.get_all_joint_states()[5]

    def get_robot_current_state(self):
        state = self.mmk2.get_robot_state()
        if not state:
            return None
        
        joint_dict = {
            name: {
                'position': pos,
                'velocity': vel,
                'effort': eff
            } for name, pos, vel, eff in zip(
                state.joint_state.name,
                state.joint_state.position,
                state.joint_state.velocity,
                state.joint_state.effort
            )
        }
        
        base_dict = {
            'x': state.base_state.pose.x,
            'y': state.base_state.pose.y,
            'theta': state.base_state.pose.theta,
        }
        
        left_ee_pose  = state.robot_pose.robot_pose['left_arm']
        right_ee_pose = state.robot_pose.robot_pose['right_arm']
        ee_pose_dict = {
            'left_arm': {
                'position': [left_ee_pose.position.x, left_ee_pose.position.y, left_ee_pose.position.z],
                'orientation': [left_ee_pose.orientation.x, left_ee_pose.orientation.y, left_ee_pose.orientation.z, left_ee_pose.orientation.w]
            },
            'right_arm': {
                'position': [right_ee_pose.position.x, right_ee_pose.position.y, right_ee_pose.position.z],
                'orientation': [right_ee_pose.orientation.x, right_ee_pose.orientation.y, right_ee_pose.orientation.z, right_ee_pose.orientation.w]
            }
        }

        return {
            'joints': joint_dict,
            'base': base_dict,
            'ee_poses': ee_pose_dict
        }

    def get_all_joint_states(self):
        state = self.get_robot_current_state()
        if not state:
            return None
        
        # 初始化关节顺序缓存
        if self._joint_order_cache is None:
            self._init_joint_order_cache(state['joints'])
        
        head = [
            state['joints']['head_yaw_joint']['position'],
            state['joints']['head_pitch_joint']['position']
        ]
        
        lift = state['joints']['slide_joint']['position']
        left_arm  = [state['joints'][f'left_arm_joint{i+1}']['position'] 
                   for i in range(6)]
        right_arm = [state['joints'][f'right_arm_joint{i+1}']['position'] 
                    for i in range(6)]
        
        left_arm_eef  = [self.get_robot_current_state()['ee_poses']['left_arm']['position'],
                         self.get_robot_current_state()['ee_poses']['left_arm']['orientation']]
        right_arm_eef = [self.get_robot_current_state()['ee_poses']['right_arm']['position'],
                         self.get_robot_current_state()['ee_poses']['right_arm']['orientation']]

        left_gripper = [state['joints']['left_arm_eef_gripper_joint']['position']]
        right_gripper = [state['joints']['right_arm_eef_gripper_joint']['position']]

        base_pose = [
            state['base']['x'],
            state['base']['y'],
            state['base']['theta']
        ]
        
        return (head, lift, 
                left_arm, right_arm,
                left_arm_eef, right_arm_eef,
                left_gripper, right_gripper, 
                base_pose)

    def get_base_pose(self):
        base_pos_dict = self.get_robot_current_state()['base']
        pose = [base_pos_dict["x"], base_pos_dict["y"], base_pos_dict["theta"]]
        return pose

    def quaternion_to_rotation_matrix(self, quaternion):
        x, y, z, w = quaternion
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),   2*(x*z + y*w)],
            [2*(x*y + z*w),   1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
        ])

    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数 (x,y,z,w 顺序)"""
        from scipy.spatial.transform import Rotation as R_scipy
        return R_scipy.from_matrix(R).as_quat()

    def get_arm_ee_pose(self, arm_type='left_arm'):
        if arm_type == 'left_arm':
            return self.left_arm_pose        
        elif arm_type == 'right_arm':
            return self.right_arm_pose
        else:
            logger.error(f"Invalid arm type: {arm_type}")
            return None,None

        # ee_pose = state['ee_poses'][arm_type]
        # 顺序是 xyzw
        # rotation_matrix = self.quaternion_to_rotation_matrix(ee_pose['orientation'])
        
        # transform = np.eye(4)
        # transform[:3, :3] = rotation_matrix
        # transform[:3, 3] = ee_pose['position']
        # return transform

    def _init_joint_order_cache(self, joints_dict):
        """初始化关节顺序验证"""
        required_joints = [
            'head_yaw_joint', 'head_pitch_joint',
            'slide_joint',
            *[f'left_arm_joint{i+1}' for i in range(6)],
            *[f'right_arm_joint{i+1}' for i in range(6)],
            'left_arm_eef_gripper_joint', 'right_arm_eef_gripper_joint'
        ]
        
        missing = [j for j in required_joints if j not in joints_dict]
        if missing:
            raise KeyError(f"Missing required joints: {missing}")
        
        self._joint_order_cache = required_joints

    def set_robot_eef(self, eef_type, value):
        """设置机械臂末端执行器状态 0为开 1为关"""
        if eef_type == "left_arm":
            eef_action = {
                MMK2Components.LEFT_ARM_EEF: JointState(position=[value]),
            }
            if(
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
                != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        elif eef_type == "right_arm":
            eef_action = {
                MMK2Components.RIGHT_ARM_EEF: JointState(position=[value]),
            }
            if(
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        elif eef_type == "all_arm":
            eef_action = {
                MMK2Components.LEFT_ARM_EEF:  JointState(position=[value]),
                MMK2Components.RIGHT_ARM_EEF: JointState(position=[value]),
            }
            if (
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
              != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        else:
            logger.error("Invalid eef type")

    def set_robot_spine(self, value=0.0):
        """设置脊柱状态, 正数向下，负数向上"""
        value = max(min(value, 1.0), -0.1)
        print(value)
        spine_action = {
            MMK2Components.SPINE: JointState(position=[value]),
        }
        if (
            self.mmk2.set_goal(spine_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set spine")

    def set_robot_head_pose(self, yaw=0.0, pitch=0.0):
        """设置头部姿态,单位为弧度,yaw为左右,pitch为上下"""
        spine_action = {
            MMK2Components.HEAD: JointState(position=[yaw, pitch]),
        }
        if (
            self.mmk2.set_goal(spine_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set spine")

    def control_robot_arm_pose_updown(self, arm_type='left_arm', updown=0.0):
        """控制机械臂末端沿机器人base坐标系Z轴移动
        Args:
            arm_type: 机械臂类型 left_arm/right_arm
            updown: 移动量（米）正数向下，负数向上
        """
        # 参数校验和类型判断
        if arm_type not in ['left_arm', 'right_arm']:
            logger.error(f"Invalid arm type: {arm_type}")
            return

        current_arm_pose = self.get_arm_ee_pose(arm_type)
        if current_arm_pose is None:
            logger.error("Failed to get arm pose")
            return

        updown = max(min(updown, 0.1), -0.1)
        
        # get_arm_ee_pose 返回 [[x,y,z], [qx,qy,qz,qw]]
        position = current_arm_pose[0].copy()  # [x,y,z]
        quaternion = current_arm_pose[1].copy()  # [qx,qy,qz,qw]
        
        # 修改Z轴位置
        position[2] += updown
        new_quat = quaternion  # 保持姿态不变
        new_pose = Pose(
            position=Position(x=position[0], y=position[1], z=position[2]),
            orientation=Orientation(
                x=new_quat[0],
                y=new_quat[1],
                z=new_quat[2],
                w=new_quat[3]
            )
        )

        component = MMK2Components.LEFT_ARM if arm_type == 'left_arm' else MMK2Components.RIGHT_ARM
        result = self.mmk2.set_goal(
            {component: new_pose},
            TrajectoryParams()
        )
        if result.value != GoalStatus.Status.SUCCESS:
            logger.error(f"Failed to set {arm_type} poses")

    def control_robot_arm_pose_rotate(self, arm_type='left_arm', rotate=0.0):
        """机械臂末端绕机器人base坐标系Z轴旋转
           rotate 正数时逆时针旋转  负数时顺时针旋转
        """
        if arm_type in ['left_arm', 'right_arm']:
            current_arm_pose = self.get_arm_ee_pose(arm_type)
            if current_arm_pose is None:
                logger.error("Failed to get arm pose")
                return
            
            # 限制旋转幅度在±0.1弧度（约±5.7度）以内
            rotate = max(min(rotate, 0.1), -0.1)
            
            # get_arm_ee_pose 返回 [[x,y,z], [qx,qy,qz,qw]]
            position = np.array(current_arm_pose[0])  # [x,y,z]
            quaternion = np.array(current_arm_pose[1])  # [qx,qy,qz,qw]
            
            # 将四元数转换为旋转矩阵
            from scipy.spatial.transform import Rotation as R_scipy
            R_current = R_scipy.from_quat(quaternion).as_matrix()
            
            # 创建绕Z轴旋转矩阵
            R_z = np.array([
                [np.cos(rotate), -np.sin(rotate), 0],
                [np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1]
            ])
            R_new = R_z @ R_current
            new_quat = self.rotation_matrix_to_quaternion(R_new)

            new_pose = Pose(
                position=Position(x=position[0], y=position[1], z=position[2]),
                orientation=Orientation(
                    x=new_quat[0],
                    y=new_quat[1],
                    z=new_quat[2],
                    w=new_quat[3]
                )
            )
            
            component = MMK2Components.LEFT_ARM if arm_type == 'left_arm' else MMK2Components.RIGHT_ARM
            result = self.mmk2.set_goal(
                {component: new_pose},
                TrajectoryParams()
            )
            if result.value != GoalStatus.Status.SUCCESS:
                logger.error(f"Failed to set {arm_type} poses")

    def control_trajectory_full(self, joint_action):
        if (
            self.mmk2.set_goal(
                joint_action, 
                TrajectoryParams(
                    max_velocity_scaling_factor=0.8,
                    max_acceleration_scaling_factor=0.8,
                )
            ).value != GoalStatus.Status.SUCCESS 
        ):
            logger.error("Failed to set goal")
            return False
        # time.sleep(5)
        return True

    def control_arm_joints(self, arm_type, joint_action):
        """控制机械臂关节, arm_type为left或right, joint_action为关节角度列表"""
        if arm_type == "left_arm":
            arm_action = {
                MMK2Components.LEFT_ARM: JointState(position=joint_action),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS 
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "right_arm":
            arm_action = {
                MMK2Components.RIGHT_ARM: JointState(position=joint_action),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "all_arm":
            arm_action = {
                MMK2Components.LEFT_ARM:  JointState(position=joint_action[0]),
                MMK2Components.RIGHT_ARM: JointState(position=joint_action[1]),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        else:
            logger.error("Invalid arm type")
        time.sleep(0.3)

    def _move_arm_single_pose(self, arm_type, target_pose_list):
        """内部方法：使用增量运动逐步接近目标位置
        target_pose_list: [x, y, z, qx, qy, qz, qw] list
        策略：分步移动，每次移动较小距离，保持当前姿态
        """
        logger.info(f"[_move_arm_single_pose] arm_type: {arm_type}")
        logger.info(f"[_move_arm_single_pose] target pose: {target_pose_list}")

        if len(target_pose_list) != 7:
            logger.error(f"位姿格式错误，需要 [x,y,z,qx,qy,qz,qw]，实际长度: {len(target_pose_list)}")
            return False

        # 获取当前末端位姿
        current_ee = self.get_arm_ee_pose(arm_type)
        if current_ee is None:
            logger.error("无法获取当前末端位姿")
            return False
        
        current_position = np.array(current_ee[0])  # [x,y,z]
        target_position = np.array(target_pose_list[:3])
        
        # 计算位置差
        delta = target_position - current_position
        distance = np.linalg.norm(delta)
        
        logger.info(f"当前位置: {current_position}")
        logger.info(f"目标位置: {target_position}")
        logger.info(f"位置差距: {distance:.3f}m")
        
        if distance < 0.01:  # 已经很接近了
            logger.info("已经在目标位置附近")
            return True
        
        # 分步移动策略：每次最多移动5cm
        max_step = 0.05
        num_steps = max(1, int(np.ceil(distance / max_step)))
        
        logger.info(f"将分 {num_steps} 步移动")
        
        component = MMK2Components.LEFT_ARM if arm_type == 'left_arm' else MMK2Components.RIGHT_ARM
        
        for i in range(num_steps):
            # 计算本步的目标位置
            if i == num_steps - 1:
                # 最后一步，直接到目标
                step_target = target_position
            else:
                # 中间步，移动一部分
                ratio = (i + 1) / num_steps
                step_target = current_position + delta * ratio
            
            # 使用当前姿态，只改变位置（Z轴方向）
            logger.info(f"步骤 {i+1}/{num_steps}: 移动到 {step_target}")
            
            # 使用 control_robot_arm_pose_updown（只改变Z轴）
            if i == 0:
                # 第一步：先调整XY
                delta_xy = step_target[:2] - current_position[:2]
                # 简单的XY调整（假设工作空间）
                # 这里暂时跳过XY调整，因为没有直接的API
                pass
            
            # 调整Z轴高度
            delta_z = step_target[2] - current_position[2]
            if abs(delta_z) > 0.005:  # 超过5mm才调整
                try:
                    self.control_robot_arm_pose_updown(arm_type, delta_z / num_steps)
                    time.sleep(0.5)
                    
                    # 更新当前位置
                    current_ee = self.get_arm_ee_pose(arm_type)
                    if current_ee:
                        current_position = np.array(current_ee[0])
                except Exception as e:
                    logger.error(f"步骤 {i+1} 失败: {e}")
                    return False
        
        logger.info("✓ 移动完成")
        return True

    def control_arm_pose(self, arm_type, pose_base):
        """控制机械臂末端移动:
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入pose:  [X,Y,Z,qx,qy,qz,qw]
            双臂输入pose:  [left,right]
        """
        if arm_type == "left_arm":
            arm_ee_pose = {
                MMK2Components.LEFT_ARM: Pose(
                    position=Position(x=pose_base[0], y=pose_base[1], z=pose_base[2]),
                    orientation=Orientation(x=pose_base[3], y=pose_base[4], 
                                            z=pose_base[5], w=pose_base[6]),
                )
            }
        elif arm_type == "right_arm":
            arm_ee_pose = {
                MMK2Components.RIGHT_ARM: Pose(
                    position=Position(x=pose_base[0], y=pose_base[1], z=pose_base[2]),
                    orientation=Orientation(x=pose_base[3], y=pose_base[4], 
                                            z=pose_base[5], w=pose_base[6]),
                )
            }
        elif arm_type == "all_arm":
            arm_ee_pose = {
                MMK2Components.LEFT_ARM: Pose(
                    position=Position(x=pose_base[0][0], y=pose_base[0][1], z=pose_base[0][2]),
                    orientation=Orientation(x=pose_base[0][3], y=pose_base[0][4], 
                                            z=pose_base[0][5], w=pose_base[0][6]),
                ),
                MMK2Components.RIGHT_ARM: Pose(
                    position=Position(x=pose_base[1][0], y=pose_base[1][1], z=pose_base[1][2]),
                    orientation=Orientation(x=pose_base[1][3], y=pose_base[1][4], 
                                            z=pose_base[1][5], w=pose_base[1][6]),
                )
            }
        else:
            logger.error("Invalid arm type")
            return
        
        if (
            self.mmk2.set_goal(arm_ee_pose, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

        time.sleep(0.1)

    def control_arm_pose_waypoints(self, arm_type, waypoints):
        """控制机械臂末端移动，输入轨迹点基于机器人base坐标系: 
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入waypoints:  [[X,Y,Z,qx,qy,qz,qw],...]
            双臂输入waypoints:  [left_waypoints,right_waypoints]
        """
        pose_sequence = []
        if arm_type in ['left_arm', 'right_arm']:
            for wp in waypoints:
                if len(wp) != 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")
                    
                pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))
        elif arm_type == "all_arm":
            left_pose_sequence = []
            for wp in waypoints[0]:
                if len(wp)!= 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")

                left_pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))

            right_pose_sequence = []
            for wp in waypoints[1]:
                if len(wp)!= 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")

                right_pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))
        else:
            logger.error("Invalid arm type")

        if arm_type == "left_arm":
            arm_ee_pose = {MMK2Components.LEFT_ARM: pose_sequence}
        elif arm_type == "right_arm":
            arm_ee_pose = {MMK2Components.RIGHT_ARM: pose_sequence}
        elif arm_type == "all_arm":
            arm_ee_pose = {
                    MMK2Components.LEFT_ARM: left_pose_sequence,
                    MMK2Components.RIGHT_ARM: right_pose_sequence,
                }
        else:
            logger.error("Invalid arm type")
        
        if (
            self.mmk2.set_goal(arm_ee_pose, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

        # time.sleep(2)

    def control_traj_servo_separate(self, joint_action):
        freq = 20
        time_sec = 5
        action_ref = joint_action.copy()
        if (
            self.mmk2.set_goal(
                {MMK2Components.SPINE: action_ref.pop(MMK2Components.SPINE)},
                TrajectoryParams(),
            ).value
            == GoalStatus.Status.SUCCESS
        ):
            for _ in range(freq * time_sec):
                start = time.time()
                if (
                    self.mmk2.set_goal(action_ref, MoveServoParams()).value
                    != GoalStatus.Status.SUCCESS
                ):
                    logger.error("Failed to set goal")
                time.sleep(max(0, 1 / freq - (time.time() - start)))
        else:
            logger.error("Failed to move spine")

    def control_move_base_pose(self, base_pose=[0,0,0]):
        # move the robot base to zero pose
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Pose3D(x=base_pose[0], y=base_pose[1], theta=base_pose[2])},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_robot_move_base_updown(self, updown=0.0):
        """控制底盘相对移动
        Args:
            updown: 正数向前 负数向后 最小分辨率0.1m
        """
        base_pose = self.get_all_joint_states()[-1]
        base_pose[0] +=  updown
        print(base_pose)
        self.control_move_base_pose(base_pose)

    def control_rotate_base(self, rotate_rad=-0.5):
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: rotate_rad},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_move_base_velocity(self, velocity=[0,0]):
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Twist3D(x=velocity[0], omega=velocity[1])},
                BaseControlParams(),
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_build_map(self):
        build_map_param = BuildMapParams()
        if (
            self.mmk2.set_goal(
                None,
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

        # move the base around to build the map and then
        # set current pose to zero pose in the map
        if (
            self.mmk2.set_goal(
                Pose3D(x=0, y=0, theta=0),
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

        # stop build map
        build_map_param.stop = True
        if (
            self.mmk2.set_goal(
                None,
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def set_move_base_zero(self):
        build_map_param = BuildMapParams()
        if (
            self.mmk2.set_goal(
                Pose3D(x=0, y=0, theta=0),
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")
        time.sleep(3)

    def printMessage(self):
        print("-" * 50)
        head, lift, left_arm, right_arm, left_arm_eef, right_arm_eef, left_gripper, right_gripper, xy_yaw = self.get_all_joint_states()
        print("real_robot.get_all_joint_states:")
        print("head                         : ", np.array(head))
        print("spine positon                : ", np.array(self.spine_position))
        print("left_arm_joints              : ", np.array(self.left_arm_joints))
        print("right_arm_joints             : ", np.array(self.right_arm_joints))
        print("left_arm_eef_pos             : ", np.array(left_arm_eef[0]))
        print("left_arm_eef_orientation     : ", np.array(left_arm_eef[1]))
        print("right_arm_eef_pos            : ", np.array(right_arm_eef[0]))
        print("right_arm_eef_orientation    : ", np.array(right_arm_eef[1]))
        print("left_gripper  : ", np.array(left_gripper))
        print("right_gripper : ", np.array(right_gripper))
        print("xy_yaw        : ", np.array(xy_yaw))

    def control_arm_joint_waypoints(self, arm_type, waypoints):
        """控制机械臂末端移动:
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入waypoints:  [[0,1,2,3,4,5,6],...]
            双臂输入waypoints:  [left_waypoints,right_waypoints]
        """
        joint_sequence = []
        if arm_type in ['left_arm', 'right_arm']:
            for wp in waypoints:
                if len(wp) != 6:
                    raise ValueError("关节角度需要6个参数")
                joint_sequence.append(JointState(position=wp))
        elif arm_type == "all_arm":
            left_joint_sequence = []
            for wp in waypoints[0]:
                if len(wp)!= 6:
                    raise ValueError("关节角度需要6个参数")
                left_joint_sequence.append(JointState(position=wp))
            right_joint_sequence = []
            for wp in waypoints[1]:
                if len(wp)!= 6:
                    raise ValueError("关节角度需要6个参数")
                right_joint_sequence.append(JointState(position=wp))
        else:
            logger.error("Invalid arm type")

        if arm_type == "left_arm":
            arm_joints = {MMK2Components.LEFT_ARM: joint_sequence}
        elif arm_type == "right_arm":
            arm_joints = {MMK2Components.RIGHT_ARM: joint_sequence}
        elif arm_type == "all_arm":
            arm_joints = {
                    MMK2Components.LEFT_ARM: left_joint_sequence,
                    MMK2Components.RIGHT_ARM: right_joint_sequence,
                }
        else:
            logger.error("Invalid arm type")
        
        if (
            self.mmk2.set_goal(arm_joints, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

    def grasp_object_from_6d_pose(self, pose_file, arm_type='left_arm', 
                                   approach_distance=0.1, lift_height=0.1):
        """
        根据6D姿态文件执行物体抓取
        
        Args:
            pose_file: 6D姿态文件路径 (4x4变换矩阵，相机坐标系)
            arm_type: 使用的机械臂 'left_arm' 或 'right_arm'
            approach_distance: 接近距离(米)
            lift_height: 抬起高度(米)
        
        Returns:
            bool: 抓取是否成功
        """
        logger.info("\n" + "="*50)
        logger.info("开始物体抓取流程")
        logger.info("="*50)
        
        # 1. 读取6D姿态
        logger.info(f"读取姿态文件: {pose_file}")
        if not os.path.exists(pose_file):
            logger.error(f"姿态文件不存在: {pose_file}")
            return False
        
        try:
            pose_in_camera = np.loadtxt(pose_file)
            if pose_in_camera.shape != (4, 4):
                logger.error(f"姿态文件格式错误，应为4x4矩阵，实际为{pose_in_camera.shape}")
                return False
            logger.info(f"物体在相机坐标系中的姿态:\n{pose_in_camera}")
        except Exception as e:
            logger.error(f"读取姿态文件失败: {e}")
            return False
        
        # 2. 相机到机器人基座的外参矩阵 (头部 pitch = -45°)
        # 通过 calculate_T_with_pitch.py 计算得到
        T_base_camera = np.array([
            [-0.45989579185046614, -0.16153130056927642, 0.873108436576856, 0.20173213411196772],
            [-0.707224277384007, 0.6612149304111493, -0.2501492658940429, -0.37988025659236263],
            [-0.5369633899223504, -0.7325260778551006, -0.4183288213247916, 1.468571383360286],
            [0.0, 0.0, 0.0, 1.0],
        ])
        logger.info(f"相机到基座的变换矩阵:\n{T_base_camera}")
        
        # 3. 坐标变换: 物体在机器人基座坐标系中的姿态
        T_base_object = T_base_camera @ pose_in_camera
        logger.info(f"物体在基座坐标系中的姿态:\n{T_base_object}")
        
        # 提取位置和旋转
        obj_position = T_base_object[:3, 3]
        obj_rotation = T_base_object[:3, :3]
        
        # 转换为四元数 (x, y, z, w)
        from scipy.spatial.transform import Rotation
        quat_obj = Rotation.from_matrix(obj_rotation).as_quat()
        
        logger.info(f"物体位置 [X,Y,Z]: {obj_position}")
        logger.info(f"物体四元数 [qx,qy,qz,qw]: {quat_obj}")
        
        # 4. 计算抓取姿态（末端执行器朝向）
        # 机器人末端执行器需要从上方接近物体
        # 创建一个朝下的姿态 (末端执行器Z轴指向下)
        # 四元数 [1, 0, 0, 0] 表示绕X轴旋转180度（朝下）
        quat_grasp = np.array([1.0, 0.0, 0.0, 0.0])
        
        logger.info(f"抓取四元数 [qx,qy,qz,qw]: {quat_grasp}")
        
        # 5. 计算抓取轨迹点
        logger.info("\n计算抓取轨迹...")
        
        # Pre-grasp: 物体上方 approach_distance 处
        pre_grasp_position = obj_position.copy()
        pre_grasp_position[2] += approach_distance
        
        # Grasp: 物体上方 20mm 处（实际接触点）
        grasp_position = obj_position.copy()
        grasp_position[2] -= 0  # 精确抓取
        
        # Post-grasp: 抬起 lift_height
        post_grasp_position = pre_grasp_position.copy()
        post_grasp_position[2] += lift_height

        # 校验位置偏差
        if np.linalg.norm(grasp_position) > 2.0 or grasp_position[2] < 0.5:
            logger.error("抓取位置偏差太大，跳过")
            return False
        
        # 6. 构造轨迹点 [X, Y, Z, qx, qy, qz, qw]
        waypoint_pre_grasp = [
            pre_grasp_position[0], pre_grasp_position[1], pre_grasp_position[2],
            quat_grasp[0], quat_grasp[1], quat_grasp[2], quat_grasp[3]
        ]
        waypoint_grasp = [
            grasp_position[0], grasp_position[1], grasp_position[2],
            quat_grasp[0], quat_grasp[1], quat_grasp[2], quat_grasp[3]
        ]
        waypoint_post_grasp = [
            post_grasp_position[0], post_grasp_position[1], post_grasp_position[2],
            quat_grasp[0], quat_grasp[1], quat_grasp[2], quat_grasp[3]
        ]
        
        # 验证waypoint
        for i, wp in enumerate([waypoint_pre_grasp, waypoint_grasp, waypoint_post_grasp], 1):
            if any(np.isnan(wp)) or any(np.isinf(wp)):
                logger.error(f"无效的waypoint#{i}: {wp}")
                return False
            logger.info(f"Waypoint#{i}: {wp}")
        
        # 7. 执行抓取序列 - 使用单个pose移动
        try:
            logger.info("\n开始执行抓取...")
            
            # 7.1 打开夹爪
            logger.info("1. 打开夹爪")
            self.set_robot_eef(arm_type, 0.0)
            time.sleep(1.0)
            
            # 7.2 移动到pre-grasp位置
            logger.info("2. 移动到pre-grasp位置")
            if not self._move_arm_single_pose(arm_type, waypoint_pre_grasp):
                logger.error("移动到pre-grasp位置失败")
                return False
            
            # 7.3 缓慢接近到grasp位置
            logger.info("3. 接近到grasp位置")
            if not self._move_arm_single_pose(arm_type, waypoint_grasp):
                logger.error("移动到grasp位置失败")
                return False
            
            # 7.4 闭合夹爪
            logger.info("4. 闭合夹爪")
            self.set_robot_eef(arm_type, 1.0)
            time.sleep(1.5)
            
            # 7.5 抬起
            logger.info("5. 抬起物体")
            if not self._move_arm_single_pose(arm_type, waypoint_post_grasp):
                logger.error("抬起物体失败")
                return False
            
            logger.info("\n✓ 抓取完成！")
            return True
            
        except Exception as e:
            logger.error(f"抓取失败: {e}")
            import traceback
            traceback.print_exc()
            return False


class Camera:
    """相机迭代器类"""
    def __init__(self, mmk2_handler):
        self.mmk2 = mmk2_handler
        self._init_camera()
        # 只请求 HEAD_CAMERA 的 COLOR 和 DEPTH
        self.image_goal = {
            MMK2Components.HEAD_CAMERA: [
                ImageTypes.COLOR,
                ImageTypes.ALIGNED_DEPTH_TO_COLOR,
            ],
            MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
            MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
        }
        time.sleep(5)  # 等待相机初始化
        
    def _init_camera(self):
        """初始化相机硬件"""
        logger.info("正在初始化相机...")
        try:
            result = self.mmk2.enable_resources({
                MMK2Components.HEAD_CAMERA: {
                    # 640×480分辨率（昨天准确的配置）
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "true",
                    "depth_module.depth_profile": "640,480,30",
                    "align_depth.enable": "true",
                },
            })
            
            if MMK2Components.OTHER in result:
                logger.error(f"相机初始化失败: {result[MMK2Components.OTHER]}")
                raise RuntimeError(f"Camera init failed: {result[MMK2Components.OTHER]}")
            
            logger.info("✓ 相机初始化成功")
            logger.info(f"初始化结果: {result}")
            
        except Exception as e:
            logger.error(f"相机初始化异常: {e}")
            raise
    
    def __iter__(self):
        return self
        
    def __next__(self):
        """获取最新图像数据"""
        try:
            comp_images = self.mmk2.get_image(self.image_goal)
            
            img_head = None
            img_depth = None
            img_left = None 
            img_right = None

            for comp, images in comp_images.items():
                for img_type, img in images.data.items():
                    if comp == MMK2Components.HEAD_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            # 确保是640x480
                            if img.shape[:2] != (480, 640):
                                img_head = cv2.resize(img, (640, 480))
                            else:
                                img_head = img
                        elif img_type == ImageTypes.ALIGNED_DEPTH_TO_COLOR:
                            # 确保是640x480
                            if img.shape[:2] != (480, 640):
                                img_depth = cv2.resize(img, (640, 480))
                            else:
                                img_depth = img
                    elif comp == MMK2Components.LEFT_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            img_left = img
                    elif comp == MMK2Components.RIGHT_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            img_right = img
            
            # 调试：如果图像为None，记录一次
            if img_head is None or img_depth is None:
                if not hasattr(self, '_warned_empty'):
                    logger.warning(f"获取图像失败: RGB={img_head is not None}, Depth={img_depth is not None}")
                    logger.warning(f"comp_images keys: {list(comp_images.keys())}")
                    for comp, images in comp_images.items():
                        logger.warning(f"  {comp}: {list(images.data.keys())}")
                    self._warned_empty = True
            
            return img_head, img_depth, img_left, img_right
            
        except Exception as e:
            logger.error(f"get_image 异常: {e}")
            return None, None, None, None


# class Camera:
#     """相机迭代器类"""
#     def __init__(self, mmk2_handler):
#         self.mmk2 = mmk2_handler
#         self._init_camera()
#         # 只请求 HEAD_CAMERA 的 COLOR 和 DEPTH
#         self.image_goal = {
#             MMK2Components.HEAD_CAMERA: [
#                 ImageTypes.COLOR,
#                 ImageTypes.ALIGNED_DEPTH_TO_COLOR,
#             ],
#             MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
#             MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
#         }
#         time.sleep(5)  # 等待相机初始化

#     def _init_camera(self):
#         """初始化相机硬件"""
#         logger.info("正在初始化相机...")
#         try:
#             result = self.mmk2.enable_resources({
#                 MMK2Components.HEAD_CAMERA: {
#                     # 设置分辨率为 640×480 (匹配K矩阵)
#                     "rgb_camera.color_profile": "640,480,30",
#                     "enable_depth": "true",
#                     "depth_module.depth_profile": "640,480,30",
#                     "align_depth.enable": "true",
#                 },
#             })
            
#             if MMK2Components.OTHER in result:
#                 logger.error(f"相机初始化失败: {result[MMK2Components.OTHER]}")
#                 raise RuntimeError(f"Camera init failed: {result[MMK2Components.OTHER]}")
            
#             logger.info("✓ 相机初始化成功")
#             logger.info(f"初始化结果: {result}")
            
#         except Exception as e:
#             logger.error(f"相机初始化异常: {e}")
#             raise
        
#     def __iter__(self):
#         return self
        
#     def __next__(self):
#         """获取最新图像数据"""
#         try:
#             comp_images = self.mmk2.get_image(self.image_goal)
            
#             img_head = None
#             img_depth = None
#             img_left = None 
#             img_right = None

#             for comp, images in comp_images.items():
#                 for img_type, img in images.data.items():
#                     if comp == MMK2Components.HEAD_CAMERA:
#                         if img_type == ImageTypes.COLOR:
#                             img_head = img
#                         elif img_type == ImageTypes.ALIGNED_DEPTH_TO_COLOR:
#                             img_depth = img
#                     elif comp == MMK2Components.LEFT_CAMERA:
#                         if img_type == ImageTypes.COLOR:
#                             img_left = img
#                     elif comp == MMK2Components.RIGHT_CAMERA:
#                         if img_type == ImageTypes.COLOR:
#                             img_right = img
            
#             # 调试：如果图像为None，记录一次
#             if img_head is None or img_depth is None:
#                 if not hasattr(self, '_warned_empty'):
#                     logger.warning(f"获取图像失败: RGB={img_head is not None}, Depth={img_depth is not None}")
#                     logger.warning(f"comp_images keys: {list(comp_images.keys())}")
#                     for comp, images in comp_images.items():
#                         logger.warning(f"  {comp}: {list(images.data.keys())}")
#                     self._warned_empty = True
            
#             return img_head, img_depth, img_left, img_right
                
#         except Exception as e:
#             logger.error(f"get_image 异常: {e}")
#             return None, None, None, None



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # mmk2 = MMK2RealRobot()
    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    mmk2.reset_start("right_arm")
    mmk2.reset_start("left_arm")
    # mmk2.set_robot_head_pose(yaw=0, pitch=-1.08)
    # mmk2.set_robot_head_pose(yaw=0, pitch=0.16)
    mmk2.set_robot_head_pose(yaw=0, pitch=0)
    # mmk2.reset_stop()
    # mmk2.set_move_base_zero()
    mmk2.printMessage()

    

