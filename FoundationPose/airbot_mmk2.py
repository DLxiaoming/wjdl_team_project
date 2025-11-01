'''
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-03-10 16:07:23
LastEditTime: 2025-03-13 18:16:03
'''

from mmk2_types.types import RobotComponents as MMK2Components, ImageTypes
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
)
from airbot_py.airbot_mmk2 import AirbotMMK2
import logging
import time
import cv2
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

class MMK2RealRobot():
    def __init__(self, ip="172.25.14.1"):
        self.mmk2 = AirbotMMK2(ip)
        self._joint_order_cache = None
        self.camera = self.Camera(self.mmk2)

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
        """返回 list [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
        return self.get_all_joint_states()[4]

    @property
    def right_arm_pose(self):
        """返回 list [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
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

    class Camera:
        """相机迭代器类"""
        def __init__(self, mmk2_handler):
            self.mmk2 = mmk2_handler
            self._init_camera()
            self.image_goal = {
                MMK2Components.HEAD_CAMERA: [
                    ImageTypes.COLOR,
                    ImageTypes.ALIGNED_DEPTH_TO_COLOR,
                ],
                MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
                MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
            }
            time.sleep(3)  # 等待相机初始化
            
        def _init_camera(self):
            """初始化相机硬件"""
            result = self.mmk2.enable_resources({
                MMK2Components.HEAD_CAMERA: {
                    # "serial_no": "'242622071873'",
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "true",
                    "depth_module.depth_profile": "640,480,30",
                    "align_depth.enable": "true",
                },
                # MMK2Components.LEFT_CAMERA: {
                #     "serial_no": "'327122077893'",
                #     "rgb_camera.color_profile": "1280,720,15",
                #     "enable_depth": "false",
                # },
                # MMK2Components.RIGHT_CAMERA: {
                #     "serial_no": "'327122076989'",
                #     "rgb_camera.color_profile": "1280,720,15",
                #     "enable_depth": "false",
                # },
            })
            assert MMK2Components.OTHER not in result, f"Camera init failed: {result[MMK2Components.OTHER]}"
        
        def __iter__(self):
            return self
            
        def __next__(self):
            """获取最新图像数据"""
            comp_images = self.mmk2.get_image(self.image_goal)
            
            img_head = None
            img_depth = None
            img_left = None 
            img_right = None

            for comp, images in comp_images.items():
                for img_type, img in images.data.items():
                    if img.shape[0] == 1:
                        continue
                    if comp == MMK2Components.HEAD_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            # Resize到640x480，匹配内参K
                            img_head = cv2.resize(img, (640, 480))
                        elif img_type == ImageTypes.ALIGNED_DEPTH_TO_COLOR:
                            # Resize到640x480，匹配内参K
                            img_depth = cv2.resize(img, (640, 480))
                    elif comp == MMK2Components.LEFT_CAMERA:
                        img_left = img
                    elif comp == MMK2Components.RIGHT_CAMERA:
                        img_right = img
            
            return img_head, img_depth, img_left, img_right

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

    def control_trajectory_full(self, joint_action):
        if (
            self.mmk2.set_goal(joint_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
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
                self.mmk2.set_goal(arm_action, TrajectoryParams()).value
               != GoalStatus.Status.SUCCESS 
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "right_arm":
            arm_action = {
                MMK2Components.RIGHT_ARM: JointState(position=joint_action),
            }
            if (
                self.mmk2.set_goal(arm_action, TrajectoryParams()).value
              != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "all_arm":
            arm_action = {
                MMK2Components.LEFT_ARM:  JointState(position=joint_action[0]),
                MMK2Components.RIGHT_ARM: JointState(position=joint_action[1]),
            }
            if (
                self.mmk2.set_goal(arm_action, TrajectoryParams()).value
              != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        else:
            logger.error("Invalid arm type")
        time.sleep(0.3)

    def control_arm_pose(self, arm_type, pose_base, slow_mode=False):
        """控制机械臂末端移动:
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入pose:  [X,Y,Z,qx,qy,qz,qw]
            双臂输入pose:  [left,right]
            slow_mode: 是否使用慢速模式（用于精细操作）
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
            return False
        
        # 设置运动参数
        if slow_mode:
            params = TrajectoryParams(
                max_velocity_scaling_factor=0.2,
                max_acceleration_scaling_factor=0.2,
            )
        else:
            params = TrajectoryParams(
                max_velocity_scaling_factor=0.5,
                max_acceleration_scaling_factor=0.5,
            )
        
        result = self.mmk2.set_goal(arm_ee_pose, params)
        if result.value != GoalStatus.Status.SUCCESS:
            logger.error(f"Failed to control_arm_pose, status: {result.value}")
            return False
        
        return True

    def control_arm_pose_waypoints(self, arm_type, waypoints):
        """控制机械臂末端移动:
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
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Pose3D(x=base_pose[0], y=base_pose[1], theta=base_pose[2])},
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
                                    approach_distance=0.15, 
                                    lift_height=0.15):
        """
        从FoundationPose的6D姿态文件抓取物体
        
        参数:
            pose_file: 姿态文件路径 (例如: 'debug/ob_in_cam/pose_first_frame.txt')
            arm_type: 使用的机械臂 'left_arm' 或 'right_arm'
            approach_distance: 抓取前接近距离（米）
            lift_height: 抓取后抬升高度（米）
        
        返回:
            bool: 抓取是否成功
        """
        logger.info("\n" + "="*60)
        logger.info("开始执行抓取流程")
        logger.info("="*60)
        
        try:
            # ========================================
            # 1. 读取相机坐标系下的6D姿态
            # ========================================
            logger.info(f"读取姿态文件: {pose_file}")
            pose_in_camera = np.loadtxt(pose_file)
            if pose_in_camera.shape != (4, 4):
                logger.error(f"姿态矩阵格式错误: {pose_in_camera.shape}")
                return False
            
            logger.info("物体在相机坐标系下的姿态:")
            logger.info(f"\n{pose_in_camera}")
            logger.info(f"位置: {pose_in_camera[:3, 3]}")
            
            # ========================================
            # 2. 相机到机器人基坐标系的转换
            # ========================================
            # 使用ROS2 TF系统提供的变换矩阵（从 script/tf_base_to_head_camera.md）
            # 这是机器人自己通过TF系统提供的准确变换关系，不需要手动标定！
            # 
            # 来源: ros2 run tf2_ros tf2_echo base_link head_camera_link
            # Translation: [0.283, 0.036, 1.578] (单位：米)
            # Rotation: Quaternion [-0.575, 0.575, -0.411, 0.411]
            #
            # 这个变换矩阵会随着头部运动而动态变化，
            # 但对于固定头部位置的抓取任务，可以使用当前值
            
            logger.info("\n使用ROS2 TF系统的 T_base_camera (base_link → head_camera_link)")
            
            T_base_camera = np.array([
                [-0.000, -0.324,  0.946, 0.283],
                [-1.000,  0.000, -0.000, 0.036],
                [-0.000, -0.946, -0.324, 1.578],
                [ 0.000,  0.000,  0.000, 1.000]
            ])
            
            # 如果需要动态获取当前头部位置的变换，可以通过ROS2 TF API
            # 或者根据头部关节角度计算正运动学
            # 当前使用固定值足够用于单次抓取任务
            
            logger.info("基座到相机的变换矩阵 T_base_camera:")
            logger.info(f"\n{T_base_camera}")
            
            # 物体在机器人基坐标系下的姿态
            T_base_object = T_base_camera @ pose_in_camera
            
            logger.info("\n物体在机器人基坐标系下的姿态:")
            logger.info(f"\n{T_base_object}")
            logger.info(f"位置: {T_base_object[:3, 3]}")
            
            # ========================================
            # 3. 计算抓取位姿
            # ========================================
            logger.info("\n计算抓取轨迹...")
            
            # 物体位置和方向
            obj_position = T_base_object[:3, 3]
            obj_rotation = T_base_object[:3, :3]
            
            # 抓取方向：从上往下抓（Z轴负方向）
            grasp_approach_direction = np.array([0, 0, -1])
            
            # 末端执行器的抓取姿态（夹爪朝下）
            R_grasp = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
            
            # Pre-grasp位置（物体上方）
            pre_grasp_position = obj_position + grasp_approach_direction * (-approach_distance)
            
            # Grasp位置（物体位置，稍微下压一点）
            grasp_position = obj_position.copy()
            grasp_position[2] -= 0.02  # 下压2cm
            
            # Post-grasp位置（抓取后抬起）
            post_grasp_position = grasp_position + np.array([0, 0, lift_height])
            
            # 转换为四元数 (x,y,z,w)
            quat_grasp = Rotation.from_matrix(R_grasp).as_quat()
            
            # 构建轨迹点 [X,Y,Z,qx,qy,qz,qw]
            waypoint_pre_grasp = list(pre_grasp_position) + list(quat_grasp)
            waypoint_grasp = list(grasp_position) + list(quat_grasp)
            waypoint_post_grasp = list(post_grasp_position) + list(quat_grasp)
            
            # 验证waypoint有效性
            for i, wp in enumerate([waypoint_pre_grasp, waypoint_grasp, waypoint_post_grasp]):
                if any(np.isnan(wp)) or any(np.isinf(wp)):
                    logger.error(f"无效的waypoint#{i}: {wp}")
                    return False
            
            logger.info(f"Pre-grasp位置: {pre_grasp_position}")
            logger.info(f"Grasp位置: {grasp_position}")
            logger.info(f"Post-grasp位置: {post_grasp_position}")
            logger.info(f"Waypoint quaternion: {quat_grasp}")
            
            # ========================================
            # 4. 执行抓取运动序列
            # ========================================
            logger.info("\n开始执行运动...")
            
            # 4.1 打开夹爪
            logger.info("1. 打开夹爪")
            self.set_robot_eef(arm_type, 0.0)  # 0=打开
            time.sleep(1.0)
            
            # 4.2 移动到pre-grasp位置
            logger.info("2. 移动到pre-grasp位置")
            logger.info(f"   Waypoint: {waypoint_pre_grasp}")
            if not self.control_arm_pose(arm_type, waypoint_pre_grasp, slow_mode=False):
                logger.error("移动到pre-grasp位置失败")
                return False
            time.sleep(3.0)
            
            # 4.3 接近物体（慢速）
            logger.info("3. 接近物体")
            logger.info(f"   Waypoint: {waypoint_grasp}")
            if not self.control_arm_pose(arm_type, waypoint_grasp, slow_mode=True):
                logger.error("接近物体失败")
                return False
            time.sleep(2.0)
            
            # 4.4 闭合夹爪
            logger.info("4. 闭合夹爪")
            self.set_robot_eef(arm_type, 1.0)  # 1=闭合
            time.sleep(1.5)
            
            # 4.5 抬起物体
            logger.info("5. 抬起物体")
            logger.info(f"   Waypoint: {waypoint_post_grasp}")
            if not self.control_arm_pose(arm_type, waypoint_post_grasp, slow_mode=False):
                logger.error("抬起物体失败")
                return False
            time.sleep(2.0)
            
            logger.info("\n" + "="*60)
            logger.info("✓ 抓取流程完成！")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"抓取失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def calibrate_camera_to_base_transform(self, checkerboard_size=(7, 10), 
                                           square_size=0.025):
        """
        标定相机到机器人基坐标系的变换
        
        参数:
            checkerboard_size: 棋盘格内角点数量 (行, 列)
            square_size: 棋盘格方块边长（米）
        
        返回:
            T_base_camera: 4x4变换矩阵
        
        说明:
            1. 将棋盘格放在机器人工作空间内
            2. 用相机拍摄棋盘格
            3. 用机械臂末端触碰棋盘格的几个角点
            4. 通过手眼标定计算变换矩阵
        """
        logger.info("\n开始相机标定流程...")
        logger.info("请按照提示操作:")
        logger.info("1. 将标定板放置在机器人可见范围内")
        logger.info("2. 确保相机能清晰看到标定板")
        logger.info("3. 记录机械臂末端位置")
        
        # TODO: 实现完整的手眼标定流程
        # 这里只是提供接口框架
        
        logger.warning("标定功能尚未实现，请手动标定后更新 T_base_camera")
        return None

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    mmk2 = MMK2RealRobot()
    # mmk2.reset_start("all_arm")
    # mmk2.reset_stop()
    mmk2.set_move_base_zero()
    mmk2.printMessage()