
import numpy as np

from robot.mmk2_robot_sdk import MMK2RealRobot

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # mmk2 = MMK2RealRobot()
    mmk2 = MMK2RealRobot(ip="192.168.11.200")

    all_joint_states = mmk2.get_all_joint_states()
    (   head, 
        lift, 
        left_arm, 
        right_arm, 
        left_arm_eef, 
        right_arm_eef, 
        left_gripper, 
        right_gripper, 
        base_pose
    ) = all_joint_states

    print("head: ", head)
    print("lift: ", lift)
    print("left_arm: ", left_arm)
    print("right_arm: ", right_arm)
    print("left_arm_eef: ", left_arm_eef)
    print("right_arm_eef: ", right_arm_eef)
    print("left_gripper: ", left_gripper)
    print("right_gripper: ", right_gripper)
    print("base_pose: ", base_pose)

    joint = [0.12942931592464447, -0.6807430982589722, 0.9195467829704285, -1.7511634826660156, 1.027885913848877, 0.134470134973526]
    mmk2.control_arm_joints('right_arm', joint)