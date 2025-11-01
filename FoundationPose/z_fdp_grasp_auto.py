#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDP自动检测抓取脚本
- 使用FDP检测获取物体xyz位置（相机坐标系）
- 自动执行抓取流程（坐标变换 + 抓取动作）
- 保持operator_process_1029.py的矩阵和参数不变
"""

import os
import sys
import logging
import time
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
import trimesh
import nvdiffrast.torch as dr

from estimater import *
from Utils import *
from last import MMK2RealRobot

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def transform_point(point_camera, transform_matrix):
    """
    将点从相机坐标系转换到基座坐标系
    :param point_camera: [x, y, z] in camera frame
    :param transform_matrix: 4x4 matrix
    :return: [x, y, z] in base_link frame
    """
    # 转成齐次坐标
    p_cam = np.array([*point_camera, 1.0])
    # 计算变换
    p_base = transform_matrix @ p_cam
    # 返回前三个分量
    return p_base[:3]

def main():
    import argparse
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='FDP自动检测抓取')
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--robot_ip', type=str, default='192.168.11.200')
    parser.add_argument('--mesh_file', type=str,
                       default=f'{code_dir}/demo_data/tube/mesh/1.obj')
    parser.add_argument('--yolo_model', type=str,
                       default=f'{code_dir}/10_30best.pt')
    
    # FDP参数
    parser.add_argument('--est_refine_iter', type=int, default=10,
                       help='FDP注册迭代次数')
    parser.add_argument('--head_pitch', type=float, default=-0.5236,
                       help='头部俯仰角度（弧度），默认-30度')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🤖 FDP自动检测抓取系统")
    print("="*70)
    print(f"机器人IP: {args.robot_ip}")
    print(f"头部角度: {np.degrees(args.head_pitch):.1f}°")
    print("="*70 + "\n")
    
    # ============ 1. 初始化机器人 ============
    logging.info("初始化机器人...")
    mmk2 = MMK2RealRobot(ip=args.robot_ip)
    time.sleep(2.0)
    
    # 设置头部角度
    logging.info(f"设置头部角度: {np.degrees(args.head_pitch):.1f}°")
    mmk2.set_robot_head_pose(0.0, args.head_pitch)
    time.sleep(1.0)
    
    # ============ 2. 加载模型 ============
    logging.info("加载YOLO模型...")
    yolo_model = YOLO(args.yolo_model)
    
    logging.info("加载FoundationPose...")
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    debug_dir = f'{code_dir}/debug'
    os.makedirs(debug_dir, exist_ok=True)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                        mesh=mesh, scorer=scorer, refiner=refiner, 
                        glctx=glctx, debug_dir=debug_dir, debug=1)
    
    logging.info("✓ 模型加载完成")
    
    # 相机内参
    K = np.array([[601.87, 0, 321.05], [0, 601.87, 252.46], [0, 0, 1]])
    camera = mmk2.camera
    time.sleep(2.0)
    
    # 创建显示窗口
    cv2.namedWindow('FDP Auto Grasp - Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FDP Auto Grasp - Detection', 640, 480)
    
    # ============ 3. 主循环：检测和抓取 ============
    print("\n" + "="*70)
    print("🔍 FDP自动检测抓取系统")
    print("="*70)
    print("提示: 按 'g' 开始抓取, 按 'r' 重新检测, 按 'q' 退出")
    print("="*70 + "\n")
    
    # 初始位置和抓取姿态（保持不变）
    init_pose = [0.4798196384291117, 0.050344892205700036, 1.3300944085789266,
                 0.004922124603778732, -0.003476176971870248, -0.6913608163682425, 0.7224845399547885]
    
    grasp_rot = [0.09538087446650391, 0.019704521010289917, -0.5310926713809747, 0.8416975674452077]
    
    # 变换矩阵（保持不变）
    T_head_to_base = np.array([
        [-0.001, -0.749, 0.662, 0.365],
        [-1.000, 0.001, -0.001, 0.036],
        [-0.000, -0.662, -0.749, 1.516],
        [0.000, 0.000, 0.000, 1.000],
    ])
    
    T_left_arm_to_base = np.array([
        [0.042, 0.999, -0.012, 0.480],
        [-0.999, 0.042, 0.004, 0.050],
        [0.005, 0.012, 1.000, 1.330],
        [0.000, 0.000, 0.000, 1.000],
    ])
    
    def detect_fdp():
        """执行FDP检测，返回检测到的xyz位置"""
        logging.info("开始FDP检测...")
        
        for img_head, img_depth, _, _ in camera:
            if img_head is None or img_depth is None:
                continue
            
            color = img_head.copy()
            depth = img_depth.astype(np.float32) / 1000.0
            
            logging.info("✓ 获取到图像")
            
            # YOLO分割
            results = yolo_model(color, verbose=False)
            
            if len(results) == 0 or results[0].masks is None:
                logging.warning("❌ YOLO未检测到物体")
                return None, None
            
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) == 0:
                logging.warning("❌ YOLO未检测到物体mask")
                return None, None
            
            mask = masks[0]
            if mask.shape != color.shape[:2]:
                mask = cv2.resize(mask, (color.shape[1], color.shape[0]))
            mask = (mask > 0.5).astype(bool)
            
            logging.info("✓ YOLO检测成功")
            
            # FDP位姿估计
            try:
                logging.info("正在进行FDP位姿估计...")
                pose_result = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, 
                                          iteration=args.est_refine_iter)
                
                if isinstance(pose_result, tuple):
                    pose_in_camera = pose_result[0]
                else:
                    pose_in_camera = pose_result
                
                # 提取物体在相机坐标系的位置（只取xyz）
                point_fdp_camera_xyz = pose_in_camera[:3, 3].tolist()
                
                print("\n" + "="*70)
                print("🎯 FDP检测成功！")
                print("="*70)
                print(f"📍 相机坐标系位置: [{point_fdp_camera_xyz[0]:+.4f}, {point_fdp_camera_xyz[1]:+.4f}, {point_fdp_camera_xyz[2]:+.4f}] m")
                print("="*70)
                
                logging.info("✓ FDP位姿估计完成")
                
                # 可视化：只显示绿色3D框和坐标轴
                vis = color.copy()
                
                # 绘制绿色3D框
                center_pose = pose_in_camera @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, linewidth=1)
                
                # 绘制三个坐标轴（RGB: X=红, Y=绿, Z=蓝）
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=1, transparency=0, is_input_rgb=True)
                
                # 显示状态文本
                cv2.putText(vis, "Press 'g' to grasp, 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])  # RGB to BGR
                cv2.waitKey(1)
                
                return point_fdp_camera_xyz, vis
                
            except Exception as e:
                logging.error(f"❌ FDP位姿估计失败: {e}")
                import traceback
                traceback.print_exc()
                return None, None
            
            # 只处理第一帧
            break
        
        return None, None
    
    def execute_grasp(point_fdp_camera_xyz):
        """执行抓取流程"""
        print("\n" + "="*70)
        print("🤖 开始执行抓取流程...")
        print("="*70)
        
        # 坐标变换：相机 → 基座
        point_prepare_base_xyz = transform_point(point_fdp_camera_xyz, T_head_to_base)
        logging.info(f"基座坐标系位置: [{point_prepare_base_xyz[0]:+.4f}, {point_prepare_base_xyz[1]:+.4f}, {point_prepare_base_xyz[2]:+.4f}] m")
        
        # 坐标变换：基座 → 左臂末端
        point_fdp_left_arm_xyz = transform_point(point_prepare_base_xyz, np.linalg.inv(T_left_arm_to_base))
        
        # 在左臂坐标系下调整位置（准备位置：物体上方15cm，Y方向偏移2cm）
        point_fdp_left_arm_xyz[2] += 0.15  # Z方向向上15cm
        point_fdp_left_arm_xyz[1] += 0.02  # Y方向偏移2cm
        
        # 转换回基座坐标系
        point_prepare_base_xyz = transform_point(point_fdp_left_arm_xyz, T_left_arm_to_base)
        print(f"\n准备位置（基座）: [{point_prepare_base_xyz[0]:+.4f}, {point_prepare_base_xyz[1]:+.4f}, {point_prepare_base_xyz[2]:+.4f}] m")
        
        # 步骤1：移动到准备位置
        logging.info("[1/5] 移动到准备位置（物体上方15cm）...")
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪
        point_prepare_base = list(point_prepare_base_xyz) + list(grasp_rot)
        mmk2.control_arm_pose('left_arm', point_prepare_base)
        time.sleep(2.0)
        
        # 步骤2：下降到抓取位置
        logging.info("[2/5] 下降到抓取位置（物体上方6cm）...")
        point_prepare_left_arm_xyz = transform_point(point_prepare_base_xyz, np.linalg.inv(T_left_arm_to_base))
        point_prepare_left_arm_xyz[2] -= 0.09  # Z方向下降9cm (15cm - 9cm = 6cm)
        
        point_grasp_base_xyz = transform_point(point_prepare_left_arm_xyz, T_left_arm_to_base)
        print(f"抓取位置（基座）: [{point_grasp_base_xyz[0]:+.4f}, {point_grasp_base_xyz[1]:+.4f}, {point_grasp_base_xyz[2]:+.4f}] m")
        
        point_grasp_base = list(point_grasp_base_xyz) + list(grasp_rot)
        mmk2.control_arm_pose('left_arm', point_grasp_base)
        time.sleep(2.0)
        
        # 步骤3：打开夹爪抓取
        logging.info("[3/5] 打开夹爪...")
        mmk2.set_robot_eef('left_arm', 0)  # 关闭夹爪
        time.sleep(1.5)
        
        # 步骤4：返回到准备位置
        logging.info("[4/5] 返回到准备位置...")
        mmk2.control_arm_pose('left_arm', point_prepare_base)
        time.sleep(2.0)
        
        # 步骤5：返回到初始位置
        logging.info("[5/5] 返回到初始位置...")
        mmk2.control_arm_pose('left_arm', init_pose)
        time.sleep(2.0)
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪
        mmk2.set_robot_eef('left_arm', 0)  # 关闭夹爪
        
        print("\n" + "="*70)
        print("✅ 抓取流程完成！")
        print("="*70)
        print("提示: 按 'r' 重新检测, 按 'q' 退出")
    
    # 主循环
    point_fdp_camera_xyz = None
    vis = None
    
    while True:
        # 执行FDP检测
        point_fdp_camera_xyz, vis = detect_fdp()
        
        if point_fdp_camera_xyz is None:
            logging.warning("❌ 检测失败，等待用户操作...")
            # 显示错误提示
            if vis is not None:
                cv2.putText(vis, "Detection Failed - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])
            
            # 等待用户按键
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    logging.info("用户退出程序")
                    cv2.destroyAllWindows()
                    return
                elif key == ord('r'):
                    logging.info("重新检测...")
                    break
            continue
        
        # 等待用户按键（检测成功）
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                logging.info("用户退出程序")
                cv2.destroyAllWindows()
                return
            elif key == ord('g'):
                execute_grasp(point_fdp_camera_xyz)
                # 抓取完成后，等待用户操作
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        logging.info("用户退出程序")
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('r'):
                        logging.info("重新检测...")
                        break
                break
            elif key == ord('r'):
                logging.info("重新检测...")
                break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("程序被中断")
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

