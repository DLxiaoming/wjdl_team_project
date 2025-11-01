#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDP自动检测抓取脚本（使用FDP完整位姿对齐抓取）
- 使用FDP检测获取物体完整位姿（位置+姿态，相机坐标系）
- 实时读取头部到基座和机械臂末端到基座的变换矩阵
- 按照FDP检测的完整位姿进行对齐并抓取
- 避免机械臂转到别扭的角度导致卡死
"""

import os
import sys
import logging
import time
import numpy as np
import cv2
import subprocess
import re
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

def transform_quaternion(quat, T):
    """
    将四元数（姿态）从局部坐标系转换到另一个坐标系
    :param quat: list 或 np.ndarray, 四元数 [x, y, z, w]
    :param T: np.ndarray, 4x4 齐次变换矩阵
    :return: new_quat: np.ndarray, 转换后的四元数 [x, y, z, w]
    """
    # 提取旋转矩阵部分
    R_transform = T[:3, :3]
    # 原四元数对应的旋转矩阵
    R_original = Rotation.from_quat(quat).as_matrix()
    # 转换后的旋转矩阵
    R_transformed = R_transform @ R_original
    # 转为四元数
    new_quat = Rotation.from_matrix(R_transformed).as_quat()
    return new_quat

def get_tf_transform(target_frame, source_frame, timeout=5.0):
    """
    通过ROS2 TF获取变换矩阵
    :param target_frame: 目标坐标系（如 'base_link'）
    :param source_frame: 源坐标系（如 'head_camera_link'）
    :param timeout: 超时时间（秒）
    :return: 4x4 变换矩阵 (target_frame <- source_frame)，如果失败返回None
    """
    try:
        # 调用 ros2 run tf2_ros tf2_echo 命令
        cmd = ['ros2', 'run', 'tf2_ros', 'tf2_echo', target_frame, source_frame]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            logging.warning(f"TF变换获取失败: {result.stderr}")
            return None
        
        # 解析输出，提取变换矩阵
        output = result.stdout + result.stderr
        lines = output.split('\n')
        
        # 查找平移部分
        translation = None
        rotation = None
        
        for i, line in enumerate(lines):
            # 查找平移: x, y, z
            if 'Translation:' in line or 'translation:' in line:
                # 提取 x, y, z 值
                trans_match = re.search(r'x:\s*([-+]?\d+\.?\d*),\s*y:\s*([-+]?\d+\.?\d*),\s*z:\s*([-+]?\d+\.?\d*)', output)
                if trans_match:
                    translation = [float(trans_match.group(1)), 
                                 float(trans_match.group(2)), 
                                 float(trans_match.group(3))]
            
            # 查找旋转: x, y, z, w
            if 'Rotation:' in line or 'rotation:' in line:
                # 提取 x, y, z, w 值
                rot_match = re.search(r'x:\s*([-+]?\d+\.?\d*),\s*y:\s*([-+]?\d+\.?\d*),\s*z:\s*([-+]?\d+\.?\d*),\s*w:\s*([-+]?\d+\.?\d*)', output)
                if rot_match:
                    rotation = [float(rot_match.group(1)), 
                              float(rot_match.group(2)), 
                              float(rot_match.group(3)), 
                              float(rot_match.group(4))]
        
        # 如果找到了平移和旋转，构建变换矩阵
        if translation is not None and rotation is not None:
            T = np.eye(4)
            # 设置平移
            T[:3, 3] = translation
            # 设置旋转（四元数转旋转矩阵）
            R_rot = Rotation.from_quat(rotation).as_matrix()
            T[:3, :3] = R_rot
            return T
        else:
            logging.warning(f"无法解析TF输出: {output[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        logging.warning(f"获取TF变换超时: {target_frame} <- {source_frame}")
        return None
    except Exception as e:
        logging.warning(f"获取TF变换时出错: {e}")
        return None

def pose_to_matrix(position, quaternion):
    """
    将位置和四元数转换为4x4变换矩阵
    :param position: [x, y, z]
    :param quaternion: [qx, qy, qz, qw]
    :return: 4x4 变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return T

def main():
    import argparse
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='FDP自动检测抓取（使用FDP完整位姿对齐抓取）')
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
    print("🤖 FDP自动检测抓取系统（使用FDP完整位姿对齐抓取）")
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
    
    # ============ 3. 获取实时变换矩阵 ============
    print("\n" + "="*70)
    print("📐 获取实时变换矩阵...")
    print("="*70)
    
    # 实时获取头部到基座的变换矩阵 (base_link <- head_camera_link)
    logging.info("获取头部到基座的变换矩阵 (base_link <- head_camera_link)...")
    T_head_to_base = get_tf_transform('base_link', 'head_camera_link')
    
    if T_head_to_base is None:
        logging.error("❌ 无法获取头部到基座的变换矩阵，使用默认值")
        T_head_to_base = np.array([
            [-0.001, -0.749, 0.662, 0.365],
            [-1.000, 0.001, -0.001, 0.036],
            [-0.000, -0.662, -0.749, 1.516],
            [0.000, 0.000, 0.000, 1.000],
        ])
    else:
        logging.info("✓ 成功获取头部到基座的变换矩阵")
        print(f"T_head_to_base:\n{T_head_to_base}")
    
    # 实时获取机械臂末端到基座的变换矩阵 (base_link <- left_arm_end_link)
    logging.info("获取机械臂末端到基座的变换矩阵 (base_link <- left_arm_end_link)...")
    T_left_arm_to_base = get_tf_transform('base_link', 'left_arm_end_link')
    
    if T_left_arm_to_base is None:
        logging.warning("⚠ 无法通过TF获取机械臂末端变换，使用机器人API...")
        # 如果TF获取失败，尝试从机器人当前状态获取
        try:
            left_arm_eef = mmk2.get_arm_ee_pose('left_arm')
            if left_arm_eef:
                position = left_arm_eef[0]  # [x, y, z]
                quaternion = left_arm_eef[1]  # [qx, qy, qz, qw]
                # 注意：这里获取的是当前位姿，不是变换矩阵
                # 我们需要的是机械臂末端坐标系到基座的变换
                # 如果无法获取，使用默认值
                logging.warning("使用默认的左臂末端变换矩阵")
                T_left_arm_to_base = np.array([
                    [0.042, 0.999, -0.012, 0.480],
                    [-0.999, 0.042, 0.004, 0.050],
                    [0.005, 0.012, 1.000, 1.330],
                    [0.000, 0.000, 0.000, 1.000],
                ])
            else:
                logging.error("无法从机器人API获取左臂位姿，使用默认值")
                T_left_arm_to_base = np.array([
                    [0.042, 0.999, -0.012, 0.480],
                    [-0.999, 0.042, 0.004, 0.050],
                    [0.005, 0.012, 1.000, 1.330],
                    [0.000, 0.000, 0.000, 1.000],
                ])
        except Exception as e:
            logging.warning(f"获取左臂位姿时出错: {e}，使用默认值")
            T_left_arm_to_base = np.array([
                [0.042, 0.999, -0.012, 0.480],
                [-0.999, 0.042, 0.004, 0.050],
                [0.005, 0.012, 1.000, 1.330],
                [0.000, 0.000, 0.000, 1.000],
            ])
    else:
        logging.info("✓ 成功获取机械臂末端到基座的变换矩阵")
        print(f"T_left_arm_to_base:\n{T_left_arm_to_base}")
    
    # 获取初始位置（从机器人当前状态）
    try:
        left_arm_eef_init = mmk2.get_arm_ee_pose('left_arm')
        if left_arm_eef_init:
            init_pose = list(left_arm_eef_init[0]) + list(left_arm_eef_init[1])
            logging.info("✓ 从机器人获取初始位置")
        else:
            init_pose = [0.4798196384291117, 0.050344892205700036, 1.3300944085789266,
                         0.004922124603778732, -0.003476176971870248, -0.6913608163682425, 0.7224845399547885]
            logging.info("使用默认初始位置")
    except Exception as e:
        logging.warning(f"获取初始位置时出错: {e}，使用默认值")
        init_pose = [0.4798196384291117, 0.050344892205700036, 1.3300944085789266,
                     0.004922124603778732, -0.003476176971870248, -0.6913608163682425, 0.7224845399547885]
    
    # ============ 4. 主循环：检测和抓取 ============
    print("\n" + "="*70)
    print("🔍 FDP自动检测抓取系统（使用FDP完整位姿对齐抓取）")
    print("="*70)
    print("提示: 按 'g' 开始抓取, 按 'r' 重新检测, 按 'q' 退出")
    print("="*70 + "\n")
    
    def detect_fdp():
        """执行FDP检测，返回检测到的完整位姿（位置+姿态）"""
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
                vis = color.copy()
                cv2.putText(vis, "YOLO Detection Failed - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])
                cv2.waitKey(1)
                return None, None, vis, None
            
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) == 0:
                logging.warning("❌ YOLO未检测到物体mask")
                vis = color.copy()
                cv2.putText(vis, "YOLO Mask Not Found - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])
                cv2.waitKey(1)
                return None, None, vis, None
            
            mask = masks[0]
            if mask.shape != color.shape[:2]:
                mask = cv2.resize(mask, (color.shape[1], color.shape[0]))
            mask = (mask > 0.5).astype(bool)
            yolo_mask = mask.copy()  # 保存用于可视化
            
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
                
                # 提取物体在相机坐标系的完整位姿
                point_fdp_camera_xyz = pose_in_camera[:3, 3].tolist()
                obj_rot_camera = pose_in_camera[:3, :3]
                obj_quat_camera = Rotation.from_matrix(obj_rot_camera).as_quat()
                
                # 获取FDP置信度分数（如果可用）
                confidence_score = None
                if hasattr(est, 'scores') and est.scores is not None and len(est.scores) > 0:
                    confidence_score = float(est.scores[0])  # 最佳匹配的分数（已排序，第一个最高）
                
                print("\n" + "="*70)
                print("🎯 FDP检测成功！")
                print("="*70)
                print(f"📍 相机坐标系位置: [{point_fdp_camera_xyz[0]:+.4f}, {point_fdp_camera_xyz[1]:+.4f}, {point_fdp_camera_xyz[2]:+.4f}] m")
                print(f"🎯 相机坐标系四元数: [{obj_quat_camera[0]:+.4f}, {obj_quat_camera[1]:+.4f}, {obj_quat_camera[2]:+.4f}, {obj_quat_camera[3]:+.4f}]")
                if confidence_score is not None:
                    print(f"📊 FDP置信度分数: {confidence_score:.4f}")
                print("="*70)
                
                logging.info("✓ FDP位姿估计完成")
                
                # 可视化：显示YOLO分割、绿色3D框和坐标轴
                vis = color.copy()
                
                # 1. 绘制YOLO分割mask
                if yolo_mask is not None:
                    # 半透明浅蓝色遮罩
                    mask_overlay = vis.copy()
                    mask_overlay[yolo_mask] = (100, 200, 255)  # 浅蓝色 (BGR格式)
                    vis = cv2.addWeighted(mask_overlay, 0.3, vis, 0.7, 0)
                    
                    # 黄色轮廓线（细线）
                    mask_uint8 = (yolo_mask.astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)  # 黄色，线宽1
                    
                    # YOLO mask中心点
                    moments = cv2.moments(mask_uint8)
                    if moments['m00'] > 0:
                        mask_cx = int(moments['m10'] / moments['m00'])
                        mask_cy = int(moments['m01'] / moments['m00'])
                        cv2.circle(vis, (mask_cx, mask_cy), 4, (0, 0, 255), -1)  # 红色圆点
                        cv2.circle(vis, (mask_cx, mask_cy), 6, (255, 255, 255), 1)  # 白色外圈（细）
                        cv2.putText(vis, "YOLO", (mask_cx + 8, mask_cy - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 小字体
                
                # 2. 绘制FDP检测结果
                center_pose = pose_in_camera @ np.linalg.inv(to_origin)
                
                # 绘制绿色3D框（细线）
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, linewidth=1)
                
                # 绘制三个坐标轴（RGB: X=红, Y=绿, Z=蓝，细线）
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=1, transparency=0, is_input_rgb=True)
                
                # FDP中心点
                obj_2d = K @ pose_in_camera[:3, 3]
                obj_2d = obj_2d / obj_2d[2]
                fdp_cx, fdp_cy = int(obj_2d[0]), int(obj_2d[1])
                if 0 <= fdp_cx < vis.shape[1] and 0 <= fdp_cy < vis.shape[0]:
                    cv2.drawMarker(vis, (fdp_cx, fdp_cy), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)  # 紫色十字（细）
                    cv2.putText(vis, "FDP", (fdp_cx + 8, fdp_cy + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)  # 小字体
                
                # 显示状态文本和置信度
                status_text = "Press 'g' to grasp, 'r' to retry, 'q' to quit"
                cv2.putText(vis, status_text, (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示置信度分数（如果可用）
                if confidence_score is not None:
                    conf_text = f"FDP Score: {confidence_score:.3f}"
                    cv2.putText(vis, conf_text, (20, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # 黄色文字
                
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])  # RGB to BGR
                cv2.waitKey(1)
                
                return point_fdp_camera_xyz, obj_quat_camera, vis, confidence_score
                
            except Exception as e:
                logging.error(f"❌ FDP位姿估计失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 即使FDP失败，也显示YOLO分割结果
                vis = color.copy()
                if yolo_mask is not None:
                    # 半透明浅蓝色遮罩
                    mask_overlay = vis.copy()
                    mask_overlay[yolo_mask] = (100, 200, 255)  # 浅蓝色 (BGR格式)
                    vis = cv2.addWeighted(mask_overlay, 0.3, vis, 0.7, 0)
                    
                    # 黄色轮廓线（细线）
                    mask_uint8 = (yolo_mask.astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)  # 黄色，线宽1
                    
                    # YOLO mask中心点
                    moments = cv2.moments(mask_uint8)
                    if moments['m00'] > 0:
                        mask_cx = int(moments['m10'] / moments['m00'])
                        mask_cy = int(moments['m01'] / moments['m00'])
                        cv2.circle(vis, (mask_cx, mask_cy), 4, (0, 0, 255), -1)  # 红色圆点
                        cv2.circle(vis, (mask_cx, mask_cy), 6, (255, 255, 255), 1)  # 白色外圈（细）
                        cv2.putText(vis, "YOLO", (mask_cx + 8, mask_cy - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 小字体
                
                cv2.putText(vis, "FDP Detection Failed - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])  # RGB to BGR
                cv2.waitKey(1)
                
                return None, None, vis, None
            
            # 只处理第一帧
            break
        
        # 如果没有图像，返回空的可视化
        vis = None
        if img_head is not None:
            vis = img_head.copy()
            cv2.putText(vis, "No Image Captured - Press 'r' to retry, 'q' to quit", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])
            cv2.waitKey(1)
        return None, None, vis, None
    
    def execute_grasp(point_fdp_camera_xyz, quat_fdp_camera, T_head_to_base_ref, T_left_arm_to_base_ref):
        """执行抓取流程（使用FDP检测的完整位姿进行对齐和抓取）"""
        print("\n" + "="*70)
        print("🤖 开始执行抓取流程（使用FDP完整位姿对齐抓取）...")
        print("="*70)
        
        # 实时更新变换矩阵（因为头部角度可能已改变）
        logging.info("实时更新变换矩阵...")
        T_head_to_base = get_tf_transform('base_link', 'head_camera_link', timeout=3.0)
        if T_head_to_base is None:
            T_head_to_base = T_head_to_base_ref  # 使用传入的变换矩阵
            logging.warning("无法获取最新变换矩阵，使用初始值")
        else:
            logging.info("✓ 已更新头部到基座的变换矩阵")
        
        # 使用传入的左臂变换矩阵（通常不会改变）
        T_left_arm_to_base = T_left_arm_to_base_ref
        
        # 坐标变换：相机 → 基座（位置）
        point_fdp_base_xyz = transform_point(point_fdp_camera_xyz, T_head_to_base)
        
        # 姿态变换：相机 → 基座（姿态）
        quat_fdp_base = transform_quaternion(quat_fdp_camera, T_head_to_base)
        
        logging.info(f"FDP检测位置（基座）: [{point_fdp_base_xyz[0]:+.4f}, {point_fdp_base_xyz[1]:+.4f}, {point_fdp_base_xyz[2]:+.4f}] m")
        logging.info(f"FDP检测姿态（基座）: [{quat_fdp_base[0]:+.4f}, {quat_fdp_base[1]:+.4f}, {quat_fdp_base[2]:+.4f}, {quat_fdp_base[3]:+.4f}]")
        
        # 计算准备位置（物体上方15cm，使用FDP完整姿态）
        # 在基座坐标系下，沿FDP检测的Z轴方向向上移动15cm
        R_fdp_base = Rotation.from_quat(quat_fdp_base).as_matrix()
        z_axis_fdp = R_fdp_base[:, 2]  # FDP检测的Z轴方向（物体坐标系）
        approach_offset = 0.15  # 15cm
        point_prepare_base_xyz = point_fdp_base_xyz + z_axis_fdp * approach_offset
        
        print(f"\n准备位置（基座，物体上方15cm）: [{point_prepare_base_xyz[0]:+.4f}, {point_prepare_base_xyz[1]:+.4f}, {point_prepare_base_xyz[2]:+.4f}] m")
        
        # 步骤1：位姿对齐 - 移动到准备位置（使用FDP完整位姿）
        logging.info("[1/6] 位姿对齐 - 移动到准备位置（物体上方15cm，使用FDP完整位姿）...")
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪
        point_prepare_base = list(point_prepare_base_xyz) + list(quat_fdp_base)  # 使用FDP完整位姿
        mmk2.control_arm_pose('left_arm', point_prepare_base)
        time.sleep(2.5)  # 暂停，观察对齐效果
        
        # 步骤2：打开夹爪
        logging.info("[2/6] 打开夹爪...")
        mmk2.set_robot_eef('left_arm', 1)  # 确保打开
        time.sleep(1.0)
        
        # 步骤3：下降到抓取位置（物体上方3cm，使用FDP完整位姿）
        logging.info("[3/6] 下降到抓取位置（物体上方3cm）...")
        grasp_offset = 0.03  # 3cm
        point_grasp_base_xyz = point_fdp_base_xyz + z_axis_fdp * grasp_offset
        print(f"抓取位置（基座）: [{point_grasp_base_xyz[0]:+.4f}, {point_grasp_base_xyz[1]:+.4f}, {point_grasp_base_xyz[2]:+.4f}] m")
        
        point_grasp_base = list(point_grasp_base_xyz) + list(quat_fdp_base)  # 使用FDP完整位姿
        mmk2.control_arm_pose('left_arm', point_grasp_base)
        time.sleep(2.0)
        
        # 步骤4：闭合夹爪抓取
        logging.info("[4/6] 闭合夹爪抓取...")
        mmk2.set_robot_eef('left_arm', 0)  # 关闭夹爪
        time.sleep(1.5)
        
        # 步骤5：抬起（物体上方5cm，使用FDP完整位姿）
        logging.info("[5/6] 抬起物体（物体上方5cm）...")
        lift_offset = 0.05  # 5cm
        point_lift_base_xyz = point_fdp_base_xyz + z_axis_fdp * lift_offset
        point_lift_base = list(point_lift_base_xyz) + list(quat_fdp_base)  # 使用FDP完整位姿
        mmk2.control_arm_pose('left_arm', point_lift_base)
        time.sleep(2.0)
        
        # 步骤6：返回到初始位置
        logging.info("[6/6] 返回到初始位置...")
        mmk2.control_arm_pose('left_arm', init_pose)
        time.sleep(2.0)
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪（释放物体）
        mmk2.set_robot_eef('left_arm', 0)  # 关闭夹爪
        
        print("\n" + "="*70)
        print("✅ 抓取流程完成！")
        print("="*70)
        print("提示: 按 'r' 重新检测, 按 'q' 退出")
    
    # 主循环
    point_fdp_camera_xyz = None
    quat_fdp_camera = None
    vis = None
    confidence_score = None
    
    while True:
        # 执行FDP检测
        point_fdp_camera_xyz, quat_fdp_camera, vis, confidence_score = detect_fdp()
        
        if point_fdp_camera_xyz is None or quat_fdp_camera is None:
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
                execute_grasp(point_fdp_camera_xyz, quat_fdp_camera, T_head_to_base, T_left_arm_to_base)
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

