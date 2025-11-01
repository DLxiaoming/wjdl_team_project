#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯FDP实时监测脚本（相机坐标系版本）
- 使用机器人摄像头进行实时FDP检测
- 显示绿色3D框和三个坐标轴（RGB）
- 输出相机坐标系下的6D位姿信息
- 不进行基座坐标转换
- 无抓取功能
"""

import os
import sys
import logging
import time
import json
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

def main():
    import argparse
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='FDP实时监测（纯检测，无抓取，仅相机坐标）')
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--robot_ip', type=str, required=False, default='192.168.11.200')
    
    # 模型和数据
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
    print("🔍 FDP实时监测系统（仅相机坐标系）")
    print("="*70)
    print(f"机器人IP: {args.robot_ip}")
    print(f"头部俯仰角: {np.degrees(args.head_pitch):.1f}° ({args.head_pitch:.4f} rad)")
    print("="*70 + "\n")
    
    logging.info("初始化机器人...")
    robot = MMK2RealRobot(ip=args.robot_ip)
    time.sleep(2.0)
    
    # 设置头部角度
    logging.info(f"设置头部角度: {np.degrees(args.head_pitch):.1f}°")
    robot.set_robot_head_pose(0.0, args.head_pitch)
    time.sleep(1.0)
    
    # 加载模型
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
    
    logging.info("✓ 初始化完成")
    logging.info("="*70)
    
    # 相机内参
    K = np.array([[601.87, 0, 321.05], [0, 601.87, 252.46], [0, 0, 1]])
    camera = robot.camera
    time.sleep(2.0)
    
    print("\n" + "="*70)
    print("🔍 开始FDP检测（相机坐标系）")
    print("="*70)
    print("提示: 按 'r' 重新检测, 按 'q' 退出程序\n")
    
    def detect_once():
        """执行一次FDP检测"""
        pose_detected = False
        pose_in_camera = None
        vis = None
        yolo_mask = None  # 保存YOLO mask用于可视化
        
        logging.info("正在获取图像...")
        for img_head, img_depth, _, _ in camera:
            if img_head is None or img_depth is None:
                continue
            
            color = img_head.copy()
            depth = img_depth.astype(np.float32) / 1000.0
            
            logging.info("✓ 获取到图像，开始检测...")
            
            # YOLO分割
            results = yolo_model(color, verbose=False)
            
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                if len(masks) > 0:
                    mask = masks[0]
                    if mask.shape != color.shape[:2]:
                        mask = cv2.resize(mask, (color.shape[1], color.shape[0]))
                    mask = (mask > 0.5).astype(bool)
                    yolo_mask = mask  # 保存mask用于可视化
                    
                    # FDP位姿估计
                    try:
                        logging.info("正在进行FDP位姿估计...")
                        pose_result = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                        
                        if isinstance(pose_result, tuple):
                            pose_in_camera = pose_result[0]
                        else:
                            pose_in_camera = pose_result
                        
                        pose_detected = True
                        
                        # 提取位姿信息（相机坐标系）
                        obj_pos_camera = pose_in_camera[:3, 3]
                        obj_rot_camera = pose_in_camera[:3, :3]
                        obj_quat_camera = Rotation.from_matrix(obj_rot_camera).as_quat()
                        euler_camera = Rotation.from_matrix(obj_rot_camera).as_euler('xyz', degrees=True)
                        
                        # 输出结果
                        print("\n" + "="*70)
                        print("🎯 FDP检测到物体！")
                        print("="*70)
                        print(f"📍 相机坐标系 (Camera Frame):")
                        print(f"   位置: [{obj_pos_camera[0]:+.4f}, {obj_pos_camera[1]:+.4f}, {obj_pos_camera[2]:+.4f}] m")
                        print(f"   四元数: [{obj_quat_camera[0]:+.4f}, {obj_quat_camera[1]:+.4f}, {obj_quat_camera[2]:+.4f}, {obj_quat_camera[3]:+.4f}]")
                        print(f"   欧拉角: Roll={euler_camera[0]:+7.2f}°, Pitch={euler_camera[1]:+7.2f}°, Yaw={euler_camera[2]:+7.2f}°")
                        print("\n📊 相机坐标系变换矩阵 (4×4):")
                        print(f"   [{pose_in_camera[0,0]:+.6f}, {pose_in_camera[0,1]:+.6f}, {pose_in_camera[0,2]:+.6f}, {pose_in_camera[0,3]:+.6f}]")
                        print(f"   [{pose_in_camera[1,0]:+.6f}, {pose_in_camera[1,1]:+.6f}, {pose_in_camera[1,2]:+.6f}, {pose_in_camera[1,3]:+.6f}]")
                        print(f"   [{pose_in_camera[2,0]:+.6f}, {pose_in_camera[2,1]:+.6f}, {pose_in_camera[2,2]:+.6f}, {pose_in_camera[2,3]:+.6f}]")
                        print(f"   [{pose_in_camera[3,0]:+.6f}, {pose_in_camera[3,1]:+.6f}, {pose_in_camera[3,2]:+.6f}, {pose_in_camera[3,3]:+.6f}]")
                        
                        # 计算并显示YOLO与FDP中心的偏移（用于判断准确性）
                        if yolo_mask is not None:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            moments = cv2.moments(mask_uint8)
                            if moments['m00'] > 0:
                                yolo_cx = int(moments['m10'] / moments['m00'])
                                yolo_cy = int(moments['m01'] / moments['m00'])
                                
                                # 计算FDP中心投影
                                fdp_2d = K @ pose_in_camera[:3, 3]
                                fdp_2d = fdp_2d / fdp_2d[2]
                                fdp_cx, fdp_cy = int(fdp_2d[0]), int(fdp_2d[1])
                                
                                # 计算像素偏移
                                offset_x = fdp_cx - yolo_cx
                                offset_y = fdp_cy - yolo_cy
                                offset_total = np.sqrt(offset_x**2 + offset_y**2)
                                
                                print(f"\n🔍 检测质量分析:")
                                print(f"   YOLO中心: ({yolo_cx}, {yolo_cy}) 像素")
                                print(f"   FDP中心:  ({fdp_cx}, {fdp_cy}) 像素")
                                print(f"   像素偏移: Δx={offset_x:+d}, Δy={offset_y:+d}, 总偏移={offset_total:.1f}像素")
                                if offset_total < 20:
                                    print(f"   ✓ 偏移较小，FDP检测准确")
                                elif offset_total < 50:
                                    print(f"   ⚠ 偏移中等，FDP可能略有偏差")
                                else:
                                    print(f"   ✗ 偏移较大，FDP检测可能不准确！")
                        
                        print("="*70)
                        logging.info("✓ FDP位姿估计完成（相机坐标系）")
                        
                    except Exception as e:
                        logging.error(f"FDP位姿估计失败: {e}")
                        import traceback
                        traceback.print_exc()
                        pose_detected = False
                else:
                    logging.warning("YOLO未检测到物体mask")
            else:
                logging.warning("YOLO未检测到物体")
            
            # 可视化
            vis = color.copy()
            
            # 1. 绘制YOLO分割mask（如果存在）
            if yolo_mask is not None:
                # 半透明蓝色遮罩
                mask_overlay = vis.copy()
                mask_overlay[yolo_mask] = (100, 200, 255)  # 浅蓝色
                vis = cv2.addWeighted(mask_overlay, 0.3, vis, 0.7, 0)
                
                # 黄色轮廓线（细）
                mask_uint8 = (yolo_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)  # 线宽改为1
                
                # YOLO mask中心点
                moments = cv2.moments(mask_uint8)
                if moments['m00'] > 0:
                    mask_cx = int(moments['m10'] / moments['m00'])
                    mask_cy = int(moments['m01'] / moments['m00'])
                    cv2.circle(vis, (mask_cx, mask_cy), 5, (0, 0, 255), -1)  # 红色圆点（更小）
                    cv2.circle(vis, (mask_cx, mask_cy), 7, (255, 255, 255), 1)  # 白色外圈（更细）
                    cv2.putText(vis, "YOLO", (mask_cx + 10, mask_cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 字体更小
            
            # 2. 绘制FDP检测结果（如果存在）
            if pose_detected:
                # 绘制绿色3D框（细线）
                center_pose = pose_in_camera @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, linewidth=1)  # 线宽改为1
                
                # 绘制三个坐标轴（细线）
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=2, transparency=0, is_input_rgb=True)  # thickness改为2
                
                # FDP中心点
                obj_2d = K @ pose_in_camera[:3, 3]
                obj_2d = obj_2d / obj_2d[2]
                fdp_cx, fdp_cy = int(obj_2d[0]), int(obj_2d[1])
                if 0 <= fdp_cx < vis.shape[1] and 0 <= fdp_cy < vis.shape[0]:
                    cv2.drawMarker(vis, (fdp_cx, fdp_cy), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)  # 更小的十字
                    cv2.putText(vis, "FDP", (fdp_cx + 10, fdp_cy + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)  # 字体更小
                
                cv2.putText(vis, "Detection Complete - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "No object detected - Press 'r' to retry, 'q' to quit", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 只处理第一帧
            break
        
        return vis, pose_detected
    
    # 主循环
    try:
        vis, pose_detected = detect_once()
        
        if vis is not None:
            cv2.imshow('FDP Monitor (Camera Frame)', vis[...,::-1])  # RGB to BGR
        
        logging.info("检测完成，等待用户操作...")
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                logging.info("用户退出程序")
                break
            elif key == ord('r'):
                logging.info("重新检测中...")
                vis, pose_detected = detect_once()
                if vis is not None:
                    cv2.imshow('FDP Monitor (Camera Frame)', vis[...,::-1])
                    
    except KeyboardInterrupt:
        logging.info("程序被中断")
    finally:
        cv2.destroyAllWindows()
        logging.info("程序结束")

if __name__ == '__main__':
    main()
