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
    
    grasp_rot = [0.13217062374287894, -0.09156655405705483, -0.5321287498599604, 0.831255367483429] 
    
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
        """执行FDP检测（支持多物体），返回检测到的物体列表（按规则排序）"""
        logging.info("开始FDP检测（多物体）...")
        
        detected_objects = []  # 存储所有检测到的物体
        
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
                return [], None
            
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) == 0:
                logging.warning("❌ YOLO未检测到物体mask")
                return [], None
            
            logging.info(f"YOLO检测到 {len(masks)} 个物体，开始FDP检测...")
            
            # 遍历所有检测到的物体
            for obj_idx, mask_raw in enumerate(masks):
                # 处理mask
                if mask_raw.shape != color.shape[:2]:
                    mask = cv2.resize(mask_raw, (color.shape[1], color.shape[0]))
                else:
                    mask = mask_raw.copy()
                mask = (mask > 0.5).astype(bool)
                
                # FDP位姿估计
                try:
                    logging.info(f"正在进行物体 #{obj_idx+1} 的FDP位姿估计...")
                    pose_result = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, 
                                              iteration=args.est_refine_iter)
                    
                    if isinstance(pose_result, tuple):
                        pose_in_camera = pose_result[0]
                    else:
                        pose_in_camera = pose_result
                    
                    # 提取物体在相机坐标系的位置
                    point_fdp_camera_xyz = pose_in_camera[:3, 3]
                    
                    # 转换到基座坐标系用于排序
                    point_fdp_base_xyz = transform_point(point_fdp_camera_xyz, T_head_to_base)
                    
                    # 保存物体信息
                    detected_objects.append({
                        'obj_idx': obj_idx + 1,
                        'point_camera_xyz': point_fdp_camera_xyz.tolist(),
                        'point_base_xyz': point_fdp_base_xyz,
                        'pose_in_camera': pose_in_camera,
                        'distance_from_base': np.linalg.norm(point_fdp_base_xyz)  # 距离基座的距离
                    })
                    
                    logging.info(f"✓ 物体 #{obj_idx+1} FDP检测成功: 位置={point_fdp_camera_xyz}, 基座距离={np.linalg.norm(point_fdp_base_xyz):.3f}m")
                    
                except Exception as e:
                    logging.error(f"物体 #{obj_idx+1} FDP位姿估计失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 按规则排序：按距离基座的距离排序（从近到远）
            detected_objects.sort(key=lambda x: x['distance_from_base'])
            
            if len(detected_objects) == 0:
                logging.warning("❌ 没有成功检测到任何物体")
                return [], None
            
            # 输出检测结果
            print("\n" + "="*70)
            print(f"🎯 FDP检测到 {len(detected_objects)} 个物体！")
            print("="*70)
            for obj in detected_objects:
                pos = obj['point_camera_xyz']
                dist = obj['distance_from_base']
                print(f"物体 #{obj['obj_idx']}: 相机位置=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m, "
                      f"基座距离={dist:.3f}m")
            print("="*70)
            grasp_order_str = ' → '.join([f"#{obj['obj_idx']}" for obj in detected_objects])
            print(f"📋 抓取顺序: {grasp_order_str}")
            print("="*70)
            
            # 可视化：只显示FDP 3D框和坐标轴（不显示YOLO信息）
            vis = color.copy()
            
            # 为不同物体分配不同颜色
            colors = [
                (0, 255, 0),    # 绿色 - 物体1（最近，第一个抓取）
                (255, 255, 0),  # 青色 - 物体2
                (255, 0, 255),  # 品红 - 物体3
                (0, 255, 255),  # 黄色 - 物体4
                (255, 0, 0),    # 红色 - 物体5
            ]
            
            # 绘制所有物体的FDP检测结果
            for obj in detected_objects:
                obj_idx = obj['obj_idx']
                pose_in_camera = obj['pose_in_camera']
                color_bgr = colors[(obj_idx - 1) % len(colors)]
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR转RGB
                
                # 绘制3D框
                center_pose = pose_in_camera @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, 
                                       linewidth=2, line_color=color_rgb)
                
                # 绘制三个坐标轴
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=2, 
                                  transparency=0, is_input_rgb=True)
                
                # 标记物体编号和抓取顺序
                obj_2d = K @ pose_in_camera[:3, 3]
                obj_2d = obj_2d / obj_2d[2]
                fdp_cx, fdp_cy = int(obj_2d[0]), int(obj_2d[1])
                if 0 <= fdp_cx < vis.shape[1] and 0 <= fdp_cy < vis.shape[0]:
                    # 显示物体编号和抓取顺序
                    grasp_order = detected_objects.index(obj) + 1
                    cv2.putText(vis, f"#{obj_idx}({grasp_order})", (fdp_cx + 10, fdp_cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            # 显示状态文本
            cv2.putText(vis, f"Detected {len(detected_objects)} objects - Press 'g' to grasp all, 'r' to retry, 'q' to quit", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('FDP Auto Grasp - Detection', vis[...,::-1])  # RGB to BGR
            cv2.waitKey(1)
            
            # 只处理第一帧
            break
        
        return detected_objects, vis
    
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
        point_fdp_left_arm_xyz[1] += 0.00  # Y方向偏移2cm
        point_fdp_left_arm_xyz[0] -= 0.01  # x方向偏移2cm      
        
        # 转换回基座坐标系
        point_prepare_base_xyz = transform_point(point_fdp_left_arm_xyz, T_left_arm_to_base)
        print(f"\n准备位置（基座）: [{point_prepare_base_xyz[0]:+.4f}, {point_prepare_base_xyz[1]:+.4f}, {point_prepare_base_xyz[2]:+.4f}] m")
        
        # 步骤1：移动到准备位置（物体上方15cm）
        logging.info("[1/5] 移动到准备位置（物体上方15cm）...")
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪
        time.sleep(0.5)  # 等待夹爪完全打开
        point_prepare_base = list(point_prepare_base_xyz) + list(grasp_rot)
        mmk2.control_arm_pose('left_arm', point_prepare_base)
        # time.sleep(1.5)
        
        # # 步骤2：在准备位置调整抓取姿态（再次确保夹爪打开）
        # logging.info("[2/5] 在准备位置调整抓取姿态...")
        # mmk2.set_robot_eef('left_arm', 1)  # 再次确保夹爪打开
        # time.sleep(0.5)  # 等待夹爪完全打开
        # grasp_rot_z=[0.23747033023505748, 0.1850776927726681, -0.612975022014487, 0.7304900494067923]
        # point_prepare_base_adjusted = list(point_prepare_base_xyz) + list(grasp_rot_z)
        # mmk2.control_arm_pose('left_arm', point_prepare_base_adjusted)
        # # time.sleep(1.5)
        
        # 步骤3：下降到抓取位置（保持调整后的抓取姿态）
        logging.info("[3/5] 下降到抓取位置（物体上方6cm，保持调整后的姿态）...")
        mmk2.set_robot_eef('left_arm', 1)  # 再次确保夹爪打开
        time.sleep(0.5)  # 等待夹爪完全打开
        point_prepare_left_arm_xyz = transform_point(point_prepare_base_xyz, np.linalg.inv(T_left_arm_to_base))
        point_prepare_left_arm_xyz[2] -= 0.04  # Z方向下降9cm (15cm - 9cm = 6cm)
        
        point_grasp_base_xyz = transform_point(point_prepare_left_arm_xyz, T_left_arm_to_base)
        print(f"抓取位置（基座）: [{point_grasp_base_xyz[0]:+.4f}, {point_grasp_base_xyz[1]:+.4f}, {point_grasp_base_xyz[2]:+.4f}] m")
        
        # 使用调整后的抓取姿态下降到抓取位置
        point_grasp_base = list(point_grasp_base_xyz) + list(grasp_rot)
        mmk2.control_arm_pose('left_arm', point_grasp_base)
        # time.sleep(1.5)
        

        # 步骤4：关闭夹爪抓取
        logging.info("[4/5] 关闭夹爪...")
        mmk2.set_robot_eef('left_arm', 0.2)  # 关闭夹爪
        # time.sleep(1.5)
        
        # 步骤5：返回到准备位置
        logging.info("[5/5] 返回到准备位置...")
        mmk2.control_arm_pose('left_arm', point_prepare_base)
        # time.sleep(2.0)
        
        # 步骤6：返回到初始位置
        logging.info("[6/6] 返回到初始位置...")
        mmk2.control_arm_pose('left_arm', init_pose)
        # time.sleep(2.0)
        # mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪
        # mmk2.set_robot_eef('left_arm', 0)  # 关闭夹爪

        # logging.info("[6/6] 移动到插入位置...")

        point1 =  [0.49772561662618986, -0.14028049410834612, 1.0937719152398795,-0.020947593952357995, 0.15451359383013447, -0.733548326784465, 0.6615085788567032]
        mmk2.control_arm_pose('left_arm', point1) 

        point2 = [0.4134121275411202, -0.3309386375775653, 1.077459282363231,0.32362502211975475, -0.5558974843278341, 0.5839813334978876, -0.49518747369254273]
        mmk2.control_arm_pose('left_arm', point2)       

        final_position =[0.4687646420653824, -0.32870128737087184, 0.9690525881772679,0.736428053817261, -0.6748555940148481, -0.04181095507594987, -0.022259667864207466]
        final_rot=[0.736428053817261, -0.6748555940148481, -0.04181095507594987, -0.022259667864207466]


        mmk2.control_arm_pose('left_arm', final_position)


        point_prepare2_base_xyz = [0.4687646420653824, -0.32870128737087184, 0.9690525881772679]

        point_sel_left_arm_xyz = transform_point(point_prepare2_base_xyz, np.linalg.inv(T_left_arm_to_base))
        
        # # 在左臂坐标系下调整位置（准备位置：物体上方15cm，Y方向偏移2cm）
        point_sel_left_arm_xyz[2] += 0.05  # Z方向向上15cm
        # point_sel_left_arm_xyz[1] += 0.00  # Y方向偏移2cm
        # point_sel_left_arm_xyz[0] -= 0.01  # x方向偏移2cm      
        
        # # 转换回基座坐标系
        point_prepare2_base_xyz = transform_point(point_sel_left_arm_xyz, T_left_arm_to_base)


        point_prepare2_base = list(point_prepare2_base_xyz) + list(final_rot)
        mmk2.control_arm_pose('left_arm', point_prepare2_base)

        time.sleep(0.5)
        mmk2.set_robot_eef('left_arm', 1)  # 打开夹爪，放物体

        mmk2.control_arm_pose('left_arm', final_position)
        point_1=[0.4700205603986788, -0.25949284159761343, 0.9996301859105843,0.6985838059280328, -0.6914012484354668, -0.012329333567599703, -0.1838286356658333]

        mmk2.control_arm_pose('left_arm', point_1)
        

        mmk2.control_arm_pose('left_arm', point1)         


        # logging.info("[7/7] 返回到初始位置...")
        mmk2.control_arm_pose('left_arm', init_pose)        

        print("\n" + "="*70)
        print("✅ 抓取流程完成！")
        print("="*70)
        print("提示: 按 'r' 重新检测, 按 'q' 退出")
    
    # 主循环
    detected_objects = []
    vis = None
    
    while True:
        # 执行FDP检测
        detected_objects, vis = detect_fdp()
        
        if len(detected_objects) == 0:
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
                # 按顺序逐个抓取所有物体
                print("\n" + "="*70)
                print(f"🤖 开始按顺序抓取 {len(detected_objects)} 个物体...")
                print("="*70)
                
                for grasp_idx, obj in enumerate(detected_objects):
                    print(f"\n📍 正在抓取物体 #{obj['obj_idx']} ({grasp_idx+1}/{len(detected_objects)})...")
                    point_fdp_camera_xyz = obj['point_camera_xyz']
                    execute_grasp(point_fdp_camera_xyz)
                    
                    # 每次抓取后等待一小段时间
                    if grasp_idx < len(detected_objects) - 1:
                        logging.info(f"物体 #{obj['obj_idx']} 抓取完成，准备抓取下一个...")
                        time.sleep(1.0)
                
                print("\n" + "="*70)
                print(f"✅ 所有 {len(detected_objects)} 个物体抓取完成！")
                print("="*70)
                
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

