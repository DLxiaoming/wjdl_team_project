#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时展示相机图像
不包含物体检测功能
"""

import os
import sys
import logging
import time
import numpy as np
import cv2

from last import MMK2RealRobot  # 使用last.py（已适配当前环境）

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def main():
    import argparse
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='实时展示相机图像（无检测功能）')
    
    parser.add_argument('--robot_ip', type=str, required=False, default='192.168.11.200')
    parser.add_argument('--head_pitch', type=float, default=-0.5236,
                       help='相机头部俯仰角度（弧度），默认-30度')
    parser.add_argument('--fps', type=int, default=30,
                       help='显示帧率')
    
    args = parser.parse_args()
    
    # 初始化
    print("="*70)
    print("📹 实时相机图像展示")
    print("="*70)
    print(f"机器人IP: {args.robot_ip}")
    print(f"头部俯仰角: {np.degrees(args.head_pitch):.1f}°")
    print("="*70 + "\n")
    
    logging.info("正在连接机器人...")
    robot = MMK2RealRobot(ip=args.robot_ip)
    time.sleep(2.0)
    
    # 设置头部角度
    logging.info(f"设置头部角度: {np.degrees(args.head_pitch):.1f}°")
    robot.set_robot_head_pose(0.0, args.head_pitch)
    time.sleep(1.0)
    
    # 获取相机
    camera = robot.camera
    time.sleep(1.0)
    
    logging.info("✓ 初始化完成")
    print("\n" + "="*70)
    print("📹 实时图像展示中...")
    print("  按 'q' 键退出")
    print("  按 's' 键保存当前帧")
    print("="*70 + "\n")
    
    frame_count = 0
    save_count = 0
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    try:
        # 实时显示循环
        for img_head, img_depth, _, _ in camera:
            if img_head is None or img_depth is None:
                continue
            
            frame_count += 1
            
            # 复制彩色图像
            color = img_head.copy()
            
            # 深度图转换为可视化图像
            depth = img_depth.astype(np.float32) / 1000.0  # 转换为米
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(img_depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 在彩色图像上添加信息
            cv2.putText(color, f"Frame: {frame_count}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color, "Press 'q' to Quit", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(color, "Press 's' to Save", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 在深度图上添加信息
            cv2.putText(depth_colormap, f"Depth Frame: {frame_count}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 并排显示彩色图像和深度图
            combined = np.hstack([color, depth_colormap])
            
            # 显示图像
            cv2.imshow("Camera View (Color + Depth)", combined)
            
            # 处理按键
            key = cv2.waitKey(1000 // args.fps) & 0xFF
            
            if key == ord('q'):
                logging.info("退出程序")
                break
            
            elif key == ord('s'):
                # 保存当前帧
                save_count += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                color_file = f'{code_dir}/saved_color_{timestamp}.png'
                depth_file = f'{code_dir}/saved_depth_{timestamp}.png'
                combined_file = f'{code_dir}/saved_combined_{timestamp}.png'
                
                cv2.imwrite(color_file, img_head)
                cv2.imwrite(depth_file, depth_colormap)
                cv2.imwrite(combined_file, combined)
                
                logging.info(f"✓ 已保存第 {save_count} 组图像:")
                logging.info(f"  - {color_file}")
                logging.info(f"  - {depth_file}")
                logging.info(f"  - {combined_file}")
                
    except KeyboardInterrupt:
        logging.info("程序被中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    cv2.destroyAllWindows()
    logging.info(f"程序结束 (共显示 {frame_count} 帧)")


if __name__ == '__main__':
    main()

