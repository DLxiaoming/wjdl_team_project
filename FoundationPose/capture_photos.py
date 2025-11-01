#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时拍照并存储
按空格键拍照，自动保存到指定文件夹
"""

import os
import sys
import logging
import time
import numpy as np
import cv2
from datetime import datetime

from last import MMK2RealRobot

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def main():
    import argparse
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='实时拍照并存储')
    
    parser.add_argument('--robot_ip', type=str, default='192.168.11.200')
    parser.add_argument('--head_pitch', type=float, default=-0.5236,
                       help='相机头部俯仰角度（弧度），默认-30度')
    parser.add_argument('--save_dir', type=str, default='captured_photos',
                       help='照片保存目录')
    parser.add_argument('--fps', type=int, default=30,
                       help='显示帧率')
    
    args = parser.parse_args()
    
    # 创建保存目录
    code_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(code_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化
    print("="*70)
    print("📸 实时拍照系统")
    print("="*70)
    print(f"机器人IP: {args.robot_ip}")
    print(f"头部俯仰角: {np.degrees(args.head_pitch):.1f}°")
    print(f"保存目录: {save_dir}")
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
    print("📸 拍照系统已就绪")
    print("="*70)
    print("  按 空格键 - 拍照并保存")
    print("  按 'q' 键 - 退出")
    print("="*70 + "\n")
    
    frame_count = 0
    photo_count = 0
    
    try:
        # 实时显示循环
        for img_head, img_depth, _, _ in camera:
            if img_head is None or img_depth is None:
                continue
            
            frame_count += 1
            
            # 复制图像
            color = img_head.copy()
            depth = img_depth.astype(np.float32) / 1000.0
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(img_depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 在图像上添加信息
            cv2.putText(color, f"Frame: {frame_count}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color, f"Photos: {photo_count}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(color, "Press SPACE to Capture", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(color, "Press 'q' to Quit", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 在深度图上添加信息
            cv2.putText(depth_colormap, f"Depth - Photos: {photo_count}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 并排显示
            combined = np.hstack([color, depth_colormap])
            
            # 显示图像
            cv2.imshow("Capture Photos (Press SPACE)", combined)
            
            # 处理按键
            key = cv2.waitKey(1000 // args.fps) & 0xFF
            
            if key == ord('q'):
                logging.info("退出程序")
                break
            
            elif key == ord(' '):  # 空格键
                # 拍照并保存
                photo_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 定义文件名
                color_file = os.path.join(save_dir, f'photo_{photo_count:04d}_{timestamp}_color.png')
                depth_file = os.path.join(save_dir, f'photo_{photo_count:04d}_{timestamp}_depth.png')
                depth_raw_file = os.path.join(save_dir, f'photo_{photo_count:04d}_{timestamp}_depth_raw.png')
                combined_file = os.path.join(save_dir, f'photo_{photo_count:04d}_{timestamp}_combined.png')
                
                # 保存图像
                cv2.imwrite(color_file, img_head)
                cv2.imwrite(depth_file, depth_colormap)
                cv2.imwrite(depth_raw_file, img_depth)  # 原始深度数据
                cv2.imwrite(combined_file, combined)
                
                # 在图像上显示"已拍照"提示
                capture_img = combined.copy()
                cv2.putText(capture_img, f"CAPTURED #{photo_count}!", (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cv2.imshow("Capture Photos (Press SPACE)", capture_img)
                cv2.waitKey(500)  # 显示0.5秒
                
                # 终端输出
                print("\n" + "="*70)
                print(f"📸 已拍照 #{photo_count}")
                print("="*70)
                print(f"✓ 彩色图: {os.path.basename(color_file)}")
                print(f"✓ 深度图: {os.path.basename(depth_file)}")
                print(f"✓ 深度原始: {os.path.basename(depth_raw_file)}")
                print(f"✓ 合并图: {os.path.basename(combined_file)}")
                print(f"保存位置: {save_dir}")
                print("="*70 + "\n")
                
    except KeyboardInterrupt:
        logging.info("程序被中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    cv2.destroyAllWindows()
    
    # 总结
    print("\n" + "="*70)
    print("📊 拍照统计")
    print("="*70)
    print(f"总帧数: {frame_count}")
    print(f"拍照数: {photo_count}")
    print(f"保存目录: {save_dir}")
    print("="*70)
    logging.info("程序结束")


if __name__ == '__main__':
    main()








