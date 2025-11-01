import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import time

def create_folder(folder):
    """创建文件夹，若已存在则清空内容"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除文件失败 {file_path}: {e}")

class RealsenseImageCapture:
    def __init__(self, data_root, total_images=30, sample_rate=1.0):
        # 配置参数
        self.total_images = total_images  # 目标采集数量
        self.collected_count = 0  # 已采集数量
        self.sample_rate = sample_rate  # 采样率(帧/秒)
        self.min_interval = 1.0 / sample_rate if sample_rate > 0 else 0
        self.last_save_time = 0
        
        # 路径设置
        self.data_root = os.path.abspath(data_root)
        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.depth_dir = os.path.join(self.data_root, "depth")
        
        # 初始化文件夹
        create_folder(self.rgb_dir)
        create_folder(self.depth_dir)
        print(f"数据保存根路径: {self.data_root}")
        print(f"RGB图像路径: {self.rgb_dir}")
        print(f"深度图像路径: {self.depth_dir}")
        
        # 初始化Realsense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置流格式
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # 深度流
        
        # 深度对齐到RGB
        self.align = rs.align(rs.stream.color)
        
        # 启动相机
        try:
            self.pipeline.start(self.config)
            print(f"相机启动成功 | 目标采集: {total_images}张 | 采样率: {sample_rate}帧/秒")
        except RuntimeError as e:
            print(f"相机启动失败: {e}")
            raise

    def start_capture(self):
        try:
            print("开始采集... (按Ctrl+C可提前终止)")
            while self.collected_count < self.total_images:
                current_time = time.time()
                
                # 获取帧数据
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames:
                    print("获取帧超时，重试...")
                    continue
                
                # 对齐深度帧到RGB帧
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                # 检查帧有效性
                if not depth_frame or not color_frame:
                    continue
                
                # 采样控制
                if current_time - self.last_save_time >= self.min_interval:
                    # 获取纳秒级时间戳(与ROS格式一致)
                    timestamp_ns = int(color_frame.get_timestamp() * 1000000)  # 毫秒转纳秒
                    timestamp_str = str(timestamp_ns)
                    
                    # 转换为numpy数组
                    color_img = np.asanyarray(color_frame.get_data())  # BGR8格式
                    depth_img = np.asanyarray(depth_frame.get_data())  # 16位深度值(mm)
                    
                    # 保存RGB图像
                    rgb_path = os.path.join(self.rgb_dir, f"{timestamp_str}.png")
                    cv2.imwrite(rgb_path, color_img)
                    
                    # 保存深度图像(16UC1格式)
                    depth_path = os.path.join(self.depth_dir, f"{timestamp_str}.png")
                    cv2.imwrite(depth_path, depth_img)
                    
                    # 更新计数
                    self.collected_count += 1
                    print(f"已采集 {self.collected_count}/{self.total_images} | 时间戳: {timestamp_str}")
                    self.last_save_time = current_time

            print(f"\n采集完成! 共保存 {self.collected_count} 组图像")

        except KeyboardInterrupt:
            print(f"\n用户终止采集，已保存 {self.collected_count} 组图像")
        finally:
            # 释放资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("相机已关闭，程序退出")

if __name__ == "__main__":
    # 配置参数
    DATA_ROOT = "demo_data/grasp_8"  # 数据保存根路径
    TOTAL_IMAGES = 60                 # 采集总数
    SAMPLE_RATE = 10.0                 # 采样率(帧/秒)

    # 启动采集
    capture = RealsenseImageCapture(
        data_root=DATA_ROOT,
        total_images=TOTAL_IMAGES,
        sample_rate=SAMPLE_RATE
    )
    capture.start_capture()
