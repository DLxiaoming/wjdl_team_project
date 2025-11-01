import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    # 初始化Realsense管道
    pipeline = rs.pipeline()
    
    # 创建配置对象并配置数据流
    config = rs.config()
    
    # 配置RGB流：分辨率640x480，格式BGR，帧率30
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 配置深度流：分辨率640x480，格式16位深度，帧率30
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动相机流
    print("启动相机...")
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"启动相机失败: {e}")
        print("请确保没有其他程序占用相机，并检查连接")
        return
    
    # 创建对齐对象，将深度图对齐到RGB图
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 提前创建窗口，避免循环中重复创建
    window_name = 'Realsense D435i - RGB和深度图'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 设置窗口初始大小，避免过大导致卡顿
    cv2.resizeWindow(window_name, 1280, 480)  # 640*2 宽度，480高度
    
    try:
        print("开始获取图像，按'q'键或ESC键退出...")
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 等待获取一帧数据，设置超时时间避免无限阻塞
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if not frames:
                print("获取帧超时")
                continue
            
            # 将深度帧对齐到RGB帧
            aligned_frames = align.process(frames)
            
            # 提取对齐后的深度帧和RGB帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # 检查帧是否有效
            if not depth_frame or not color_frame:
                continue
            
            # 计算帧率（每10帧计算一次）
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"帧率: {fps:.1f} FPS", end='\r')
            
            # 将帧数据转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 为深度图应用颜色映射以便可视化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),  # 缩放深度值以适应8位显示
                cv2.COLORMAP_JET
            )
            
            # 获取图像尺寸并拼接显示
            images = np.hstack((color_image, depth_colormap))
            
            # 显示图像（只使用一个窗口）
            cv2.imshow(window_name, images)
            
            # 按'q'键或ESC键退出循环
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 27是ESC键
                print("\n用户请求退出")
                break

    except Exception as e:
        print(f"运行过程中出错: {e}")
    finally:
        # 确保资源正确释放
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已停止，所有窗口已关闭")

if __name__ == "__main__":
    main()
