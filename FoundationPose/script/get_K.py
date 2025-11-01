import pyrealsense2 as rs

def get_rgb_intrinsics_scientific():
    # 初始化管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用RGB流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动管道
        pipeline.start(config)
        
        # 等待一帧以确保相机已初始化
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("无法获取RGB帧")
            return
        
        # 获取内参
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        
        # 按指定科学计数法格式打印内参矩阵
        print("RGB相机内参矩阵:")
        print(f"{intrinsics.fx:23.18e} {0.0:23.18e} {intrinsics.ppx:23.18e}")
        print(f"{0.0:23.18e} {intrinsics.fy:23.18e} {intrinsics.ppy:23.18e}")
        print(f"{0.0:23.18e} {0.0:23.18e} {1.0:23.18e}")
        
        return intrinsics
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    get_rgb_intrinsics_scientific()
