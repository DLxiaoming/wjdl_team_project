import numpy as np
import transformations as tf

def transform_matrix(translation, rotation):
    """将平移和旋转转换为4x4变换矩阵"""
    # 从四元数创建旋转矩阵 (x, y, z, w) -> (w, x, y, z)
    rot_matrix = tf.quaternion_matrix([
        rotation['w'], 
        rotation['x'], 
        rotation['y'], 
        rotation['z']
    ])
    
    # 设置平移部分
    rot_matrix[0, 3] = translation['x']
    rot_matrix[1, 3] = translation['y']
    rot_matrix[2, 3] = translation['z']
    
    return rot_matrix

def main():
    # 从数据中提取各个变换关系（第一帧数据）
    transforms = {
        # base_link到slide_link的变换
        'base_to_slide': {
            'translation': {'x': 0.03393979587858495, 'y': -0.00025000000809619, 'z': 1.105973000008096},
            'rotation': {'x': 0.7071054825064662, 'y': 0.7071080798547033, 'z': -2.5973530075574157e-06, 'w': -2.5973434669646155e-06}
        },
        # slide_link到head_yaw_link的变换
        'slide_to_head_yaw': {
            'translation': {'x': 0.0, 'y': 0.1735, 'z': 0.07242},
            'rotation': {'x': 0.9999999954456416, 'y': 9.536888668693756e-05, 'z': 3.5030948128130356e-10, 'w': -3.6732050866422617e-06}
        },
        # head_yaw_link到head_pitch_link的变换
        'head_yaw_to_pitch': {
            'translation': {'x': 0.00099952, 'y': 3.1059e-05, 'z': 0.058},
            'rotation': {'x': 0.6457243924223424, 'y': -0.28816982082332726, 'z': 0.6364502991788389, 'w': -0.30810579363688534}
        }
    }
    
    # 计算各个变换的矩阵
    mat_base_to_slide = transform_matrix(
        transforms['base_to_slide']['translation'],
        transforms['base_to_slide']['rotation']
    )
    
    mat_slide_to_head_yaw = transform_matrix(
        transforms['slide_to_head_yaw']['translation'],
        transforms['slide_to_head_yaw']['rotation']
    )
    
    mat_head_yaw_to_pitch = transform_matrix(
        transforms['head_yaw_to_pitch']['translation'],
        transforms['head_yaw_to_pitch']['rotation']
    )
    
    # 计算从base_link到head_pitch_link的总变换矩阵
    # 矩阵乘法顺序是从右到左的，因为变换是复合的
    mat_base_to_head_pitch = mat_base_to_slide @ mat_slide_to_head_yaw @ mat_head_yaw_to_pitch
    
    # 提取平移部分（单位：米）
    translation = {
        'x': mat_base_to_head_pitch[0, 3],
        'y': mat_base_to_head_pitch[1, 3],
        'z': mat_base_to_head_pitch[2, 3]
    }
    
    # 提取旋转部分（四元数）
    rot_quat = tf.quaternion_from_matrix(mat_base_to_head_pitch)
    rotation = {
        'w': rot_quat[0],
        'x': rot_quat[1],
        'y': rot_quat[2],
        'z': rot_quat[3]
    }
    
    # 转换为欧拉角（弧度），方便理解
    euler = tf.euler_from_quaternion(rot_quat)
    euler_deg = [np.degrees(angle) for angle in euler]  # 转换为度
    
    print("头部(head_pitch_link)相对于底盘(base_link)的位姿:")
    print(f"平移: x={translation['x']:.6f}m, y={translation['y']:.6f}m, z={translation['z']:.6f}m")
    print(f"旋转(四元数): w={rotation['w']:.6f}, x={rotation['x']:.6f}, y={rotation['y']:.6f}, z={rotation['z']:.6f}")
    print(f"旋转(欧拉角): 滚转={euler_deg[0]:.2f}°, 俯仰={euler_deg[1]:.2f}°, 偏航={euler_deg[2]:.2f}°")

if __name__ == "__main__":
    main()
