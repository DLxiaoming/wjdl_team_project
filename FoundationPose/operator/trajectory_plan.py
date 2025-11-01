import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

# -------------------------- 1. 单独的姿态插值函数（仅需初始/目标2个姿态） --------------------------
def slerp(rot1, rot2, t):
    """球面线性插值（兼容旧版SciPy）"""
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    dot = np.dot(q1, q2)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        return R.from_quat(q / np.linalg.norm(q))
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    q = s1 * q1 + s2 * q2
    return R.from_quat(q)

def generate_smooth_rotation(num_points, start_rot, end_rot):
    """
    生成平滑的姿态轨迹（仅需初始/目标2个姿态）
    参数:
        num_points: 生成的姿态数量（需与位置轨迹点数量一致）
        start_rot: 初始姿态（Rotation对象）
        end_rot: 目标姿态（Rotation对象）
    返回:
        quat_trajectory: 姿态轨迹（n个四元数，shape=(num_points, 4)）
    """
    quat_trajectory = []
    for t in np.linspace(0, 1, num_points):
        interp_rot = slerp(start_rot, end_rot, t)
        quat_trajectory.append(interp_rot.as_quat())
    return np.array(quat_trajectory)

# -------------------------- 2. 单独的位置轨迹生成函数（支持多关键点） --------------------------
def generate_smooth_position(waypoints_pos, num_points=200):
    """
    生成平滑的位置轨迹（支持任意数量关键点）
    参数:
        waypoints_pos: 位置关键点列表（shape=(m, 3)，m≥1）
        num_points: 生成的位置轨迹点数量
    返回:
        pos_trajectory: 位置轨迹（shape=(num_points, 3)）
    """
    waypoints_pos = np.array(waypoints_pos)
    if waypoints_pos.ndim != 2 or waypoints_pos.shape[1] != 3:
        raise ValueError("waypoints_pos必须是形状为(m, 3)的二维数组")
    
    if len(waypoints_pos) == 1:
        return np.tile(waypoints_pos, (num_points, 1))
    
    # B样条插值（经过所有位置关键点）
    m_pos = len(waypoints_pos)
    k_pos = min(3, m_pos - 1)
    tck_pos, u_pos = splprep(waypoints_pos.T, s=0, k=k_pos)
    u_pos_new = np.linspace(u_pos.min(), u_pos.max(), num_points)
    pos_trajectory = np.array(splev(u_pos_new, tck_pos)).T
    return pos_trajectory

# -------------------------- 3. 位姿合并函数（位置+姿态） --------------------------
def merge_position_rotation(pos_trajectory, quat_trajectory):
    """
    合并位置轨迹与姿态轨迹，生成完整位姿轨迹
    参数:
        pos_trajectory: 位置轨迹（shape=(n, 3)）
        quat_trajectory: 姿态轨迹（shape=(n, 4)）
    返回:
        pose_trajectory: 完整位姿轨迹（shape=(n, 7)，每个元素[px,py,pz,qx,qy,qz,qw]）
    """
    if len(pos_trajectory) != len(quat_trajectory):
        raise ValueError("位置轨迹与姿态轨迹的点数量必须一致")
    return np.hstack((pos_trajectory, quat_trajectory))

# -------------------------- 4. 箭头起点计算（不变） --------------------------
def calculate_arrow_start(pose, arrow_length=0.05, trajectory_offset=0.04):
    """计算箭头起点（轨迹点是箭头4cm处）"""
    traj_pos = np.array(pose[:3])
    rot = R.from_quat(pose[3:])
    arrow_dir = rot.as_matrix()[:, 0]  # 物体X轴为箭头朝向
    arrow_dir = arrow_dir / np.linalg.norm(arrow_dir)
    arrow_start = traj_pos - arrow_dir * trajectory_offset
    return arrow_start, arrow_dir

# -------------------------- 5. 可视化函数（修改箭头样式：身体蓝、头部红） --------------------------
def plot_pose_trajectory(waypoints_pos, start_rot, end_rot, pose_trajectory, arrow_length=0.05):
    """绘制位姿轨迹（确保坐标轴量度一致，补上终点位姿箭头）"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    traj_pos = pose_trajectory[:, :3]
    all_pos = np.vstack((waypoints_pos, traj_pos))

    # 1. 全局坐标系（X红、Y绿、Z蓝）
    coord_max = np.max(np.abs(all_pos)) * 0.2
    ax.quiver(0, 0, 0, coord_max, 0, 0, color='r', linewidth=2, label='全局X轴')
    ax.quiver(0, 0, 0, 0, coord_max, 0, color='g', linewidth=2, label='全局Y轴')
    ax.quiver(0, 0, 0, 0, 0, coord_max, color='b', linewidth=2, label='全局Z轴')

    # 2. 平滑轨迹线（青色）
    ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2], 
            'c-', linewidth=2, alpha=0.7, label='平滑轨迹（箭头4cm处）')

    # 3. 位置关键点（红圆+黄星终点）
    if len(waypoints_pos) > 1:
        ax.scatter(waypoints_pos[:-1, 0], waypoints_pos[:-1, 1], waypoints_pos[:-1, 2], 
                   c='r', s=120, marker='o', edgecolors='black', label='位置关键点')
    ax.scatter(waypoints_pos[-1, 0], waypoints_pos[-1, 1], waypoints_pos[-1, 2], 
               c='y', s=180, marker='*', edgecolors='black', linewidth=2, label='位置终点')

    # 4. 位姿箭头（身体蓝、头部红）：正常绘制轨迹中间箭头
    step = max(1, len(pose_trajectory) // 20)
    for i in range(0, len(pose_trajectory), step):
        pose = pose_trajectory[i]
        arrow_start, arrow_dir = calculate_arrow_start(pose)
        arrow_vec = arrow_dir * arrow_length

        # 箭头身体（蓝色，无头部）
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                  arrow_vec[0], arrow_vec[1], arrow_vec[2],
                  color='blue', linewidth=2.5, arrow_length_ratio=0, alpha=0.8)
        
        # 箭头头部（红色）
        head_len = arrow_length * 0.2
        head_start = arrow_start + arrow_vec * (1 - 0.2)
        head_vec = arrow_dir * head_len
        ax.quiver(head_start[0], head_start[1], head_start[2],
                  head_vec[0], head_vec[1], head_vec[2],
                  color='red', linewidth=3, arrow_length_ratio=0.3, alpha=1.0)

    # -------------------------- 新增：单独绘制终点位姿箭头（确保不遗漏） --------------------------
    if len(pose_trajectory) > 0:
        end_pose = pose_trajectory[-1]  # 提取轨迹最后一个位姿（终点位姿）
        end_arrow_start, end_arrow_dir = calculate_arrow_start(end_pose)
        end_arrow_vec = end_arrow_dir * arrow_length

        # 终点箭头：身体用深蓝色（更粗），头部用橙色（醒目，与普通箭头区分）
        # 终点箭头身体
        ax.quiver(end_arrow_start[0], end_arrow_start[1], end_arrow_start[2],
                  end_arrow_vec[0], end_arrow_vec[1], end_arrow_vec[2],
                  color='darkblue', linewidth=4, arrow_length_ratio=0, alpha=1.0, label='终点位姿')
        
        # 终点箭头头部
        end_head_len = arrow_length * 0.25  # 终点头部更长，更醒目
        end_head_start = end_arrow_start + end_arrow_vec * (1 - 0.25)
        end_head_vec = end_arrow_dir * end_head_len
        ax.quiver(end_head_start[0], end_head_start[1], end_head_start[2],
                  end_head_vec[0], end_head_vec[1], end_head_vec[2],
                  color='orange', linewidth=5, arrow_length_ratio=0.4, alpha=1.0)

    # 5. 调整坐标轴范围：确保X、Y、Z轴量度一致
    all_elements = np.vstack((all_pos, [[0,0,0]]))  # 包含原点
    x_range = np.ptp(all_elements[:, 0])
    y_range = np.ptp(all_elements[:, 1])
    z_range = np.ptp(all_elements[:, 2])
    
    max_range = max(x_range, y_range, z_range) * 2  
    mid_x = np.mean(all_elements[:, 0])
    mid_y = np.mean(all_elements[:, 1])
    mid_z = np.mean(all_elements[:, 2])
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.set_box_aspect([1, 1, 1])  # 强制三轴等比例

    # 6. 标签与图例（新增终点位姿标签）
    ax.set_xlabel('X轴 (m)', fontsize=12)
    ax.set_ylabel('Y轴 (m)', fontsize=12)
    ax.set_zlabel('Z轴 (m)', fontsize=12)
    ax.set_title('三维位姿平滑轨迹（含终点位姿，坐标轴量度一致）', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.show()
    return fig, ax

# -------------------------- 6. 主函数（按“位置生成→姿态生成→合并→可视化”流程调用） --------------------------
def main():
    """主流程：多位置关键点 + 仅2个姿态关键点 → 生成完整位姿轨迹"""
    # 1. 定义输入：多位置关键点 + 仅初始/目标2个姿态关键点
    ## 1.1 位置关键点（3个：grasp → inset_on_ready → inset）
    init = [0.6071971944073851, 0.15423178989249392, 0.8627445894164919, 0.11923737406360825, 0.1825040434325132, -0.6101763468285463, 0.7616820521242235]
    grasp_for_ready = []
    grasp = [0.6990480979915086, 0.04203325186609186, 0.8323758826672858, 0.010351212968842217, 0.07404863192388889, -0.4628662680264759, 0.8832691947665313]
    safe_point = [0.7, 0.2, 0.9]
    inset_on_ready = [0.8219137684732842, 0.2165218772179783, 1.02292169762169, 0.9934553949641332, 0.06257672309462979, 0.09442909409380326, 0.014617733477523414]
    inset = [0.8219137684732842, 0.2165218772179783, 1.05292169762169, 0.9934553949641332, 0.06257672309462979, 0.09442909409380326, 0.014617733477523414]    
    waypoints_pos = [
        grasp[0:3],
        safe_point,
        inset_on_ready[0:3],
        # inset[0:3],
    ]
    waypoints_pos = np.array(waypoints_pos)
    ## 1.2 姿态关键点（仅2个：初始姿态 → 目标姿态）
    start_rot = R.from_euler('zyx', [0, -np.pi/2, 0])  # 初始姿态（Euler角转Rotation）
    end_rot = R.from_euler('zyx', [0, np.pi/2, 0])      # 目标姿态（Euler角转Rotation）

    # 2. 生成轨迹（分开生成位置和姿态，再合并）
    num_points = 200  # 轨迹总点数
    pos_trajectory = generate_smooth_position(waypoints_pos, num_points=num_points)  # 位置轨迹
    quat_trajectory = generate_smooth_rotation(num_points, start_rot, end_rot)       # 姿态轨迹
    pose_trajectory = merge_position_rotation(pos_trajectory, quat_trajectory)       # 合并为完整位姿

    # 3. 可视化
    plot_pose_trajectory(waypoints_pos, start_rot, end_rot, pose_trajectory)

if __name__ == "__main__":
    main()