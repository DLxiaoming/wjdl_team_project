import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体（优先使用 SimHei，没有则改用 Noto Sans CJK）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'Heiti TC']  # 依次尝试可用字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


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

# -------------------------- 新增：由物件 z 与偏好 x 生成完整局部坐标系 --------------------------
def compute_frame_from_z_and_preferred_x(z_vec, preferred_x):
    """
    给定 z 向量（全局坐标系下）和偏好 x 向量（全局坐标系下），返回一个右手坐标系的旋转矩阵 R (3x3)
    原理：将 preferred_x 投影到与 z 垂直的平面上并归一化得到 x，然后 y = z × x，保证 x,y,z 两两正交且右手规则。
    若投影接近零（preferred_x 与 z 平行或接近平行），则选择一个任意垂直方向作为 x。
    返回：3x3 矩阵，列为 x,y,z（三个轴在全局系下的表示）
    """
    z = np.array(z_vec, dtype=float)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-8:
        raise ValueError("输入的 z 向量长度太小")
    z = z / z_norm

    pref = np.array(preferred_x, dtype=float)
    # 在与 z 垂直的平面上投影 pref
    pref_proj = pref - np.dot(pref, z) * z
    proj_norm = np.linalg.norm(pref_proj)
    if proj_norm < 1e-6:
        # 投影太小，选择任何与 z 垂直的向量作为 x
        # 用与 z 不共线的基矢量（如 [1,0,0] 或 [0,1,0]）叉乘得到
        if abs(z[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
        x = x / np.linalg.norm(x)
    else:
        x = pref_proj / proj_norm

    # 确保正交且右手：先计算 y = z x x，然后重新正交化 x = y x z 保证一致
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)

    Rmat = np.column_stack((x, y, z))
    return Rmat

def generate_object_frames_from_pose_trajectory(
    pose_trajectory, 
    initial_x_dir=np.array([0.0, -1.0, 0.0]),
    final_x_dir=np.array([0.0, 1.0, 0.0])
):
    """
    根据物件位姿轨迹生成每帧的工件局部坐标系（允许初末 x 方向不同）
    参数：
        pose_trajectory: n×7 的位姿轨迹（每帧 [px,py,pz,qx,qy,qz,qw]）
        initial_x_dir: 起始时偏好 x（全局坐标）
        final_x_dir: 结束时偏好 x（全局坐标）
    说明：
        - 从 pose_trajectory 中提取每帧的 z（来自 slerp 内插的姿态）
        - 对 x_dir 在全局系中做线性插值（逐渐过渡）
        - 用 z 和当前 x 生成正交坐标系（右手系）
    返回：
        frames: n×7，每帧的局部坐标系位姿
    """
    n = len(pose_trajectory)
    frames = np.zeros((n, 7))

    initial_x = np.array(initial_x_dir, dtype=float)
    final_x = np.array(final_x_dir, dtype=float)

    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        pos = pose_trajectory[i, :3]
        quat = pose_trajectory[i, 3:]
        rot_obj = R.from_quat(quat)
        z_vec = rot_obj.as_matrix()[:, 2]  # 当前帧 z 方向（由slerp生成）

        # 当前帧的x偏好方向：线性插值 + 归一化
        interp_x = (1 - t) * initial_x + t * final_x
        interp_x = interp_x / np.linalg.norm(interp_x)

        # 用z和插值x生成正交右手坐标系
        Rmat = compute_frame_from_z_and_preferred_x(z_vec, interp_x)
        rot_frame = R.from_matrix(Rmat)

        frames[i, :3] = pos
        frames[i, 3:] = rot_frame.as_quat()

    return frames


# -------------------------- 绘制单个局部坐标系（三个箭头） --------------------------
def draw_frame(ax, origin, Rmat, axis_length=0.04, linewidth=2.5, alpha=1.0):
    """
    在 ax（三维坐标轴）上画一个局部坐标系：
    origin: (3,)
    Rmat: 3x3 矩阵，列向量为 x, y, z 在全局坐标系下表示
    """
    x = Rmat[:, 0] * axis_length
    y = Rmat[:, 1] * axis_length
    z = Rmat[:, 2] * axis_length
    ox, oy, oz = origin

    # X 轴（红）
    ax.quiver(ox, oy, oz, x[0], x[1], x[2], linewidth=linewidth, arrow_length_ratio=0.2, alpha=alpha, color='r')
    # Y 轴（绿）
    ax.quiver(ox, oy, oz, y[0], y[1], y[2], linewidth=linewidth, arrow_length_ratio=0.2, alpha=alpha, color='g')
    # Z 轴（蓝）
    ax.quiver(ox, oy, oz, z[0], z[1], z[2], linewidth=linewidth, arrow_length_ratio=0.2, alpha=alpha, color='b')

# -------------------------- 4. 箭头起点计算（修正：显示物件 Z 轴方向） --------------------------
def calculate_arrow_start(pose, arrow_length=0.05, trajectory_offset=0.04):
    """计算箭头起点（轨迹点是箭头4cm处）
       修正：用物件的局部 Z 轴（第三列）作为显示方向，确保与生成的小坐标系 z 方向一致
    """
    traj_pos = np.array(pose[:3])
    rot = R.from_quat(pose[3:])
    arrow_dir = rot.as_matrix()[:, 2]  # 使用 Z 轴
    arrow_dir = arrow_dir / np.linalg.norm(arrow_dir)
    arrow_start = traj_pos - arrow_dir * trajectory_offset
    return arrow_start, arrow_dir

# -------------------------- 5. 可视化函数（绘制轨迹与每帧小坐标系） --------------------------
def plot_pose_trajectory_with_frames(waypoints_pos, start_rot, end_rot, pose_trajectory,
                                     frames=None, frame_step=1, axis_length=0.04, arrow_length=0.05):
    """
    绘制轨迹与每帧的局部坐标系：
      - waypoints_pos: 输入的关键点（用于绘制关键点）
      - pose_trajectory: 原始物件 pose 轨迹（用于绘制轨迹线）
      - frames: 若提供，为每帧计算得到的小坐标系（n x 7）
      - frame_step: 每隔多少帧绘制一个坐标系（避免太拥挤）
      - axis_length: 小坐标系轴长度
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    traj_pos = pose_trajectory[:, :3]
    all_pos = np.vstack((waypoints_pos, traj_pos))

    # 全局坐标系（X红、Y绿、Z蓝）
    coord_max = np.max(np.abs(all_pos)) * 0.2 if np.max(np.abs(all_pos))>0 else 0.2
    ax.quiver(0, 0, 0, coord_max, 0, 0, color='r', linewidth=2, label='全局X轴')
    ax.quiver(0, 0, 0, 0, coord_max, 0, color='g', linewidth=2, label='全局Y轴')
    ax.quiver(0, 0, 0, 0, 0, coord_max, color='b', linewidth=2, label='全局Z轴')

    # 轨迹线
    ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2], 'c-', linewidth=2, alpha=0.7, label='平滑轨迹（原点在4cm处）')

    # 位置关键点显示
    if len(waypoints_pos) > 1:
        ax.scatter(waypoints_pos[:-1, 0], waypoints_pos[:-1, 1], waypoints_pos[:-1, 2], 
                   c='r', s=120, marker='o', edgecolors='black', label='位置关键点')
    ax.scatter(waypoints_pos[-1, 0], waypoints_pos[-1, 1], waypoints_pos[-1, 2], 
               c='y', s=180, marker='*', edgecolors='black', linewidth=2, label='位置终点')

    # 若提供 frames，则绘制每帧小坐标系
    if frames is not None:
        n = len(frames)
        step = max(1, frame_step)
        for i in range(0, n, step):
            pos = frames[i, :3]
            rot = R.from_quat(frames[i, 3:])
            Rmat = rot.as_matrix()
            # 绘制小坐标系（x红，y绿，z蓝）
            draw_frame(ax, pos, Rmat, axis_length=axis_length, linewidth=2.0, alpha=0.9)
        # 画最后一帧加粗以便识别
        pos_end = frames[-1, :3]
        Rmat_end = R.from_quat(frames[-1, 3:]).as_matrix()
        draw_frame(ax, pos_end, Rmat_end, axis_length=axis_length*1.6, linewidth=3.5, alpha=1.0)

    # 另外为可视化保留 “旧样式”的箭头（可选）：显示物件Z方向箭头（蓝->红头）
    step2 = max(1, len(pose_trajectory)//20)
    for i in range(0, len(pose_trajectory), step2):
        pose = pose_trajectory[i]
        arrow_start, arrow_dir = calculate_arrow_start(pose)
        arrow_vec = arrow_dir * arrow_length
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                  arrow_vec[0], arrow_vec[1], arrow_vec[2],
                  color='grey', linewidth=5, arrow_length_ratio=0, alpha=0.5)
        head_len = arrow_length * 0.2
        head_start = arrow_start + arrow_vec * (1 - 0.2)
        head_vec = arrow_dir * head_len
        ax.quiver(head_start[0], head_start[1], head_start[2],
                  head_vec[0], head_vec[1], head_vec[2],
                  color='black', linewidth=5, arrow_length_ratio=0.3, alpha=0.8)
    # -------------------------- 新增：单独绘制终点位姿箭头（确保不遗漏） --------------------------
    if len(pose_trajectory) > 0:
        end_pose = pose_trajectory[-1]  # 提取轨迹最后一个位姿（终点位姿）
        end_arrow_start, end_arrow_dir = calculate_arrow_start(end_pose)
        end_arrow_vec = end_arrow_dir * arrow_length

        # 终点箭头：身体用深蓝色（更粗），头部用橙色（醒目，与普通箭头区分）
        # 终点箭头身体
        ax.quiver(end_arrow_start[0], end_arrow_start[1], end_arrow_start[2],
                  end_arrow_vec[0], end_arrow_vec[1], end_arrow_vec[2],
                  color='black', linewidth=5, arrow_length_ratio=0, alpha=1.0, label='终点位姿')
        
        # 终点箭头头部
        end_head_len = arrow_length * 0.25  # 终点头部更长，更醒目
        end_head_start = end_arrow_start + end_arrow_vec * (1 - 0.25)
        end_head_vec = end_arrow_dir * end_head_len
        ax.quiver(end_head_start[0], end_head_start[1], end_head_start[2],
                  end_head_vec[0], end_head_vec[1], end_head_vec[2],
                  color='orange', linewidth=5, arrow_length_ratio=0.4, alpha=1.0)


    # 调整坐标轴范围：保持等比例
    all_elements = np.vstack((all_pos, [[0,0,0]]))
    x_range = np.ptp(all_elements[:, 0])
    y_range = np.ptp(all_elements[:, 1])
    z_range = np.ptp(all_elements[:, 2])
    max_range = max(x_range, y_range, z_range) * 2 if max(x_range, y_range, z_range)>0 else 1.0
    mid_x = np.mean(all_elements[:, 0])
    mid_y = np.mean(all_elements[:, 1])
    mid_z = np.mean(all_elements[:, 2])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    ax.set_box_aspect([1,1,1])

    ax.set_xlabel('X轴 (m)', fontsize=12)
    ax.set_ylabel('Y轴 (m)', fontsize=12)
    ax.set_zlabel('Z轴 (m)', fontsize=12)
    ax.set_title('轨迹与每帧工件局部坐标系可视化', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.show()
    return fig, ax


def run(point_list, x_dir=[]):
    grasp = point_list[0]
    safe_point = point_list[1]
    inset_on_ready = point_list[2]
    waypoints_pos = [
        grasp[0:3],
        safe_point[0:3],
        inset_on_ready[0:3],
    ]
    waypoints_pos = np.array(waypoints_pos)

    # 姿态关键点（仅2个）
    start_rot = R.from_euler('zyx', [0, 0, 0])
    end_rot = R.from_euler('zyx', [0, np.pi, 0])

    # 生成轨迹
    num_points = 100
    pos_trajectory = generate_smooth_position(waypoints_pos, num_points=num_points)
    quat_trajectory = generate_smooth_rotation(num_points, start_rot, end_rot)
    pose_trajectory = merge_position_rotation(pos_trajectory, quat_trajectory)

    # 生成每帧的工件局部坐标系（初始 x 设为全局 y 轴的负方向）
    # initial_x_dir = np.array([0.0, -1.0, 0.0])
    # final_x_dir = np.array([1.0, 0.0, 0.0])
    initial_x_dir = x_dir[0]
    final_x_dir = x_dir[1]
    frames = generate_object_frames_from_pose_trajectory(
        pose_trajectory, 
        initial_x_dir=initial_x_dir,
        final_x_dir=final_x_dir
    )

    # 可视化：轨迹 + 每帧局部坐标系
    # frame_step 控制绘制密度（若想全部画设为1）
    show = 0
    if show == True:
        plot_pose_trajectory_with_frames(waypoints_pos, start_rot, end_rot, pose_trajectory,
                                        frames=frames, frame_step=max(1, len(frames)//20),
                                        axis_length=0.04, arrow_length=0.05)
    # 返回轨迹与帧，方便后续保存或检查
    return pose_trajectory, frames

# -------------------------- 6. 主函数（按“位置生成→姿态生成→合并→生成小坐标系→可视化”流程调用） --------------------------
def main():
    """主流程：多位置关键点 + 仅2个姿态关键点 → 生成完整位姿轨迹，并生成每帧的工件局部坐标系进行可视化"""
    # 1. 输入（和你原来的一样）
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
    ]
    waypoints_pos = np.array(waypoints_pos)

    # 姿态关键点（仅2个）
    start_rot = R.from_euler('zyx', [0, 0, 0])
    end_rot = R.from_euler('zyx', [0, np.pi, 0])

    # 生成轨迹
    num_points = 200
    pos_trajectory = generate_smooth_position(waypoints_pos, num_points=num_points)
    quat_trajectory = generate_smooth_rotation(num_points, start_rot, end_rot)
    pose_trajectory = merge_position_rotation(pos_trajectory, quat_trajectory)

    # 生成每帧的工件局部坐标系（初始 x 设为全局 y 轴的负方向）
    initial_x_dir = np.array([0.0, -1.0, 0.0])
    final_x_dir = np.array([1.0, 0.0, 0.0])
    frames = generate_object_frames_from_pose_trajectory(
        pose_trajectory, 
        initial_x_dir=initial_x_dir,
        final_x_dir=final_x_dir
    )

    # 可视化：轨迹 + 每帧局部坐标系
    # frame_step 控制绘制密度（若想全部画设为1）
    show = 1
    if show == True:
        plot_pose_trajectory_with_frames(waypoints_pos, start_rot, end_rot, pose_trajectory,
                                        frames=frames, frame_step=max(1, len(frames)//20),
                                        axis_length=0.04, arrow_length=0.05)
    # 返回轨迹与帧，方便后续保存或检查
    return pose_trajectory, frames

if __name__ == "__main__":
    pose_trajectory, frames = main()
