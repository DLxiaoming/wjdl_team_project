#!/bin/bash
# 真空管抓取配置脚本
# 使用方法：
# 1. 先运行 python z_pre_grasp.py 获取四元数
# 2. 修改下面的四元数值
# 3. 运行 ./grasp_tube_config.sh

# ===== 示教的抓取姿态（四元数）=====
# 从 z_pre_grasp.py 获取，替换下面的值
GRASP_QX=0.0
GRASP_QY=0.707
GRASP_QZ=0.0
GRASP_QW=0.707

# ===== 抓取参数调整 =====
OFFSET_Z=0.0        # 基座坐标系Z轴偏移（米），正数向上，负数向下
DESCENT=0.05        # 下降距离（米）
LIFT=0.10           # 抬起距离（米）

# ===== 末端坐标系偏移（参考operator_process_1029.py）=====
ARM_OFFSET_X=0.0    # 末端坐标系X轴偏移（米）
ARM_OFFSET_Y=0.0    # 末端坐标系Y轴偏移（米）
ARM_OFFSET_Z=0.0    # 末端坐标系Z轴偏移（米）

# ===== 相机参数 =====
HEAD_PITCH=-0.5236  # 头部俯仰角（弧度），-30度

# ===== 机器人配置 =====
ROBOT_IP="192.168.11.200"

# ===== 执行脚本 =====
echo "========================================"
echo "真空管抓取系统"
echo "========================================"
echo "抓取姿态: [$GRASP_QX, $GRASP_QY, $GRASP_QZ, $GRASP_QW]"
echo "基座Z偏移: ${OFFSET_Z}m"
echo "末端偏移: x=${ARM_OFFSET_X}m, y=${ARM_OFFSET_Y}m, z=${ARM_OFFSET_Z}m"
echo "下降距离: ${DESCENT}m"
echo "抬起距离: ${LIFT}m"
echo "========================================"
echo ""

python z_fdp_grasp.py \
    --robot_ip "$ROBOT_IP" \
    --head_pitch "$HEAD_PITCH" \
    --grasp_qx "$GRASP_QX" \
    --grasp_qy "$GRASP_QY" \
    --grasp_qz "$GRASP_QZ" \
    --grasp_qw "$GRASP_QW" \
    --grasp_offset_z "$OFFSET_Z" \
    --descent "$DESCENT" \
    --lift "$LIFT" \
    --arm_offset_x "$ARM_OFFSET_X" \
    --arm_offset_y "$ARM_OFFSET_Y" \
    --arm_offset_z "$ARM_OFFSET_Z"

