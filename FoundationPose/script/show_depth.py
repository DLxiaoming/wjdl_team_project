# 假设使用 OpenCV 读取深度图
import cv2

# 读取深度图
file = 'demo_data/mustard0/depth/1581120424100262102.png'
file = 'demo_data/mc_piston/depth/1760069349211632640.png'
depth_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# 输出相关信息
print(depth_image[200])
print(depth_image.max(), depth_image.min())
print("格式:", "灰度图 (Grayscale)")
print("Size:", depth_image.shape)
print("维度:", "2D")
print("Type:", depth_image.dtype)
print("数据类型:", "uint8" if depth_image.dtype == 'uint8' else "其他")
print("位数:", 8 if depth_image.dtype == 'uint8' else 16 if depth_image.dtype == 'uint16' else 32)