from pycocotools import mask as maskUtils
from PIL import Image
import numpy as np
import json

# 读取 COCO 标注文件
with open('/home/kyono/下载/dataset_20251013_194415_coco/dataset/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# 遍历每个标注
for i, annotation in enumerate(coco_data['annotations']):
    # 获取图像信息
    image_id = annotation['image_id']
    image_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            image_info = img
            break
    if not image_info:
        continue
    height = image_info['height']
    width = image_info['width']
    
    # 从 RLE 编码获取掩码数组
    rle = annotation['segmentation']
    mask = maskUtils.decode(rle)
    
    # 将掩码数组转换为 PIL 图像（黑白）
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    
    # 保存 mask 图片，文件名可根据需求自定义，这里以 image_id 和 annotation_id 命名
    mask_img.save(f"demo_data/{i}.png")
    # if i==0:
    #     break