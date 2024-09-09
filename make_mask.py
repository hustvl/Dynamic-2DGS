import cv2
import numpy as np
import torch


'''
imgpath = '/data3/zhangshuai/SC-2DGSv2/data/bouncingballs/train/r_003.png'
# 读取 PNG 图像文件
img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)

# 提取 Alpha 通道数据
alpha_channel = img[:, :, 3]

# 打印前 5 个 Alpha 值
print(np.max(alpha_channel))
cv2.imwrite("mask2.png", alpha_channel)
'''

imgpath = '/data3/zhangshuai/SC-2DGSv2/rendered_image2.png'
# 读取 PNG 图像文件
img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
print(img.shape)
img = torch.from_numpy(img)

mask = (1-(torch.all(img == 0, dim=2)).to(torch.int))
#mask = (1-(np.all(img == 0, axis=2))).astype(int)*255
mask = mask.numpy()*255

cv2.imwrite("mask3.png", mask)