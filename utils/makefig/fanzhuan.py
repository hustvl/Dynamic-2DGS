
import cv2
import numpy as np

img_path = "/data3/zhangshuai/SC-2DGSv2/outputs/jumpingjacks_0731_node/test/ours_60000/vis/depth_00002.png"


# Load the depth image
image = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Convert black background to white
h, w, c = image.shape
white_background = 255 * np.ones_like(image, dtype=np.uint8)
mask = cv2.inRange(image, (0, 0, 0), (10, 10, 10))
white_background[mask != 255] = image[mask != 255]

cv2.imwrite('white_background_depth_image.png', white_background)