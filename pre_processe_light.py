import cv2
import numpy as np

# 读取图像
image_path = 'dataset/merge_sample5_0115_C.jpeg'
image = cv2.imread(image_path)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用大津阈值法，生成二值图像
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('11', binary)
cv2.waitKey(0)

# 对二值图像进行形态学操作，去除噪声
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# 计算背景和前景
sure_bg = cv2.dilate(binary, kernel, iterations=3)
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通域
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)
# image[markers == -1] = [0, 0, 255]  # 标记边界

# 将背景统一为一种颜色
background_mask = (markers == 1)
image[background_mask] = [255, 255, 255]

# 保存并展示结果
output_path = 'dataset/merge_sample5_0115_C_pre_pro_test.jpeg'
cv2.imwrite(output_path, image)

cv2.waitKey(0)
cv2.destroyAllWindows()