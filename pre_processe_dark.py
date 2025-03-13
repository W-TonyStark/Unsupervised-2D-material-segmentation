import cv2
import numpy as np

# 读取图像
image = cv2.imread('dataset/merge_sample5_0118_C.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === 高亮杂质过滤 ===
_, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
gray[bright_mask == 255] = 0

# cv2.imshow('Improved Watershed with Edge Repair', bright_mask)
# cv2.waitKey(0)

# 获取左边图像中杂质（黑色像素）的坐标位置
highlight_y, highlight_x = np.where(bright_mask == 255)

# 记录需要填充黑色的坐标位置
fill_points = []
for y, x in zip(highlight_y, highlight_x):
    fill_points.append((y, x))

# === 对比度增强 ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# === 边缘检测并修补 ===
edges = cv2.Canny(gray_clahe, 50, 150)
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=0)  # 膨胀边缘

# === 确定前景和背景 ===
# 二值化
_, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

for y, x in fill_points:
    thresh[y, x] = [255, 255, 255]

cv2.imwrite('dataset/merge_sample5_0118_C_pre_pro.png', thresh)
# cv2.imshow('Improved Watershed with Edge Repair', thresh)
# cv2.waitKey(0)

# # 确定背景
# sure_bg = cv2.dilate(thresh, kernel, iterations=3)
#
# # 确定前景（基于距离变换）
# dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist_transform, 0.9 * dist_transform.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
#
# # 修补边缘标记
# combined = cv2.bitwise_or(sure_fg, edges_dilated)
#
# cv2.imshow('Improved Watershed with Edge Repair', combined)
# cv2.waitKey(0)
#
# # 未知区域
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# # === 标记并应用分水岭 ===
# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
#
# # 将边缘标记添加到 markers
# markers[combined > 0] = 2  # 边缘区域标记
#
# # 分水岭
# cv2.watershed(image, markers)
# image[markers == -1] = [0, 0, 255]  # 边界标红
#
# # 显示结果
# cv2.imshow('Improved Watershed with Edge Repair', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# import cv2
# import numpy as np
#
# # 读取图像
# image = cv2.imread('test_dark.jpeg')
# image_copy = image.copy()
#
# # 定义种子点（选择合适的区域内部点）
# seed_point = (50, 50)  # 根据实际情况调整种子点的坐标
#
# # 定义填充颜色和容差
# fill_color = (0, 255, 0)  # 填充绿色
# lo_diff = (20, 20, 20)  # 低容差
# up_diff = (20, 20, 20)  # 高容差
#
# # 洪水填充函数
# mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)  # 掩膜需要比图像大2
# cv2.floodFill(image_copy, mask, seedPoint=seed_point, newVal=fill_color, loDiff=lo_diff, upDiff=up_diff)
#
# # 显示结果
# cv2.imshow('Flood Fill Result', image_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()