import cv2
import numpy as np

# 读取原始图像
origin_img = cv2.imread('dataset/merge_sample5_0115_C.jpeg')

# 读取左边和右边的图像
left_img = cv2.imread('dataset/evaluate_output_image.jpeg')
right_img = cv2.imread('dataset/merge_sample5_0118_C_pre_pro.png')
right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))

# 将左边图像转换为灰度图并进行阈值分割
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
_, left_mask = cv2.threshold(left_gray, 100, 255, cv2.THRESH_BINARY)

# 对右边图像进行处理
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
_, right_binary = cv2.threshold(right_gray, 127, 255, cv2.THRESH_BINARY)

# 获取右边图像中杂质（白色像素）的坐标位置
impurity_y, impurity_x = np.where(right_binary > 120)

# 获取左边图像中杂质（黑色像素）的坐标位置
left_y, left_x = np.where(left_mask < 10)

# 记录需要填充黑色的坐标位置
fill_points = []
for y, x in zip(impurity_y, impurity_x):
    if left_mask[y, x] < 130:
        fill_points.append((y, x))

# 记录需要填充白色的坐标位置
useful_points = []
for y, x in zip(left_y, left_x):
    useful_points.append((y, x))

# 将fill_points转换为集合，方便查找和比较
fill_points_set = set(fill_points)

# 过滤掉在fill_points中出现的坐标
filtered_useful_points = [point for point in useful_points if point not in fill_points_set]
useful_points = list(filtered_useful_points)

# 将对应坐标位置的像素在左边图像中填充为白色
for y, x in useful_points:
    origin_img[y, x] = [255, 255, 255]

# 显示结果
cv2.imwrite('dataset/Result_test.jpeg', origin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()