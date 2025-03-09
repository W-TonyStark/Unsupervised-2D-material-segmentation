import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

def align_images(bright_field_path, dark_field_path, output_bright_path, output_dark_path):
    bright_field = cv2.imread(bright_field_path)
    dark_field = cv2.imread(dark_field_path)

    height, width, _ = bright_field.shape
    dark_field = cv2.resize(dark_field, (width, height))

    bright_gray = cv2.cvtColor(bright_field, cv2.COLOR_BGR2GRAY)
    dark_gray = cv2.cvtColor(dark_field, cv2.COLOR_BGR2GRAY)

    shift, error, _ = phase_cross_correlation(bright_gray, dark_gray, upsample_factor=10)
    shift_y, shift_x = shift
    print(f"Detected shifts -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")

    aligned_dark_field = cv2.warpAffine(
        dark_field,
        np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR
    )

    return shift_x, shift_y

# 输入的明场与暗场图像路径
bright_field_path = './dataset/merge_sample5_0115_C.jpeg'
dark_field_path = './dataset/merge_sample5_0118_C.jpeg'
# 输出图像的路径
output_bright_path = './dataset/merge_sample5_0115_C.png'
output_dark_path = './dataset/merge_sample5_0118_C.png'

shift_x, shift_y = align_images(bright_field_path, dark_field_path, output_bright_path, output_dark_path)
print(f"Final shift values -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")
