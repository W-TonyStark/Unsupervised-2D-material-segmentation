import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

def align_images(bright_field_path, dark_field_path, output_bright_path, output_dark_path):
    # Step 1: Load the bright and dark field images
    bright_field = cv2.imread(bright_field_path)
    dark_field = cv2.imread(dark_field_path)

    # Ensure images are the same size
    height, width, _ = bright_field.shape
    dark_field = cv2.resize(dark_field, (width, height))

    # Step 2: Convert to grayscale for precise alignment
    bright_gray = cv2.cvtColor(bright_field, cv2.COLOR_BGR2GRAY)
    dark_gray = cv2.cvtColor(dark_field, cv2.COLOR_BGR2GRAY)

    # Step 3: Calculate translation shift using phase correlation
    shift, error, _ = phase_cross_correlation(bright_gray, dark_gray, upsample_factor=10)
    shift_y, shift_x = shift
    # shift_x -= 8
    # shift_y += 3
    print(f"Detected shifts -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")

    # Step 4: Apply translation to align dark field image
    aligned_dark_field = cv2.warpAffine(
        dark_field,
        np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR
    )

    # Step 5: Save aligned images
    # cv2.imwrite(output_bright_path, bright_field)
    # cv2.imwrite(output_dark_path, aligned_dark_field)

    return shift_x, shift_y

# 输入的明场与暗场图像路径（请自行修改为你的图像路径）
bright_field_path = './dataset/merge_sample5_0115_C.jpeg'
dark_field_path = './dataset/merge_sample5_0118_C.jpeg'
# 输出图像的路径
output_bright_path = './dataset/merge_sample5_0115_C.png'
output_dark_path = './dataset/merge_sample5_0118_C.png'

# Run alignment
shift_x, shift_y = align_images(bright_field_path, dark_field_path, output_bright_path, output_dark_path)
print(f"Final shift values -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")

# import cv2
# import numpy as np
# from skimage.registration import phase_cross_correlation
#
# def align_images_optimized(bright_field_path, dark_field_path, output_bright_path, output_dark_path):
#     # Step 1: Load the bright and dark field images
#     bright_field = cv2.imread(bright_field_path)
#     dark_field = cv2.imread(dark_field_path)
#
#     # Ensure images are the same size
#     height, width, _ = bright_field.shape
#     dark_field = cv2.resize(dark_field, (width, height))
#
#     # Step 2: Convert to grayscale for precise alignment
#     bright_gray = cv2.cvtColor(bright_field, cv2.COLOR_BGR2GRAY)
#     dark_gray = cv2.cvtColor(dark_field, cv2.COLOR_BGR2GRAY)
#
#     # Step 3: Apply pyramid levels for coarse-to-fine alignment
#     num_levels = 3
#     bright_pyramid = [bright_gray]
#     dark_pyramid = [dark_gray]
#     for _ in range(num_levels - 1):
#         bright_pyramid.append(cv2.pyrDown(bright_pyramid[-1]))
#         dark_pyramid.append(cv2.pyrDown(dark_pyramid[-1]))
#
#     shift_x, shift_y = 0, 0
#     for level in reversed(range(num_levels)):
#         # Calculate phase correlation at the current level
#         shift, error, _ = phase_cross_correlation(bright_pyramid[level], dark_pyramid[level], upsample_factor=10)
#         shift_y += shift[0] * (2**level)
#         shift_x += shift[1] * (2**level)
#
#     print(f"Optimized shifts -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")
#
#     # Step 4: Apply translation to align dark field image
#     aligned_dark_field = cv2.warpAffine(
#         dark_field,
#         np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32),
#         (width, height),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0)
#     )
#
#     # Step 5: Save aligned images
#     # cv2.imwrite(output_bright_path, bright_field)
#     cv2.imwrite(output_dark_path, aligned_dark_field)
#
#     return shift_x, shift_y
#
# # 输入的明场与暗场图像路径（请自行修改为你的图像路径）
# bright_field_path = './dataset/merge_sample5_0115_C.jpeg'
# dark_field_path = './dataset/merge_sample5_0118_C.jpeg'
# # 输出图像的路径
# output_bright_path = './dataset/merge_sample5_0115_C.png'
# output_dark_path = './dataset/merge_sample5_0118_C.png'
#
# # Run alignment
# shift_x, shift_y = align_images_optimized(bright_field_path, dark_field_path, output_bright_path, output_dark_path)
# print(f"Final optimized shift values -> x: {shift_x:.2f} pixels, y: {shift_y:.2f} pixels")
