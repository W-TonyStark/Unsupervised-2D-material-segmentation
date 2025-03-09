import cv2
import numpy as np
from skimage import measure


def compute_boundary_alignment_score(original_image, processed_image, edge_threshold1=100, edge_threshold2=200):
    """
    计算边界对齐评分（BAS）。

    参数:
    - original_image: 原始图像，BGR格式。
    - processed_image: 处理后的分割图像，BGR或二值格式。
    - edge_threshold1, edge_threshold2: Canny边缘检测的阈值。

    返回:
    - BAS: 边界对齐评分。
    """
    # 转换为灰度图
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()

    # 提取边缘
    edges_orig = cv2.Canny(gray_orig, edge_threshold1, edge_threshold2)
    edges_proc = cv2.Canny(gray_proc, edge_threshold1, edge_threshold2)

    # 计算重叠边缘
    overlap = np.logical_and(edges_orig, edges_proc).sum()
    total_proc_edges = edges_proc.sum()

    BAS = overlap / (total_proc_edges + 1e-8)
    return BAS


def compute_impurity_exclusion_score(processed_image, dark_field_image, impurity_threshold=120):
    """
    计算杂质排除评分（IES）。

    参数:
    - processed_image: 处理后的分割图像，BGR或二值格式。
    - dark_field_image: 暗场图像，灰度格式。
    - impurity_threshold: 杂质区域的强度阈值。

    返回:
    - IES: 杂质排除评分。
    """
    # 转换为灰度图
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()

    # 提取纯白区域
    white_mask = (gray_proc >= 250).astype(np.uint8)  # 纯白区域阈值可调

    # 提取杂质区域
    impurity_mask = (dark_field_image > impurity_threshold).astype(np.uint8)

    # 计算重叠
    overlap = np.logical_and(white_mask, impurity_mask).sum()
    total_white = white_mask.sum()

    IES = overlap / (total_white + 1e-8)
    return IES


def compute_region_compactness(processed_image):
    """
    计算区域紧凑度（RC）。

    参数:
    - processed_image: 处理后的分割图像，BGR或二值格式。

    返回:
    - RC: 平均区域紧凑度。
    """
    # 转换为灰度图并提取纯白区域
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()
    white_mask = (gray_proc >= 250).astype(np.uint8)

    # 标记连通区域
    labeled_proc = measure.label(white_mask, connectivity=2)
    regions = measure.regionprops(labeled_proc)

    compactness_list = []
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        if perimeter == 0:
            continue
        C = (4 * np.pi * area) / (perimeter ** 2)
        compactness_list.append(C)

    RC = np.mean(compactness_list) if compactness_list else 0
    return RC


def compute_region_homogeneity(original_image, processed_image):
    """
    计算区域均匀性（RH）。

    参数:
    - original_image: 原始图像，BGR格式。
    - processed_image: 处理后的分割图像，BGR或二值格式。

    返回:
    - RH: 区域均匀性（标准差）。
    """
    # 转换为灰度图
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()

    # 提取纯白区域
    white_mask = (gray_proc >= 250).astype(np.uint8)

    # 计算纯白区域的像素强度标准差
    white_pixels = gray_orig[white_mask == 1]
    if white_pixels.size == 0:
        return 0
    RH = np.std(white_pixels)
    return RH


def compute_intensity_difference_score(original_image, processed_image):
    """
    计算强度差异评分（IDS）。

    参数:
    - original_image: 原始图像，BGR格式。
    - processed_image: 处理后的分割图像，BGR或二值格式。

    返回:
    - IDS: 强度差异评分。
    """
    # 转换为灰度图
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()

    # 提取纯白区域和背景区域
    white_mask = (gray_proc >= 250).astype(np.uint8)
    background_mask = (white_mask == 0).astype(np.uint8)

    # 计算平均强度
    I_white = gray_orig[white_mask == 1].mean() if np.any(white_mask) else 0
    I_background = gray_orig[background_mask == 1].mean() if np.any(background_mask) else 0

    IDS = I_background - I_white
    return IDS


def compute_region_overlap(processed_image, reference_mask, overlap_threshold=1):
    """
    计算处理图像与参考掩码的重叠程度（可选）。

    参数:
    - processed_image: 处理后的分割图像，BGR或二值格式。
    - reference_mask: 参考掩码，二值格式。
    - overlap_threshold: 重叠阈值。

    返回:
    - Overlap Score: 重叠评分。
    """
    # 转换为灰度图
    if len(processed_image.shape) == 3:
        gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_proc = processed_image.copy()

    # 提取纯白区域
    white_mask = (gray_proc >= 250).astype(np.uint8)

    # 计算重叠
    overlap = np.logical_and(white_mask, reference_mask).sum()
    total_white = white_mask.sum()

    Overlap_Score = overlap / (total_white + 1e-8)
    return Overlap_Score


def main_evaluation(original_image_path, processed_image_path, dark_field_image_path,
                    output_path='Evaluation_Results.txt'):
    """
    主评估函数，计算并保存所有评价指标。

    参数:
    - original_image_path: 原始图像路径。
    - processed_image_path: 处理后的图像路径（4号图）。
    - dark_field_image_path: 暗场图像路径（3号图）。
    - output_path: 结果保存路径。
    """
    # 读取图像
    original_image = cv2.imread(original_image_path)
    processed_image = cv2.imread(processed_image_path)
    dark_field_image = cv2.imread(dark_field_image_path, cv2.IMREAD_GRAYSCALE)
    # 由于暗场分别率与明场有轻微不一致，直接resize调整一下
    dark_field_image = cv2.resize(dark_field_image, (original_image.shape[1], original_image.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    if original_image is None or processed_image is None or dark_field_image is None:
        print("Error: 图像读取失败，请检查路径是否正确。")
        return

    # 计算各项指标
    BAS = compute_boundary_alignment_score(original_image, processed_image)
    IES = compute_impurity_exclusion_score(processed_image, dark_field_image)
    RC = compute_region_compactness(processed_image)
    RH = compute_region_homogeneity(original_image, processed_image)
    IDS = compute_intensity_difference_score(original_image, processed_image)

    # # 保存结果
    # with open(output_path, 'w') as f:
    #     f.write(f"Boundary Alignment Score (BAS): {BAS:.4f}\n")
    #     f.write(f"Impurity Exclusion Score (IES): {IES:.4f}\n")
    #     f.write(f"Region Compactness (RC): {RC:.4f}\n")
    #     f.write(f"Region Homogeneity (RH): {RH:.4f}\n")
    #     f.write(f"Intensity Difference Score (IDS): {IDS:.4f}\n")

    # 打印结果
    print(f"Boundary Alignment Score (BAS): {BAS:.4f}")
    print(f"Impurity Exclusion Score (IES): {IES:.4f}")
    print(f"Region Compactness (RC): {RC:.4f}")
    print(f"Region Homogeneity (RH): {RH:.4f}")
    print(f"Intensity Difference Score (IDS): {IDS:.4f}")
    print(f"评价结果已保存到 {output_path}")


if __name__ == "__main__":
    # 示例用法
    original_image_path = r'D:\pythonProject\Unsupervised-Segmentation-master\compare\merge_sample3_0104_C.jpeg'  # 1号图
    processed_image_path = r'D:\pythonProject\Unsupervised-Segmentation-master\compare\SCM_result.jpeg'  # 4号图
    dark_field_image_path = r'D:\pythonProject\Unsupervised-Segmentation-master\compare\merge_sample3_0107_C.jpeg'  # 3号图

    main_evaluation(original_image_path, processed_image_path, dark_field_image_path)
