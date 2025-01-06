import numpy as np
import cv2
from pathlib import Path

def rt_to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵 R 和平移向量 T 转换为 4x4 的齐次变换矩阵。
    
    Args:
        R (np.ndarray): 3x3 的旋转矩阵。
        t (np.ndarray): 3x1 或 1x3 的平移向量。
    
    Returns:
        np.ndarray: 4x4 的齐次变换矩阵。
    """
    # 确保 R 是 3x3，T 是 3x1
    assert R.shape == (3, 3), "R 必须是 3x3 的旋转矩阵"
    assert t.shape in [(3,), (3, 1), (1, 3)], "T 必须是 3x1 的平移向量"

    # 转换 T 为列向量 (3x1)
    t = t.reshape(3, 1)

    # 构造齐次变换矩阵
    homogeneous_matrix = np.eye(4)  # 初始化 4x4 单位矩阵
    homogeneous_matrix[:3, :3] = R  # 设置旋转部分
    homogeneous_matrix[:3, 3] = t.flatten()  # 设置平移部分

    return homogeneous_matrix
  
def posevec_to_homogeneous(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    将旋转向量和平移向量转换为 4x4 的齐次变换矩阵。
    
    Args:
        rvec (np.ndarray): 旋转向量 (3x1)。
        tvec (np.ndarray): 平移向量 (3x1)。
    
    Returns:
        np.ndarray: 4x4 的齐次变换矩阵。
    """
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    # 使用之前定义的函数转换为齐次矩阵
    return rt_to_homogeneous(R, tvec)

def preprocess_image(image, resize_dim=(256, 256), apply_blur=False, equalize_hist=True, normalize=False):
    """
    图像预处理函数：对输入图像进行一系列的预处理操作。
    
    Args:
        image (numpy.ndarray): 输入图像，可以是彩色图像或灰度图像（BGR 格式）。
        resize_dim (tuple): 调整图像尺寸（宽，高），默认为 (256, 256)。
        apply_blur (bool): 是否应用高斯滤波去噪，默认开启。
        equalize_hist (bool): 是否应用直方图均衡化（仅对灰度图像有效），默认开启。
        normalize (bool): 是否将像素值归一化到 [0, 1] 范围，默认开启。
    
    Returns:
        numpy.ndarray: 预处理后的图像。
    """
    # 如果输入是彩色图像，将其转换为灰度图像
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if image is RGB/BGR
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 调整图像尺寸
    # resized_image = cv2.resize(gray_image, resize_dim)
    resized_image = gray_image

    # 应用高斯滤波去噪
    if apply_blur:
        blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
    else:
        blurred_image = resized_image

    # 应用直方图均衡化，增强对比度
    if equalize_hist:
        enhanced_image = cv2.equalizeHist(blurred_image)
    else:
        enhanced_image = blurred_image

    # 将像素值归一化到 [0, 1]
    if normalize:
        normalized_image = enhanced_image.astype(np.float32) / 255.0
    else:
        normalized_image = enhanced_image

    return normalized_image

if __name__ == "__main__":
    # 读取图像
    current_dir = Path(__file__).resolve().parent.parent
    dataset_path = current_dir / "kitti"
    img_path = dataset_path / "05/image_0" / "000000.png"
    img = cv2.imread(str(img_path), 0) 

    # 调用预处理函数
    preprocessed_image = preprocess_image(img)

    # 显示原图和预处理后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Preprocessed Image', preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()