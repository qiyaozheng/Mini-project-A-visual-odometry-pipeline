import numpy as np
import cv2

def feature_detection(img: np.array):
  '''
  given a image, return its key point and corresponding descriptor
  Args:
    img (np.array) : 
  Returns:
    Tuple[List[cv2.KeyPoint], np.ndarray]:

  '''
  
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  img = clahe.apply(img)
  
  sift = cv2.SIFT.create(
        nfeatures=500,              # 限制最多检测 500 个特征点
        contrastThreshold=0.03,     # 降低对比度阈值以检测更多特征点
        edgeThreshold=15,           # 增大边缘阈值，避免丢失边缘特征
        sigma=1.2                   # 调整高斯模糊的标准差
    )
  kp, descriptor = sift.detectAndCompute(img, None)
  return kp, descriptor


def detect_features_with_grid(img, grid_size=(4, 4), max_features=500):
    """
    extract keypoints from even divided subgrid of img
    """
    sift = cv2.SIFT_create()
    h, w = img.shape
    step_h, step_w = h // grid_size[0], w // grid_size[1]

    keypoints, descriptors = [], []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 提取子区域
            roi = img[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            kp, des = sift.detectAndCompute(roi, None)
            if kp:
                # 调整 keypoint 的坐标到全图
                for k in kp:
                    k.pt = (k.pt[0] + j * step_w, k.pt[1] + i * step_h)
                keypoints.extend(kp[:max_features])  # 限制每个网格的特征点数量
                if des is not None:
                    descriptors.append(des[:max_features])

    if descriptors:
        descriptors = np.vstack(descriptors)
    return keypoints, descriptors
  
def keypoint_feature_detection(img:np.array, keypoints):
  pass
  sift = cv2.SIFT.create()
  kp, descriptor = sift.compute(img, keypoints=keypoints)
  return kp, descriptor