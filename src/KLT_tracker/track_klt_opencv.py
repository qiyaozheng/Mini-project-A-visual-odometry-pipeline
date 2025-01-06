import cv2
import numpy as np

def trackKLTByOpencv(I_prev, I, keypoint, r_T, n_iter, threshold, use_bidirectional = False):
    """ 
    使用 OpenCV 实现的 KLT 跟踪，包含双向光流检测。
    这是对trackerKLTRobustly的opencv实现，同样的封装，输入输出
    追踪太容易false了，是双向检测的原因吗？试着增大金字塔层数3→5：单纯增大金字塔层数似乎不行
    似乎就是双向检测的原因，下面的trackKLTOpencvNobidirect效果不错
    Input:
        I_prev      np.ndarray 前一帧参考图像
        I           np.ndarray 当前帧图像
        keypoint    N x 2 np.ndarray 要跟踪的关键点坐标 [x, y]（列，行）
        r_T         scalar, 跟踪窗口大小（OpenCV 中通过 winSize 控制）
        n_iter      scalar, 最大迭代次数（OpenCV 中通过 criteria 控制）
        threshold   scalar, 双向误差阈值
    Output:
        delta       N x 2 np.ndarray，关键点的位移 [dx, dy]
        keep        boolean, 如果该关键点通过双向误差检测为 True，否则为 False
    """
    keypoint = np.array(keypoint, dtype=np.float32).reshape(-1, 1, 2)  # 转换为 OpenCV 的输入格式, N x 1 x 2

    # 单向（前向）跟踪：从 I_prev 跟踪到 I
    keypoints_next, status, _ = cv2.calcOpticalFlowPyrLK(
        I_prev, I, keypoint, None,
        winSize=(2 * r_T + 1, 2 * r_T + 1),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
    )

    if not status[0]:  # 如果光流跟踪失败，直接返回
        return np.zeros(2), False

    delta = (keypoints_next - keypoint).reshape(2)  # 计算位移 Δx, Δy

    # 逆向（反向）跟踪：从 I 跟踪回 I_prev
    keypoints_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        I, I_prev, keypoints_next, None,
        winSize=(2 * r_T + 1, 2 * r_T + 1),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
    )

    if not status_back[0]:  # 如果逆向光流跟踪失败，直接返回
        print(f"keypoint {keypoint} track fail")
        return np.zeros(2), False

    # 计算逆向位移
    dkpinv = (keypoints_back - keypoint).reshape(2)  # Δx_inv, Δy_inv

    # 双向误差检测
    keep = np.linalg.norm(delta + dkpinv) < threshold
    print(f"keypoint {keypoint} track, delta = {delta};  keep = {keep}")
    return delta, keep
  
def trackKLTOpencvNobidirect(I_prev, I, keypoint, r_T, n_iter, threshold, use_bidirectional = False):
  """ 
  使用 OpenCV 实现的 KLT 跟踪，使用 `status` 判断关键点是否有效。
  Input:
      I_prev             np.ndarray 前一帧参考图像
      I                  np.ndarray 当前帧图像
      keypoint           1 x 2 np.ndarray 要跟踪的单个关键点坐标 [x, y]（列，行）
      r_T                scalar 跟踪窗口大小（OpenCV 中通过 winSize 控制）
      n_iter             scalar 最大迭代次数（OpenCV 中通过 criteria 控制）
      use_bidirectional  boolean 是否使用逆向光流检测
  Output:
      delta              1 x 2 np.ndarray 关键点的位移 [dx, dy]
      keep               boolean 如果该关键点通过 `status` 检测为 True，否则为 False
  """
  # 转换关键点为 OpenCV 的输入格式
  keypoint = np.array(keypoint, dtype=np.float32).reshape(1, 1, 2)

  # 前向跟踪
  keypoints_next, status, _ = cv2.calcOpticalFlowPyrLK(
      I_prev, I, keypoint, None,
      winSize=(2 * r_T + 1, 2 * r_T + 1),  # 跟踪窗口大小
      maxLevel=3,  # 金字塔层数
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
  )

  # 检查前向跟踪状态
  if status[0][0] == 0:
      return np.zeros(2), False

  # 计算位移 Δx, Δy
  delta = (keypoints_next - keypoint).reshape(2)

  # 如果不使用逆向检测，直接返回
  if not use_bidirectional:
      return delta, True

  # 逆向跟踪
  keypoints_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
      I, I_prev, keypoints_next, None,
      winSize=(2 * r_T + 1, 2 * r_T + 1),
      maxLevel=3,
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
  )

  # 检查逆向跟踪状态
  if status_back[0][0] == 0:
      return np.zeros(2), False

  # 如果前向和逆向跟踪都成功，则认为关键点有效
  return delta, True
  
  
def trackKLTOpencvBatch(I_prev, I, keypoints, r_T, n_iter):
  """
  对多个关键点进行批量光流跟踪，并剔除不可靠点。
  solution中的trackKltRobustly是对单个关键点进行追踪，在main里面调用时需要做一个for循环，使用keep值剔除未被追踪到的点
  这里我们直接实现并封装对所有关键点的追踪和剔除，在main里面的keep和dkp不再需要
  因为需要可视化，所以我们需要将剔除后的keypoints和keypoints_new都输出
  使用了opencv就不再用手动双向检测了
  Input:
      I_prev         np.ndarray 前一帧参考图像
      I              np.ndarray 当前帧图像
      keypoints      2 x N np.ndarray 要跟踪的关键点坐标集合 [x, y]（列，行）
      r_T            scalar, 跟踪窗口大小（通过 winSize 控制）
      n_iter         scalar, 最大迭代次数（通过 criteria 控制）
  Output:
      keypoints_prev  2 x M np.ndarray，成功跟踪的上一帧关键点位置 [x, y]
      keypoints_new   2 x M np.ndarray，成功跟踪的关键点的新位置 [x', y']
  """
  # 转换关键点为 OpenCV 的输入格式
  keypoints = keypoints.T.astype(np.float32).reshape(-1, 1, 2)  # 转为 N x 1 x 2

  # 前向跟踪：从 I_prev 跟踪到 I
  keypoints_next, status, _ = cv2.calcOpticalFlowPyrLK(
      I_prev, I, keypoints, None,
      winSize=(2 * r_T + 1, 2 * r_T + 1),
      maxLevel=3,  # 使用金字塔
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
  )
  
  keypoints_next = keypoints_next.reshape(-1, 2)  # 转换回 N x 2
  keypoints_orig = keypoints.reshape(-1, 2)  # 转换回 N x 2
  
  # 筛选有效关键点：前向跟踪成功
  valid_mask = (status.flatten() == 1)
  keypoints_prev = keypoints_orig[valid_mask].T  # 形状 2 x M
  keypoints_new = keypoints_next[valid_mask].T  # 形状 2 x M


  return keypoints_prev, keypoints_new



def trackKLTOpencvBatch_return_mask(I_prev, I, keypoints, r_T, n_iter):
  """
  对多个关键点进行批量光流跟踪，并剔除不可靠点。
  solution中的trackKltRobustly是对单个关键点进行追踪，在main里面调用时需要做一个for循环，使用keep值剔除未被追踪到的点
  这里我们直接实现并封装对所有关键点的追踪和剔除，在main里面的keep和dkp不再需要
  因为需要可视化，所以我们需要将剔除后的keypoints和keypoints_new都输出
  
  在VO的pipeline里使用时，接受的是上一帧成功追踪到的landmarks和对应的keypoints
  由于我们需要在KLT时，对上一帧里面的landmarks和landmarks对应的keypoints进行同步筛选，所以我们需要输出valid_mask
  使用了opencv就不再用手动双向检测了
  Input:
    I_prev         np.ndarray 前一帧参考图像
    I              np.ndarray 当前帧图像
    keypoints      2 x N np.ndarray 要跟踪的关键点坐标集合 [x, y]（列，行）
    r_T            scalar, 跟踪窗口大小（通过 winSize 控制）
    n_iter         scalar, 最大迭代次数（通过 criteria 控制）
  Output:
    keypoints_prev  2 x M np.ndarray，成功跟踪的上一帧关键点位置 [x, y]
    keypoints_new   2 x M np.ndarray，成功跟踪的关键点的新位置 [x', y']
    valid_mask     M np.ndarray<boolean> 成功追踪为 True，否则为 False
  """
  # 转换关键点为 OpenCV 的输入格式
  if type(keypoints) == list:
    keypoints = np.array(keypoints)

  print(keypoints.shape)
  keypoints = keypoints.T.astype(np.float32).reshape(-1, 1, 2)  # 转为 N x 1 x 2

  # 前向跟踪：从 I_prev 跟踪到 I
  keypoints_next, status, _ = cv2.calcOpticalFlowPyrLK(
      I_prev, I, keypoints, None,
      winSize=(2 * r_T + 1, 2 * r_T + 1),
      maxLevel=5,  # 使用金字塔
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
  )
  
  keypoints_next = keypoints_next.reshape(-1, 2)  # 转换回 N x 2
  keypoints_orig = keypoints.reshape(-1, 2)  # 转换回 N x 2
  
  # 筛选有效关键点：前向跟踪成功
  valid_mask = (status.flatten() == 1)
  keypoints_prev = keypoints_orig[valid_mask].T  # 形状 2 x M
  keypoints_new = keypoints_next[valid_mask].T  # 形状 2 x M, 没有跟踪到的keypoints在keypoints_next中会有一个无意义的值，所以也需要过滤


  return keypoints_prev, keypoints_new, valid_mask


def trackKLTOpencvBatch_bidirection_return_mask(I_prev, I, keypoints, r_T, n_iter, thresh=1.0):
    """
    使用双向光流跟踪对多个关键点进行批量跟踪，并剔除不可靠点。

    Input:
        I_prev         np.ndarray 前一帧参考图像
        I              np.ndarray 当前帧图像
        keypoints      2 x N np.ndarray 要跟踪的关键点坐标集合 [x, y]（列，行）
        r_T            scalar, 跟踪窗口大小（通过 winSize 控制）
        n_iter         scalar, 最大迭代次数（通过 criteria 控制）
        thresh         scalar, 双向跟踪的误差阈值（像素单位）

    Output:
        keypoints_prev  2 x M np.ndarray，成功跟踪的上一帧关键点位置 [x, y]
        keypoints_new   2 x M np.ndarray，成功跟踪的关键点的新位置 [x', y']
        valid_mask      M np.ndarray<boolean> 成功追踪为 True，否则为 False
    """
    # 转换关键点为 OpenCV 的输入格式
    if type(keypoints) == list:
        keypoints = np.array(keypoints)
    keypoints = keypoints.T.astype(np.float32).reshape(-1, 1, 2)  # 转为 N x 1 x 2

    # 前向跟踪：从 I_prev 跟踪到 I
    keypoints_next, status, _ = cv2.calcOpticalFlowPyrLK(
        I_prev, I, keypoints, None,
        winSize=(2 * r_T + 1, 2 * r_T + 1),
        maxLevel=5,  # 使用金字塔
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
    )
    
    # 转换回 N x 2
    keypoints_next = keypoints_next.reshape(-1, 2)
    keypoints_orig = keypoints.reshape(-1, 2)

    # 反向跟踪：从 I 再跟踪回 I_prev
    keypoints_retracked, status_back, _ = cv2.calcOpticalFlowPyrLK(
        I, I_prev, keypoints_next.reshape(-1, 1, 2), None,
        winSize=(2 * r_T + 1, 2 * r_T + 1),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 0.03)
    )
    keypoints_retracked = keypoints_retracked.reshape(-1, 2)

    # 计算双向误差
    error = np.linalg.norm(keypoints_orig - keypoints_retracked, axis=1)

    # 筛选有效关键点：前向和反向跟踪均成功，且误差小于阈值
    valid_mask = (status.flatten() == 1) & (status_back.flatten() == 1) & (error < thresh)
    keypoints_prev = keypoints_orig[valid_mask].T  # 形状 2 x M
    keypoints_new = keypoints_next[valid_mask].T  # 形状 2 x M

    return keypoints_prev, keypoints_new, valid_mask