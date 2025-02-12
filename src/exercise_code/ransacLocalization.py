import cv2
import numpy as np

from exercise_code.estimate_pose_dlt import estimatePoseDLT
from exercise_code.projectPoints import projectPoints


def ransacLocalization(matched_query_keypoints, corresponding_landmarks, K):
    """
    input:
      corresponding_landmarks: shape = (N, 3)
      matched_query_keypoints: shape = (N, 2)
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
    pass
    use_p3p = True
    tweaked_for_more = True
    adaptive = True  # whether or not to use ransac adaptively

    if use_p3p:
        num_iterations = 1000 if tweaked_for_more else 200
        pixel_tolerance = 10
        k = 3
    else:
        num_iterations = 2000
        pixel_tolerance = 10
        k = 6

    if adaptive:
        num_iterations = float("inf")

    # Initialize RANSAC
    best_inlier_mask = np.zeros(matched_query_keypoints.shape[1])

    # (row, col) to (u, v)
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:

        # Model from k samples (DLT or P3P)
        indices = np.random.permutation(corresponding_landmarks.shape[0])[:k]
        landmark_sample = corresponding_landmarks[indices, :]
        keypoint_sample = matched_query_keypoints[:, indices]

        if use_p3p:
            success, rotation_vectors, translation_vectors = cv2.solveP3P(
                landmark_sample,
                keypoint_sample.T,
                K,
                None,
                flags=cv2.SOLVEPNP_P3P,
            )
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)

        else:
            M_C_W_guess = estimatePoseDLT(
                keypoint_sample.T, landmark_sample, K
            )
            R_C_W_guess = M_C_W_guess[:, :3]
            t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        if not use_p3p:
            C_landmarks = (
                np.matmul(
                    R_C_W_guess, corresponding_landmarks[:, :, None]
                ).squeeze(-1)
                + t_C_W_guess[None, :]
            )
            projected_points = projectPoints(C_landmarks, K)
            difference = matched_query_keypoints - projected_points.T
            errors = (difference**2).sum(0)
            is_inlier = errors < pixel_tolerance**2

        else:
            # If we use p3p, also consider inliers for the 4 solutions.
            is_inlier = np.zeros(corresponding_landmarks.shape[0])
            for alt_idx in range(len(R_C_W_guess)):

                # Project points
                C_landmarks = np.matmul(
                    R_C_W_guess[alt_idx], corresponding_landmarks[:, :, None]
                ).squeeze(-1) + t_C_W_guess[alt_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_landmarks, K)

                difference = matched_query_keypoints - projected_points.T
                errors = (difference**2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance**2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier

        min_inlier_count = 30 if tweaked_for_more else 6

        if (
            is_inlier.sum() > max_num_inliers
            and is_inlier.sum() >= min_inlier_count
        ):
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        if adaptive:
            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]

            # formula to compute number of iterations from estimated outlier ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log(1 - confidence) / np.log(
                1 - (1 - outlier_ratio) ** k
            )

            # cap the number of iterations at 15000
            num_iterations = min(15000, num_iterations)

        num_iteration_history.append(num_iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1
    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W = estimatePoseDLT(
            matched_query_keypoints[:, best_inlier_mask].T,
            corresponding_landmarks[best_inlier_mask, :],
            K,
        )
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

        if adaptive:
            print(
                "    Adaptive RANSAC: Needed {} iteration to converge.".format(
                    i - 1
                )
            )
            print(
                "    Adaptive RANSAC: Estimated Ouliers: {} %".format(
                    100 * outlier_ratio
                )
            )

    return (
        R_C_W,
        t_C_W,
        best_inlier_mask,
        max_num_inliers_history,
        num_iteration_history,
    )

def ransacLocalizationOpenCV(matched_query_keypoints, corresponding_landmarks, K):
    """
    使用 RANSAC 计算相机位姿 (R_C_W, t_C_W)

    参数：
    - matched_query_keypoints: 2D 图像点 (N, 2)
    - corresponding_landmarks: 对应的 3D 地标点 (N, 3)
    - K: 相机内参矩阵 (3, 3)

    返回：
    - R_C_W: 旋转矩阵 (相机坐标系到世界坐标系的旋转)
    - t_C_W: 平移向量
    - bool_inlier_mask: 布尔掩码
    - max_num_inliers_history: 最大内点数量历史记录 (这里只是存储当前的内点数量)
    - num_iteration_history: 迭代次数历史记录 (这里是设置的最大迭代次数)
    """

    print(f"opencv ransac num(matched_query_keypoints):{matched_query_keypoints.shape[0]}, num(corresponding_landmarks):{corresponding_landmarks.shape[0]}")
    assert np.all(np.isfinite(corresponding_landmarks)), "3D points contain invalid values!"
    assert np.all(np.isfinite(matched_query_keypoints)), "2D points contain invalid values!"

    # 初始化历史记录
    max_num_inliers_history = []
    num_iteration_history = []

    # 设置 RANSAC 参数
    reprojection_error_threshold = 5.0  # 重投影误差阈值（像素单位）
    confidence = 0.99  # 置信度
    max_iterations = 1500  # 最大迭代次数

    # 使用 OpenCV 的 RANSAC PnP 方法
    retval, rvec, tvec, inlier_mask = cv2.solvePnPRansac(
        objectPoints=corresponding_landmarks,  # 3D 点
        imagePoints=matched_query_keypoints,  # 2D 图像点
        cameraMatrix=K,  # 相机内参矩阵
        distCoeffs=None,  # 假设无畸变
        iterationsCount=max_iterations,  # 最大迭代次数
        reprojectionError=reprojection_error_threshold,  # 重投影误差阈值
        confidence=confidence,  # RANSAC 的置信度
        flags=cv2.SOLVEPNP_ITERATIVE  # 使用迭代 PnP 方法
    )
        
    # print(f"inlier_mask shape: {inlier_mask.shape}")
    # inlier_mask = inlier_mask.flatten()
    
    # 检查是否成功
    if not retval:
        raise ValueError("RANSAC 位姿估计失败")
    
    bool_inlier_mask = np.zeros(matched_query_keypoints.shape[0], dtype=bool)

    # 将 inlier_mask 对应的索引设置为 True
    bool_inlier_mask[inlier_mask.ravel()] = True

    # 将旋转向量转换为旋转矩阵
    R_C_W, _ = cv2.Rodrigues(rvec)

    # 平移向量
    t_C_W = tvec

    # 记录最大内点数量历史（这里只记录最后的内点数量）
    max_num_inliers_history.append(np.sum(inlier_mask))  # 内点数量
    num_iteration_history.append(max_iterations)  # 迭代次数

    # 返回结果
    return R_C_W, t_C_W, bool_inlier_mask, max_num_inliers_history, num_iteration_history

def refine_pose_with_nonlinear_optimization(S, R_C_W, t_C_W, intrinsicMatrix):
    """
    使用非线性优化对相机位姿进行进一步优化。

    Args:
        S: voState, 包含跟踪的关键点和三角化的3D点。
        R_C_W: 初始旋转矩阵 (3, 3)。
        t_C_W: 初始平移向量 (3, )。
        intrinsicMatrix: 相机内参矩阵 (3, 3)。

    Returns:
        R_C_W_refined: 优化后的旋转矩阵 (3, 3)。
        t_C_W_refined: 优化后的平移向量 (3, )。
    """
    # 转换 3D 点和 2D 点为 OpenCV 格式
    object_points = S.X.T  # 3D 点，形状 (N, 3)
    image_points = S.P.T  # 2D 点，形状 (N, 2)

    # 将旋转矩阵转换为罗德里格斯向量
    rvec, _ = cv2.Rodrigues(R_C_W)

    # 使用 Levenberg-Marquardt 方法进行优化
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=intrinsicMatrix,
        distCoeffs=None,
        rvec=rvec,
        tvec=t_C_W
    )

    # 将优化后的罗德里格斯向量转换回旋转矩阵
    R_C_W_refined, _ = cv2.Rodrigues(rvec_refined)

    # 返回优化后的旋转和平移矩阵
    return R_C_W_refined, tvec_refined