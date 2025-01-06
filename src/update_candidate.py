import numpy as np
from feature_detection import feature_detection, detect_features_with_grid


def update_candidates(curr_img, S, intrinsicMatrix, T_W_C, discard_radius=5):
    """
    Update candidates by filtering existing points and adding new ones.
    :param curr_img: Current frame image
    :param S: Current VO state (voState)
    :param intrinsicMatrix: Camera intrinsic matrix
    :param T_W_C: Current camera pose in world coordinates
    :param discard_radius: Radius for discarding nearby points
    :return: Updated candidates, first observations, and their poses
    """
    # candidates_keypoints, candidates_feature_descriptors = feature_detection(curr_img) # Returns: Tuple[List[cv2.KeyPoint], np.ndarray]:
    candidates_keypoints, candidates_feature_descriptors = detect_features_with_grid(curr_img,max_features=1000)
    # candidates_keypoints, candidates_feature_descriptors = feature_detection_surf(curr_img) # Returns: Tuple[List[cv2.KeyPoint], np.ndarray]:
    candidates_keypoints_coordinates = [candidates_keypoint.pt for candidates_keypoint in candidates_keypoints] # obtain keypoints coordinates (u, v) (col, row)
    candidates_keypoints_coordinates = np.array(candidates_keypoints_coordinates).T
    # print(f"candidates_keypoints_coordinates.shape = {candidates_keypoints_coordinates.shape}")
    # print(f"candidates_feature_descriptors.shape = {candidates_feature_descriptors.shape}")
    assert candidates_keypoints_coordinates.shape[-1]==candidates_feature_descriptors.T.shape[-1], f"number of candidates' coordinates:{candidates_keypoints_coordinates.shape[-1]} doesn't match number of corresponding descriptors:{candidates_feature_descriptors.shape[-1]}"
    
    def filter_candidates(coords, existing_coords, radius):
      # 计算每个候选点与每个已有点的距离
      # coords: (2, N), existing_coords: (2, M)
      diffs = existing_coords[:, :, None] - coords[:, None]
      distances = np.linalg.norm(diffs, axis=0)  # 计算欧几里得距离

      # 筛选候选点：距离所有已有点都大于 radius 的点
      mask = np.all(distances > radius, axis=0)
      return coords[:, mask]

    # Filter out candidates too close to existing keypoints
    if S.P.shape[1] > 0:
        candidates_keypoints_coordinates = filter_candidates(candidates_keypoints_coordinates, S.P, discard_radius)

    # Filter out candidates too close to existing candidates
    if S.C.shape[1] > 0:
        candidates_keypoints_coordinates = filter_candidates(candidates_keypoints_coordinates, S.C, discard_radius)
        
    #candidates_keypoints_coordinates = candidates_keypoints_coordinates.T # shape 2 x len(filtered_candidates)
    

    # update candidates
    if candidates_keypoints_coordinates.shape[1] > 0:
      # print(f'candidates_keypoints_coordinates.shape = {candidates_keypoints_coordinates.shape}')
      S.C = np.hstack([S.C, candidates_keypoints_coordinates])
      S.F = np.hstack([S.F, candidates_keypoints_coordinates])
      # print(f'S.T.shape = {S.T.shape}, T_W_C.shape = {T_W_C.shape}')
      T_W_C_repeat = np.repeat(T_W_C[:,:, np.newaxis], candidates_keypoints_coordinates.shape[1], axis=2)
      S.T = np.concatenate((S.T, T_W_C_repeat), axis=2)
      
    # print(f'number of candidates C:{S.C.shape[1]}, F:{S.F.shape[1]}, T:{S.T.shape[-1]}')
    
    return S