import numpy as np
import cv2

from KLT_tracker.track_klt_opencv import trackKLTOpencvBatch_return_mask, trackKLTOpencvBatch_bidirection_return_mask
from exercise_code.ransacLocalization import ransacLocalization, ransacLocalizationOpenCV, refine_pose_with_nonlinear_optimization
from update_candidate import update_candidates

from util import rt_to_homogeneous

from update_landmarks import update_landmarks

from state import voState

from feature_detection import feature_detection


def process_frame(prev_img, curr_img, S:voState, intrinsicMatrix):
  
  # part 1: keypoints tracking & pose estimation
  # perform KLT tracking
  # keypoints_coordinates_prev, keypoints_coordinates_new, valid_mask = trackKLTOpencvBatch_return_mask(I_prev=prev_img, I=curr_img, keypoints=S.P, r_T=15, n_iter=100)
  keypoints_coordinates_prev, keypoints_coordinates_new, valid_mask = trackKLTOpencvBatch_bidirection_return_mask(I_prev=prev_img, I=curr_img, keypoints=S.P, r_T=15, n_iter=100)
  
  S.P_prev = keypoints_coordinates_prev
  S.P_curr = keypoints_coordinates_new
  
  # first filtering to keypoints, discard untracked landmarks & keypoints
  # print(f"shape of landmarks = {S.X.shape}")
  # print(f"first keypoints filter, klt tracking")
  # print(f"before filter: {S.P.shape[1]}, after filter: {keypoints_coordinates_new.shape[1]}")
  S.P = keypoints_coordinates_new
  S.X = S.X[:, valid_mask]
  
  # 加多一个，剔除在相机后方的landmark
  if S.poses.shape[-1]!=0:
    S.X, valid_mask_2 = filter_landmarks_behind_camera(pose=S.poses[:,:,-1], landmarks=S.X)
    # print(f"filter keypoints behinde camera")
    # print(f"before filter: {S.P.shape[1]}, after filter: {sum(valid_mask_2)}")
    S.P = S.P[:, valid_mask_2]
  
  # performan ransac pnp to find the pose
  assert S.P.shape[1] == S.X.shape[1], f'len of keypoints: {S.P.shape[1]} != len of landmarks{S.X.shape[1]}'
  #assert S.P.dtype == S.X.dtype, f' dtype of keypoints: {S.P.dtype} != dtype of  landmarks: {S.X.dtype}'
  print(f'len of keypoints: {S.P.shape[1]} and len of landmarks{S.X.shape[1]}\n')
  # R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = ransacLocalization(matched_query_keypoints=S.P, corresponding_landmarks=S.X.T, K = intrinsicMatrix)
  R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = ransacLocalizationOpenCV(matched_query_keypoints=S.P.T, corresponding_landmarks=S.X.T, K = intrinsicMatrix)

  # non-linear refinement
  R_C_W_refined, t_C_W_refined = refine_pose_with_nonlinear_optimization(S, R_C_W, t_C_W, intrinsicMatrix)

  # print(f"t_C_W = {t_C_W}")
  # print(f"second keypoints filter, ransac localization")
  # print(f"before filter: {S.P.shape[1]}, after filter: {sum(inlier_mask)}")
  # second filter to keypoints, discard umatched landmarks & keypoints
  S.P = S.P[:, inlier_mask>0]
  S.X = S.X[:, inlier_mask>0]
  
  T_C_W = rt_to_homogeneous(R=R_C_W_refined, t=t_C_W_refined)
  T_W_C = np.linalg.inv(T_C_W)
  
  # part 2: candidates tracking & triangluations
  # update already tracked candidates: KLT tracking
  S.P, S.X, S.C, S.F, S.T = update_landmarks(S, intrinsicMatrix, prev_img, curr_img, R_C_W, t_C_W)
  # print(f"landmarks: {S.X}")
  
  # part 3: add new candidates
  # extract candidate feature from current image
  '''
  candidates_keypoints, candidates_feature_descriptors = feature_detection(curr_img) # Returns: Tuple[List[cv2.KeyPoint], np.ndarray]:
  candidates_keypoints_coordinates = [candidates_keypoint.pt for candidates_keypoint in candidates_keypoints] # obtain keypoints coordinates (u, v) (col, row)
  candidates_keypoints_coordinates = np.array(candidates_keypoints_coordinates).T
  print(f"candidates_keypoints_coordinates.shape = {candidates_keypoints_coordinates.shape}")
  print(f"candidates_feature_descriptors.shape = {candidates_feature_descriptors.shape}")
  assert candidates_keypoints_coordinates.shape[-1]==candidates_feature_descriptors.T.shape[-1], f"number of candidates' coordinates:{candidates_keypoints_coordinates.shape[-1]} doesn't match number of corresponding descriptors:{candidates_feature_descriptors.shape[-1]}"
  
  discard_radius = 10
  
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
  

  # TODO:update candidates
  if candidates_keypoints_coordinates.shape[1] > 0:
    print(f'candidates_keypoints_coordinates.shape = {candidates_keypoints_coordinates.shape}')
    S.C = np.hstack([S.C, candidates_keypoints_coordinates])
    S.F = np.hstack([S.F, candidates_keypoints_coordinates])
    print(f'S.T.shape = {S.T.shape}, T_W_C.shape = {T_W_C.shape}')
    T_W_C_repeat = np.repeat(T_W_C[:,:, np.newaxis], candidates_keypoints_coordinates.shape[1], axis=2)
    S.T = np.concatenate((S.T, T_W_C_repeat), axis=2)
      
  print(f'number of candidates C:{S.C.shape[1]}, F:{S.F.shape[1]}, T:{S.T.shape[-1]}')
  print("\n\n a round of continous operation complete \n\n")
  '''
  S = update_candidates(curr_img, S, intrinsicMatrix, T_W_C, discard_radius=15)
  
  S.num_kpts = np.append(S.num_kpts, S.P.shape[-1])
  S.num_candidates = np.append(S.num_candidates, S.C.shape[-1])
  S.poses = np.concatenate((S.poses, T_W_C[:,:,np.newaxis]), axis=2)
  # print(S.poses)
  # print(f"T_W_C = {T_W_C}, T_C_W = {T_C_W}")
  
  # TODO: configurate return
  return T_W_C, S

def filter_landmarks_behind_camera(pose, landmarks):
    R = pose[:3, :3]  
    t = pose[:3, 3].reshape(-1, 1) 
    
    landmarks_cam = R @ landmarks + t 

    valid_indices = landmarks_cam[2, :] > 0 
    filtered_landmarks = landmarks[:, valid_indices] 

    return filtered_landmarks, valid_indices