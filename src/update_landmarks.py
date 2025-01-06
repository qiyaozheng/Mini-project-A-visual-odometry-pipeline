import numpy as np
import cv2

from state import voState
from KLT_tracker.track_klt_opencv import trackKLTOpencvBatch_return_mask
from exercise_code.linear_triangulation import linearTriangulation


def update_landmarks(S:voState, intrinsicMatrix, prev_img, curr_img, R_C_W, t_C_W):
  '''
  input:
    S:voState: tracked information
    intrinsicMatrix: K
    R_C_W
    t_C_W
  output: update information
    S.P,
    S.X,
    S.C,
    S.F,
    S.T
  '''
  if S.C.shape[1]!=0:
    candidates_coordinates_prev, candidates_coordinates_new, valid_mask_candidates = trackKLTOpencvBatch_return_mask(I_prev=prev_img, I=curr_img, keypoints=S.C, r_T=10, n_iter=50)
    # discard untracked candidates & its first observation & pose of first observation
    print(f"candidates before filter: {S.C.shape[1]}, after filter: {candidates_coordinates_new.shape[1]}")
    S.C = candidates_coordinates_new
    S.F = S.F[:, valid_mask_candidates]
    S.T = S.T[:, :, valid_mask_candidates]
    
    '''
    # TODO:triangluate certain candidates 
    # calculate angle for all candidates
    candidates_h = np.vstack((S.C, np.ones(S.C.shape[1]))) # (3, M) homogenous representation
    candidates_h_firstobserved = np.vstack((S.F, np.ones(S.F.shape[1]))) # (3, M) homogenous representation

    candidates_h_normalized =  np.linalg.inv(intrinsicMatrix) @ candidates_h # normalized image coordinates, also the observed ray vector
    candidates_h_f_normalized = np.linalg.inv(intrinsicMatrix) @ candidates_h_firstobserved 
    
    dot_products = np.sum(candidates_h_normalized * candidates_h_f_normalized, axis=0) # shape(M,)
    norms_v1 = np.linalg.norm(candidates_h_normalized, axis=0) # shape(M,)
    norms_v2 = np.linalg.norm(candidates_h_f_normalized, axis=0) 
    cos_angles = dot_products / (norms_v1 * norms_v2)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0)) * (180.0 / np.pi)
    
    
    
    
    # select candidates obeserved angle exceed threshold
    angle_threshold = 5.0
    valid_angle_mask = angles > angle_threshold # bool mask, shape(M,)
    '''
    T_W_C = np.hstack((R_C_W, t_C_W.reshape(3, 1)))
    T_W_C = np.vstack((T_W_C, [0, 0, 0, 1])) 
    valid_angle_mask = calculate_parallax_angles(S, intrinsicMatrix, T_W_C)
    
    
    triangluate_candidates = S.C[:, valid_angle_mask]
    triangluate_candidates_f = S.F[:, valid_angle_mask]
    triangluate_candidates_t = S.T[:, :, valid_angle_mask]
    assert triangluate_candidates.shape[1] == triangluate_candidates_f.shape[1] and triangluate_candidates_f.shape[1] == triangluate_candidates_t.shape[-1], "triangluate candidates number doesnt match"
    
    # triangluate
    if triangluate_candidates.shape[1]>0:
      R1_list = triangluate_candidates_t[:3, :3, :] # rotation matrices
      t1_list = triangluate_candidates_t[:3, 3, :]  #  translation vectors
      Rt_list = np.concatenate((R1_list, t1_list[:,np.newaxis, :]), axis=1) # shape(3, 4, M')
      Project1_list = np.einsum('ij,jkn->ikn', intrinsicMatrix, Rt_list) #shape(3, 4, M')
      
      Project2 = intrinsicMatrix @ np.hstack((R_C_W ,t_C_W.reshape(3, 1)))

      candidates_3d = []
      for i in range(triangluate_candidates.shape[1]):
          Project1 = intrinsicMatrix @ Rt_list[:,:, i]
          
          candidate_4d = cv2.triangulatePoints(
              Project1, Project2, triangluate_candidates_f[:, i:i+1], triangluate_candidates[:, i:i+1]
          )
          candidate_3d = (candidate_4d[:3] / candidate_4d[3]).flatten()  
          # p1_homogeneous = np.vstack((triangluate_candidates[:, i:i+1], np.array([[1]])))
          # p2_homogeneous = np.vstack((triangluate_candidates_f[:, i:i+1], np.array([[1]])))
          # candidate_3d = linearTriangulation(p1=p1_homogeneous, p2 = p2_homogeneous, M1=Project2, M2=Project1)
          # P_cartesian = (candidate_3d[:3] / candidate_3d[3]).flatten()
          candidates_3d.append(candidate_3d)
          
      candidates_3d = np.array(candidates_3d).T  # 形状为 (3, N)
      # print(f"candidates_3d.shape = {candidates_3d.shape}")
      candidate_3d = candidate_3d.reshape(3, -1)
      

      # update landmark, remove triangluated candidates
      S.X = np.hstack((S.X, candidates_3d))  
      S.P = np.hstack((S.P, triangluate_candidates))
      S.C = S.C[:, ~valid_angle_mask]
      S.F = S.F[:, ~valid_angle_mask]
      S.T = S.T[:, :, ~valid_angle_mask]
        
  return S.P, S.X, S.C, S.F, S.T

def calculate_parallax_angles(S, intrinsicMatrix, T_C_W):
    # Current frame's homogeneous coordinates (3xM)
    candidates_h = np.vstack((S.C, np.ones(S.C.shape[1])))
    candidates_h_normalized = np.linalg.inv(intrinsicMatrix) @ candidates_h

    # First observed points' normalized homogeneous coordinates (3xM)
    candidates_h_f_normalized = np.zeros_like(candidates_h_normalized)

    for i in range(S.C.shape[1]):
        # Compute transformation from current frame to first observed frame
        # print("S.T shape:", S.T.shape)
        T_W_C_first = S.T[:, :, i]
        T_C_Cf = np.linalg.inv(T_C_W) @ T_W_C_first

        # Transform the first observed point to current frame's coordinates
        first_point_h = np.hstack((S.F[:, i], [1]))
        candidates_h_f_normalized[:, i] = T_C_Cf[:3, :3] @ (np.linalg.inv(intrinsicMatrix) @ first_point_h)

    # Compute parallax angles
    dot_products = np.sum(candidates_h_normalized * candidates_h_f_normalized, axis=0)
    norms_v1 = np.linalg.norm(candidates_h_normalized, axis=0)
    norms_v2 = np.linalg.norm(candidates_h_f_normalized, axis=0)
    cos_angles = dot_products / (norms_v1 * norms_v2)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0)) * (180.0 / np.pi)

    # Select candidates with parallax angles exceeding the threshold
    angle_threshold = 5.0
    valid_angle_mask = angles > angle_threshold

    return valid_angle_mask