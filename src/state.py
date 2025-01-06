import cv2
from typing import List
import numpy as np

class voState:
  '''
  attributs:
    P: tracked keypoints (coordinates correspond to tracked 3D landmark X, format: [x, y]==[col, row]), shape: (2, N) type ndarray
    X: tracked 3D landmark (correspond to tracked keypoints P, format: [X, Y, Z]), shape: (3, N)
    C: keypoints coorinates candidates, changes when perform KLT_tracking for candidates
    F: first observed coordinates of candidates, do not change when perform KLT_tracking for candidates
    T: camera poses of first observation, shape (4,4, M)
    poses: camera poses of all frames shape (4, 4, i), i = number of  img frame
    num_kpts: numer of keypoints of every frame
    num_candidates: number of candidates of every frame 
  '''
  def __init__(self, keypoints, landmarks):
    pass
    self.P = keypoints
    self.X = landmarks
    self.C = np.empty((2, 0))
    self.F = np.empty((2, 0))
    self.T = np.empty((4, 4, 0))
    self.poses = np.empty((4, 4, 0))
    self.num_kpts = np.array([self.P.shape[-1]])
    self.num_candidates = np.array([self.C.shape[-1]])
    
    self.P_prev = keypoints
    self.P_curr = keypoints
    
  def getKeypointCoordinates(self):
    keypoint_coordinates = np.array([kp.pt for kp in self.P], dtype=np.float32)
    return keypoint_coordinates
  def getCandidateCoordinates(self):
    candidate_coordinates = np.array([kp.pt for kp in self.C], dtype=np.float32)
    return candidate_coordinates
    
class candidate:
  def __init__(self, pt, first_observation_pose_index, des):
    pass
    self.pt = pt
    self.des = des
    self.first_observation_pose_index = first_observation_pose_index
    