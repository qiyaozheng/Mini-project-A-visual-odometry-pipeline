import numpy as np
import cv2

DISTANCE_RATIO = 0.7

def feature_matching(kp1: cv2.KeyPoint, kp2:cv2.KeyPoint, des1:np.ndarray, des2:np.ndarray):
  '''
  look for corresponding kp2 for every kp1,
  
  Returns length equal to len(kp1)
  Returns:
    List(cv2.Keypoint): kpset1
    List(cv2.Keypoint): kpset2, matched keypoints, includes its coordinate in img2 
  '''
  pass
  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)
  good_matches = []
  kpset1 = []
  kpset2 = []
  mask1 = np.zeros(len(kp1), dtype=bool)  # query图像的关键点 mask
  mask2 = np.zeros(len(kp2), dtype=bool)  # train图像的关键点 mask
  for m, n in matches:
      if m.distance < DISTANCE_RATIO * n.distance:
          good_matches.append(m)
          kpset1.append(kp1[m.queryIdx])
          kpset2.append(kp2[m.trainIdx])
          mask1[m.queryIdx] = True
          mask2[m.trainIdx] = True
  return kpset1, kpset2, good_matches, mask1, mask2
  
  
  
'''
   def _detect_and_match_features(self, 
                                   img1: np.ndarray, 
                                   img2: np.ndarray):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None) #keypoints, description
        kp2, des2 = sift.detectAndCompute(img2, None)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        pts1 = []
        pts2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        return np.float32(pts1), np.float32(pts2), good_matches
'''