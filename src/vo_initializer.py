import numpy as np
import cv2

class VOInitializer:

    """
    This class initializes the visual odometry (VO) pipeline by reconstructing 3D points and estimating the 
    relative pose (rotation and translation) between two frames of a monocular camera. First, the `__init__` 
    method sets up the initialization parameters, including the camera intrinsic matrix `K`, the minimum number 
    of matches required for reliable initialization, and the RANSAC threshold. Next, the `_detect_and_match_features` 
    function detects and matches keypoints between two input frames using the SIFT algorithm and FLANN-based matcher, 
    returning good matches and corresponding keypoints. Then, the `_triangulate_points` function uses matched keypoints 
    and the recovered pose to calculate 3D landmarks by triangulation. After that, the `_check_reconstruction_quality` 
    function ensures the validity of the reconstructed 3D points by checking their depth and number. Finally, the 
    `initialize` function orchestrates the entire process: it matches features, computes the essential matrix `E` 
    and recovers the relative pose (R, t), triangulates 3D points, and verifies the reconstruction quality. This is 
    the main function of the module, enabling the initialization of the VO pipeline.
    """

    def __init__(self, K: np.ndarray):
        self.K = K
        self.min_matches = 100
        self.ransac_thresh = 1.0

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
    
    def _triangulate_points(self, 
                            pts1: np.ndarray, 
                            pts2: np.ndarray, 
                            R: np.ndarray, t: 
                            np.ndarray):
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = self.K @ np.hstack((R, t))
        points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3D = (points4D[:3, :] / points4D[3, :]).T
        return points3D
    
    def _check_reconstruction_quality(self, points3D: np.ndarray):
        depths = points3D[:, 2]
        if np.median(depths) < 0:
            print("Invalid reconstruction: majority of points behind the camera")
            return False
        if len(points3D) < 50:
            print("Insufficient 3D points for reliable reconstruction")
            return False
        return True
    
    def initialize(self, 
                   frame1: np.ndarray, 
                   frame2: np.ndarray):

        pts1, pts2, matches = self._detect_and_match_features(frame1, frame2)
        if len(matches) < self.min_matches:
            print(f"Too few matches: {len(matches)} < {self.min_matches}")
            return False, None, None, None, None, None
        
        F, mask = cv2.findFundamentalMat(pts1, 
                                            pts2, 
                                            cv2.FM_RANSAC, 
                                            self.ransac_thresh, 
                                            0.99)
        if F is None:
            print("Failed to compute fundamental matrix")
            return False, None, None, None, None, None
        E = self.K.T @ F @ self.K #Relationships in the normalized camera plane where camera intrinsic effects have been removed
        # E = [t]_x * R

        _, R, t, mask = cv2.recoverPose(E, 
                                        pts1, 
                                        pts2, 
                                        self.K, 
                                        mask=mask)
        
        inlier_pts1 = pts1[mask.ravel()==1]
        inlier_pts2 = pts2[mask.ravel()==1]
        points3D = self._triangulate_points(inlier_pts1, inlier_pts2, R, t)
        
        if not self._check_reconstruction_quality(points3D):
            print("Poor 3D reconstruction quality")
            return False, None, None, None, None, None
        
        return True, R, t, points3D, inlier_pts1, inlier_pts2