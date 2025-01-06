import numpy as np
import cv2
import matplotlib.pyplot as plt
from state import voState
import time

def visualization(landmark, camera_poses: np.ndarray):
  
  camera_positions = np.array([pose[:3, 3] for pose in camera_poses])
  
  plt.clf()
  plt.close()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(landmark[:, 0], landmark[:, 1], landmark[:, 2], s=1, c=0.5*np.ones(len(landmark)), label = "landmark")
  ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
           s=20, c='red', label="Camera positions")
  ax.legend(loc='upper right')
  # 设置视角 (elev: 俯视角度, azim: 方位角)
  ax.view_init(elev=0, azim=90)
  ax.set_xlabel("X (meters)")
  ax.set_ylabel("Y (meters)")
  ax.set_zlabel("Z (meters)")
  plt.show()

def visualAll(curr_img, S: voState, fig_idx=1, img_frame_num=0, max_depth=10):
  
  plt.figure(fig_idx, figsize=(16, 9))
  plt.clf()
  
  # subgraph 1: visualize candidates and keypoints on current img
  ax1 = plt.subplot(2, 4, (1, 2))
  ax1.imshow(curr_img, cmap='gray')
  if S.C.size>0:
    ax1.scatter(S.C[0, :], S.C[1, :], s=20, c='blue', marker='x', label='candidates')
  if S.P.size>0:
    ax1.scatter(S.P[0, :], S.P[1, :], s=20, c='g', marker='o', label='keypoints')
  ax1.set_title(f"Current frame: No. {img_frame_num}")
  ax1.legend(fontsize=8)
  
  # subgraph 2: trend of number of keypoints
  ax2 = plt.subplot(2, 4, 5)
  
  ax2.plot(S.num_kpts, 'g', label='number of keypoints')
  ax2.plot(S.num_candidates, 'blue', label='number of candidates')
  ax2.set_title(f"number of track points")
  ax2.legend(fontsize=8)
  
  # subgraph 3: trajectory ()
  ax3 = plt.subplot(2, 4, 6)
  positions = S.poses[:3, 3, :] # extract translation
  ax3.plot(positions[0, :], positions[2, :], 'r*-', markersize=3, label='Estimated pose') 
  ax3.set_title("Estimated trajectory")
  ax3.axis('equal')
  ax3.legend(fontsize=8, loc='lower left')
  
  # subgraph 4: 
  ax4 = plt.subplot(2, 4, (7, 8))
  if S.X.size > 0:  # 如果三维点云非空
    ax4.scatter(S.X[0, :], S.X[2, :], s=50, c='b', marker='x', label='landmarks')
    
  positions = S.poses[:3, 3, :]  # 提取平移向量
  if positions.shape[1] < 20:
      ax4.plot(positions[0, :], positions[2, :], 'r.-', markersize=10, label='trajectory')
  else:
      ax4.plot(positions[0, -20:], positions[2, -20:], 'r.-', markersize=10, label='recent trajectory')
      ax4.plot(positions[0, :-20], positions[2, :-20], color=[0.5, 0.5, 0.5], label='past trajectory')

  # 动态调整视图范围
  fr_win = 15
  if positions.shape[1] > fr_win:
      x_scale = positions[0, -1] - positions[0, -fr_win]
      y_scale = positions[2, -1] - positions[2, -fr_win]
      if abs(x_scale) > abs(y_scale):
          ax4.set_xlim(positions[0, -1] - max_depth/2, positions[0, -1] + max_depth/2)
          ax4.set_ylim(positions[2, -1] - max_depth/2, positions[2, -1] + max_depth/2)
      else:
          ax4.set_xlim(positions[0, -1] - max_depth/2, positions[0, -1] + max_depth/2)
          ax4.set_ylim(positions[2, -1] - max_depth/2, positions[2, -1] + max_depth/2)
  ax4.set_title("landmarks")
  ax4.legend(fontsize=12)
  
  # subgraph 5:
  ax5 = plt.subplot(2, 4, (3, 4))
  ax5.imshow(curr_img, cmap='gray')
  keypoints_ud = np.flipud(S.P_curr)
  kpold_ud = np.flipud(S.P_prev)
  x_from = keypoints_ud[0, :]
  x_to = kpold_ud[0,:]
  y_from = keypoints_ud[1, :]
  y_to = kpold_ud[1,:]
  ax5.plot(np.r_[y_from[np.newaxis, :], y_to[np.newaxis,:]], 
                np.r_[x_from[np.newaxis,:], x_to[np.newaxis,:]], 'g-',
                linewidth=3)
  ax5.set_xlim([0, curr_img.shape[1]])
  ax5.set_ylim([curr_img.shape[0], 0])
  ax5.set_title("KLT tracking")
  
  plt.tight_layout()
  plt.show(block=False)
  plt.pause(0.1)
  #time.sleep(0.1)