# Mini Project: A Visual Odometry Pipeline

This project implements a visual odometry pipeline capable of estimating camera poses and reconstructing 3D landmarks from image sequences. The pipeline has been tested on three datasets: KITTI, Malaga Urban Dataset, and Parking.

## Project Structure

Mini-project-A-visual-odometry-pipeline
├── kitti                           # KITTI dataset (download link below)
├── malaga-urban-dataset-extract-07 # Malaga Urban Dataset (download link below)
├── parking                         # Parking dataset (download link below)
├── src                             # Source code for the project
│   ├── main.py                     # Entry point for the pipeline
│   ├── feature_detection.py        # Feature detection methods
│   ├── KLT_tracker                 # KLT tracking implementation
│   ├── update_landmarks.py         # Landmark update logic
│   ├── update_candidate.py         # Candidate feature management
│   ├── visualization.py            # Visualization tools
│   ├── util.py                     # Utility functions
│   └── …                         # Additional modules
└── .gitignore                      # Git ignore file

## Datasets

The pipeline supports the following datasets. Please download the datasets and place them in the project directory at the same level as the `src` folder:

1. **KITTI Dataset**
   - **Download link:** [KITTI 05](https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip)
   - **Description:** Structured urban environment with clear features for tracking.
   - **Folder name:** `kitti`

2. **Malaga Urban Dataset**
   - **Download link:** [Malaga Urban Dataset Extract 07](https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip)
   - **Description:** Complex dataset with high frame rates, varying lighting, and texture-less scenes.
   - **Folder name:** `malaga-urban-dataset-extract-07`

3. **Parking Dataset**
   - **Download link:** [Parking Dataset](https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip)
   - **Description:** Simplified environment with static objects for testing triangulation and tracking.
   - **Folder name:** `parking`

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git


2.	Place the datasets in the project directory at the same level as the src folder:

Mini-project-A-visual-odometry-pipeline
├── kitti
├── malaga-urban-dataset-extract-07
├── parking
└── src


3.	Navigate to the src folder:
cd src


4.	Run the pipeline:
python main.py

	5.	Follow the prompts to select a dataset and observe the results.

Results

The pipeline generates visualizations of the estimated camera trajectory, 3D landmarks, and tracking performance. Examples can be found in the project documentation.

Challenges and Future Work

The pipeline faces challenges such as:
	•	Keypoint dropout during tracking.
	•	High outlier rates in pose estimation.
	•	Triangulation inconsistencies.

Future improvements include implementing bundle adjustment for refining pose and landmark estimates and exploring robust pose estimation techniques.









