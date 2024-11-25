import numpy as np
import cv2
from pathlib import Path
from vo_initializer import VOInitializer

def setup(dataset_type: int):

    current_dir = Path(__file__).resolve().parent.parent
    
    if dataset_type == 0: #kitti
        dataset_path = current_dir / "kitti"
        K = np.array([[718.8560, 0, 607.1928],
                     [0, 718.8560, 185.2157],
                     [0, 0, 1]])
        last_frame = 4540
        
    elif dataset_type == 1: #Malaga
        dataset_path = current_dir / "malaga-urban-dataset-extract-07"
        K = np.array([[621.18428, 0, 404.0076],
                     [0, 621.18428, 309.05989],
                     [0, 0, 1]])
        img_dir = dataset_path / "Images"
        last_frame = len(list(img_dir.glob("*.jpg")))
        
    elif dataset_type == 2: #Parking
        dataset_path = current_dir / "parking"
        K = np.array([[331.37, 0, 320],
                  [0, 369.568, 240],
                  [0, 0, 1]])
        last_frame = 598
    
    else:
        raise ValueError("Invalid dataset type!")
        
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist!"
    return K, last_frame, dataset_path

def load_image(dataset_type: int, path: Path, frame_idx: int) -> np.ndarray:

    if dataset_type == 0:  #kitti
        img_path = path / "05/image_0" / f"{frame_idx:06d}.png"
        img = cv2.imread(str(img_path), 0) 

    elif dataset_type == 1:  # Malaga
        img_dir = path / "malaga-urban-dataset-extract-07_rectified_800x600_Images"
        img_list = sorted(list(img_dir.glob("*_left.jpg")))
        img_path = img_list[frame_idx-1]
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    elif dataset_type == 2: #Parking
        img_path = path / "images" / f"img_{frame_idx:05d}.png"
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    assert img_path.exists(), f"Image {img_path} does not exist!"
    assert img is not None, f"Failed to read image: {img_path}"
    return img

def select_dataset():
    print("\nAvailable datasets:")
    print("0: KITTI")
    print("1: Malaga")
    print("2: Parking")
    while True:
        try:
            choice = int(input("\nPlease select dataset type (0-2): "))
            if choice in [0, 1, 2]:
                return choice
            else:
                print("Invalid choice! Please select 0, 1, or 2.")
        except ValueError:
            print("Invalid input! Please enter a number.")

def main():

    dataset_type = select_dataset() 
    print(f"\nSelected dataset: {['KITTI', 'Malaga', 'Parking'][dataset_type]}")
    K, last_frame, dataset_path = setup(dataset_type)
    print(f"Using dataset path: {dataset_path}")
    
    bootstrap_frames = [1, 3]
    frame1 = load_image(dataset_type, dataset_path, bootstrap_frames[0])
    frame2 = load_image(dataset_type, dataset_path, bootstrap_frames[1])
    
    initializer = VOInitializer(K)
    success, R, t, points3D, pts1, pts2 = initializer.initialize(frame1, frame2)
    
    if success:
        print("Initialization successful!")
        print(f"- Number of 3D points: {len(points3D)}")
        print(f"- Translation vector: {t.reshape(-1)}")
    else:
        print("Initialization failed!")
        return
    
    # Continuous operation (will be implemented later)
    print("\nStarting continuous operation...")
    for i in range(bootstrap_frames[1] + 1, last_frame):
        """ 
        TODO: implement continuous operation
        """ 

if __name__ == "__main__":
    main()