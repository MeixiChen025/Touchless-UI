import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv("JESTER_DATASET_PATH")

LABEL_DICT = {
    "Thumb Up": 0,
    "Thumb Down": 1,
    "Swiping Up": 2,
    "Swiping Down": 3,
    "Swiping Left": 4,
    "Swiping Right": 5,
    "Stop Sign": 6
}

class JesterMiddleFrameDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.split_folder = "Train" if "train" in csv_file.lower() else "Validation"
        self.root_dir = os.path.join(root_dir, self.split_folder)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        video_id = str(self.data_frame.iloc[idx, 0])
        label_str = self.data_frame.iloc[idx, 1]
        label_idx = LABEL_DICT[label_str]
        
        video_folder = os.path.join(self.root_dir, video_id)
        image_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
        middle_idx = len(image_files) // 2
        img_path = os.path.join(video_folder, image_files[middle_idx])
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        cropped = img_rgb[start_y:start_y+size, start_x:start_x+size]
        
        final_img = cv2.resize(cropped, (112, 112))
        
        image_tensor = self.transform(final_img)
        
        return image_tensor, label_idx

if __name__ == "__main__":
    csv_path = "processed_labels/filtered_train.csv"
    if not os.path.exists(csv_path):
        csv_path = "filtered_train.csv" 
        
    train_dataset = JesterMiddleFrameDataset(csv_file=csv_path, root_dir=DATASET_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    images, labels = next(iter(train_loader))
    
    print(f" Images Tensor: {images.shape}")
    print(f" Labels Tensor: {labels.shape}")
    print(f" Labels: {labels}")