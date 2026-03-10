import torch
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from src.data_prep.gesture_dataset_3d import Jester3DDataset, DATASET_PATH, LABEL_DICT

class Primary3DCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(Primary3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 2 * 28 * 28, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def plot_prediction():
    device = torch.device("cpu")
    
    model = Primary3DCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load('best_primary_model.pth', map_location=device))
    model.eval() 
    
    val_csv = "processed_labels/filtered_val.csv" if os.path.exists("processed_labels/filtered_val.csv") else "filtered_val.csv"
    val_dataset = Jester3DDataset(csv_file=val_csv, root_dir=DATASET_PATH, num_frames=8)
    
    sample_idx = 42 
    video_tensor, true_label_idx = val_dataset[sample_idx]
    
    inv_label_dict = {v: k for k, v in LABEL_DICT.items()}
    true_label_name = inv_label_dict[true_label_idx]
    
    with torch.no_grad():
        input_tensor = video_tensor.unsqueeze(0).to(device) 
        raw_scores = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(raw_scores[0], dim=0).numpy() * 100

    labels = list(LABEL_DICT.keys())
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, probabilities, color='skyblue')
    
    bars[true_label_idx].set_color('salmon')
    
    plt.xlabel('Probability (%)')
    plt.title(f'Model Prediction Confidence\nTrue Label: {true_label_name}')
    plt.xlim(0, 100)
    

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
        
    plt.gca().invert_yaxis()
    plt.savefig('qualitative_result.png', dpi=300, bbox_inches='tight')
    print(f"true label is '{true_label_name}'Diagram save to qualitative_result.png")
    plt.show()

if __name__ == "__main__":
    plot_prediction()