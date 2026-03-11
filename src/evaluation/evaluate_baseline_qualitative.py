import torch
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data_prep.gesture_dataset import JesterMiddleFrameDataset, DATASET_PATH, LABEL_DICT

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def run_baseline_evaluation():
    device = torch.device("cpu")
    
    model = BaselineCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load('baseline_model.pth', map_location=device))
    model.eval() 
    
    val_csv = "processed_labels/filtered_val.csv"
    val_dataset = JesterMiddleFrameDataset(csv_file=val_csv, root_dir=DATASET_PATH)

    sample_idx = 10  
    image_tensor, true_label_idx = val_dataset[sample_idx]
    
    inv_label_dict = {v: k for k, v in LABEL_DICT.items()}
    true_label_name = inv_label_dict[true_label_idx]

    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        raw_scores = model(input_tensor)
        probabilities = torch.nn.functional.softmax(raw_scores[0], dim=0).cpu().numpy() * 100

    labels = list(LABEL_DICT.keys())
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, probabilities, color='gray')
    
    bars[true_label_idx].set_color('salmon')
    
    plt.xlabel('Probability (%)')
    plt.title(f'Baseline Model Prediction Confidence (Single Frame)\nTrue Label: {true_label_name}')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('baseline_qualitative_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Qualitative result saved as baseline_qualitative_result.png")
    for i, label in enumerate(labels):
        print(f"{label}: {probabilities[i]:.2f}%")

if __name__ == "__main__":
    run_baseline_evaluation()