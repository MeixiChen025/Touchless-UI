import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_prep.gesture_dataset_3d import Jester3DDataset, LABEL_DICT

class GestureCNN_LSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=256, num_lstm_layers=1):
        super(GestureCNN_LSTM, self).__init__()
        
        resnet = models.resnet18(weights=None) 
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.5 if num_lstm_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(b, t, -1)
        lstm_out, _ = self.lstm(cnn_features)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

def evaluate_on_custom_data():
    device = torch.device("cpu")
    
    custom_csv = "custom_val.csv"
    custom_root = "Custom_Frames"
    
    if not os.path.exists(custom_csv) or not os.path.exists(custom_root):
        print("Error: Cannot find Custom_Dataset files. Run prep_custom_data.py first!")
        return
        
    validation_dir = os.path.join(custom_root, "Validation")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
        for item in os.listdir(custom_root):
            if item.startswith("custom_") and os.path.isdir(os.path.join(custom_root, item)):
                shutil.move(os.path.join(custom_root, item), os.path.join(validation_dir, item))
        
    custom_dataset = Jester3DDataset(csv_file=custom_csv, root_dir=custom_root, num_frames=8)
    custom_loader = DataLoader(custom_dataset, batch_size=4, shuffle=False)
    
    model = GestureCNN_LSTM(num_classes=7).to(device)
    model_path = 'saved_models/best_final_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    
    with torch.no_grad():
        for videos, labels in custom_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"Custom Dataset Accuracy: {accuracy:.2f}%")

    
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = [""] * 7
    for name, idx in LABEL_DICT.items():
        class_names[idx] = name
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('Confusion Matrix: Final Model on Custom Unseen Data', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('custom_confusion_matrix.png', dpi=300)
    print("Confusion matrix saved to 'custom_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_on_custom_data()