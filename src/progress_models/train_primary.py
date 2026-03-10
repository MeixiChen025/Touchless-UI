import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from src.data_prep.gesture_dataset_3d import Jester3DDataset, DATASET_PATH

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
def main():
    device = torch.device("cpu")

    train_csv = "processed_labels/filtered_train.csv" if os.path.exists("processed_labels/filtered_train.csv") else "filtered_train.csv"
    val_csv = "processed_labels/filtered_val.csv" if os.path.exists("processed_labels/filtered_val.csv") else "filtered_val.csv"
    
    train_dataset = Jester3DDataset(csv_file=train_csv, root_dir=DATASET_PATH, num_frames=8)
    val_dataset = Jester3DDataset(csv_file=val_csv, root_dir=DATASET_PATH, num_frames=8)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = Primary3DCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 20 
    best_val_acc = 0.0 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f" Epoch [{epoch+1}/{num_epochs}] end | validation set Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_primary_model.pth')
            print(f"New best model save to best_primary_model.pth")

if __name__ == "__main__":
    main()