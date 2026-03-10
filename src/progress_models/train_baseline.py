import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
from src.data_prep.gesture_dataset import JesterMiddleFrameDataset, DATASET_PATH

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 112x112 -> 56x56
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 56x56 -> 28x28
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cpu")
    
    train_csv = "processed_labels/filtered_train.csv"
    val_csv = "processed_labels/filtered_val.csv"

    train_dataset = JesterMiddleFrameDataset(csv_file=train_csv, root_dir=DATASET_PATH)
    val_dataset = JesterMiddleFrameDataset(csv_file=val_csv, root_dir=DATASET_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = BaselineCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad() 
            loss.backward()      
            optimizer.step()     
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        model.eval() 
        correct = 0
        total = 0
        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f" Epoch [{epoch+1}/{num_epochs}] end | average train loss: {running_loss/len(train_loader):.4f} | validation set accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()