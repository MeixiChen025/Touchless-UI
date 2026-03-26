import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_prep.gesture_dataset_3d import Jester3DDataset, DATASET_PATH

class GestureCNN_LSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=256, num_lstm_layers=1):
        super(GestureCNN_LSTM, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')        
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
        
        cnn_features = self.cnn(x) # shape: (Batch * Time, 512, 1, 1)
        cnn_features = cnn_features.view(b, t, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(cnn_features) # lstm_out shape: (Batch, Time, hidden_size)
        
        last_time_step_out = lstm_out[:, -1, :] # shape: (Batch, hidden_size)
        
        out = self.fc(last_time_step_out) 
        return out

def main():
    device = torch.device("cpu")

    train_csv = "processed_labels/filtered_train.csv" if os.path.exists("processed_labels/filtered_train.csv") else "filtered_train.csv"
    val_csv = "processed_labels/filtered_val.csv" if os.path.exists("processed_labels/filtered_val.csv") else "filtered_val.csv"
    
    train_dataset = Jester3DDataset(csv_file=train_csv, root_dir=DATASET_PATH, num_frames=8)
    val_dataset = Jester3DDataset(csv_file=val_csv, root_dir=DATASET_PATH, num_frames=8)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = GestureCNN_LSTM(num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)    
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
        print(f"Epoch [{epoch+1}/{num_epochs}] end | Validation Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_final_model.pth')
            print(f" New best final model saved to 'saved_models/best_final_model.pth' with Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()