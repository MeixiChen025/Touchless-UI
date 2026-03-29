import torch
import os
from torch.utils.data import DataLoader
from src.data_prep.gesture_dataset_3d import Jester3DDataset, LABEL_DICT
from src.evaluation.eval_custom_data import GestureCNN_LSTM

def find_thumb_mismatches():
    device = torch.device("cpu")
    
    custom_csv = "custom_val.csv"
    custom_root = "Custom_Frames"
    
    dataset = Jester3DDataset(csv_file=custom_csv, root_dir=custom_root, num_frames=8)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = GestureCNN_LSTM(num_classes=7).to(device)
    model.load_state_dict(torch.load('saved_models/best_final_model.pth', map_location=device))
    model.eval()

    try:
        target_true_idx = LABEL_DICT["Thumb Up"]
        target_pred_idx = LABEL_DICT["Swiping Right"]
    except KeyError:
        return

    print(f"find True='Thumb Up' & Pred='Swiping Right':")
    found_any = False

    with torch.no_grad():
        for i, (video, label) in enumerate(loader):
            video = video.to(device)
            output = model(video)
            _, pred = torch.max(output, 1)
            
            true_val = label.item()
            pred_val = pred.item()
            
            if true_val == target_true_idx and pred_val == target_pred_idx:
                video_id = dataset.data_frame.iloc[i, 0]
                print(f"Finad Case ID: {video_id}")
                found_any = True
    
    if not found_any:
        print("Null this type mistake")


if __name__ == "__main__":
    find_thumb_mismatches()