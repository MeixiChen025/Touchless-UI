import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.eval_custom_data import GestureCNN_LSTM
from src.data_prep.gesture_dataset_3d import LABEL_DICT

MODEL_PATH = os.path.join('saved_models', 'best_final_model.pth')
FAILING_VIDEO_DIR = os.path.join('Custom_Frames', 'Validation', 'custom_00161') 
OUTPUT_PLOT_PATH = 'temporal_failure_trajectory.png'

def progressive_inference_and_plot():
    device = torch.device("cpu") 
    model = GestureCNN_LSTM(num_classes=7).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(FAILING_VIDEO_DIR):
        print(f"No Path: {FAILING_VIDEO_DIR}")
        return

    all_image_files = sorted([f for f in os.listdir(FAILING_VIDEO_DIR) if f.endswith('.jpg')])
    total_frames = len(all_image_files)
    
    if total_frames < 8:
        return

    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    sampled_image_files = [all_image_files[i] for i in indices]
    
    frames_tensor_list = []
    for img_file in sampled_image_files:
        img_path = os.path.join(FAILING_VIDEO_DIR, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        frames_tensor_list.append(img_tensor)

    true_probs_history = []
    false_probs_history = []
    
    true_label_name = "Thumb Up"
    false_label_name = "Swiping Right"
    
    true_idx = LABEL_DICT[true_label_name]
    false_idx = LABEL_DICT[false_label_name]

    
    for seq_len in range(1, 9):
        current_seq_tensors = frames_tensor_list[:seq_len]
        video_tensor = torch.stack(current_seq_tensors).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        true_probs_history.append(probabilities[true_idx].item())
        false_probs_history.append(probabilities[false_idx].item())
        
    print(f"\n True {true_label_name} Final P: {true_probs_history[-1]:.2f}\n Predicted {false_label_name} Final P: {false_probs_history[-1]:.2f}")

    frames_axis = np.arange(1, 9)
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(frames_axis, false_probs_history, 
             color='#1f77b4', marker='o', markersize=8, linestyle='-', linewidth=3, 
             label=f'Probability for "{false_label_name}" (Incorrect Prediction)')
    
    plt.plot(frames_axis, true_probs_history, 
             color='#d62728', marker='s', markersize=8, linestyle='--', linewidth=3, 
             label=f'Probability for "{true_label_name}" (True Label)')
    
    plt.title(f'Temporal Failure Trajectory: "{true_label_name}" misclassified as "{false_label_name}"', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Frame Number in Dynamic Sequence (Temporal Dimension)', fontsize=12, fontweight='bold')
    plt.ylabel('Model Prediction Probability (Softmax)', fontsize=12, fontweight='bold')
    
    plt.xticks(frames_axis, fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    plt.xlim(0.5, 8.5)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    progressive_inference_and_plot()