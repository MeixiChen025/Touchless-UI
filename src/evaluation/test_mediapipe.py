import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv("JESTER_DATASET_PATH")

if not DATASET_PATH:
    raise ValueError("Path not found. Please check the .env file.")

VIDEO_ID = "31"  

video_folder = os.path.join(DATASET_PATH, "Train", VIDEO_ID)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def test_single_image():
    if not os.path.exists(video_folder):
        print(f"Can not find the folder")
        return
        
    image_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    

    middle_frame_name = image_files[len(image_files) // 2]
    img_path = os.path.join(video_folder, middle_frame_name)
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return
        
    h, w, _ = img.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for landmark in results.multi_hand_landmarks[0].landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
        
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cropped_hand = img_rgb[y_min:y_max, x_min:x_max]
    
    resized_hand = cv2.resize(cropped_hand, (112, 112))
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Frame (Jester Dataset)")
    axes[0].axis('off')
    
    axes[1].imshow(resized_hand)
    axes[1].set_title("Cropped & Resized (112x112)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("mediapipe_demo.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_single_image()