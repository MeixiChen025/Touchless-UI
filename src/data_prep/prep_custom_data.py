import cv2
import os
import csv

def process_custom_videos():
    input_root = "Custom_Dataset"       
    output_root = "Custom_Frames"        
    csv_file_path = "custom_val.csv"    
    
    os.makedirs(output_root, exist_ok=True)
    
    csv_data = []
    video_counter = 1
    
    for gesture_name in os.listdir(input_root):
        gesture_dir = os.path.join(input_root, gesture_name)
        
        if not os.path.isdir(gesture_dir):
            continue
            
        
        for video_file in os.listdir(gesture_dir):
            if not (video_file.endswith('.mp4') or video_file.endswith('.mov')):
                continue
                
            video_path = os.path.join(gesture_dir, video_file)
            
            video_id = f"custom_{video_counter:05d}"
            video_out_dir = os.path.join(output_root, video_id)
            os.makedirs(video_out_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 1
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break 
                
                frame_filename = f"{frame_count:05d}.jpg"
                frame_path = os.path.join(video_out_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                
            cap.release()

            if frame_count == 1:
                os.rmdir(video_out_dir)
                continue 
            
            formatted_label = gesture_name.replace("_", " ").title() 

            
            csv_data.append([video_id, formatted_label])
            video_counter += 1

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for row in csv_data:
            writer.writerow(row)
            
    print(f"\n Processed {video_counter-1} videos.")
    print(f"Frames saved in: {output_root}/")
    print(f"Labels saved in: {csv_file_path}")

if __name__ == "__main__":
    process_custom_videos()