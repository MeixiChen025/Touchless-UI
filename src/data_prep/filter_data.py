import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("JESTER_DATASET_PATH")
if not DATASET_PATH:
    raise ValueError("Cannot find the dataset path. Please create a .env file in the root directory and set the JESTER_DATASET_PATH")

TARGET_CLASSES = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Swiping Down",
    "Thumb Up",
    "Thumb Down",
    "Stop Sign"
]

def clean_jester_labels(csv_name, output_name):
    csv_path = os.path.join(DATASET_PATH, csv_name)
    
    if not os.path.exists(csv_path):
        print(f"Can not find: {csv_path}")
        return

    
    df = pd.read_csv(csv_path)

    filtered_df = df[df['label'].isin(TARGET_CLASSES)]
    
    print(f"Video before: {len(df)}")
    print(f"Video kept(now): {len(filtered_df)}")
    print("Distribution by category:")
    class_counts = filtered_df['label'].value_counts()
    print(class_counts.to_string())
    
    output_dir = os.path.join(os.getcwd(), "processed_labels")
    os.makedirs(output_dir, exist_ok=True) 
    
    output_path = os.path.join(output_dir, output_name)
    filtered_df.to_csv(output_path, index=False)
    print(f"Save to: {output_path}\n")

if __name__ == "__main__":
    clean_jester_labels("Train.csv", "filtered_train.csv")
    clean_jester_labels("Validation.csv", "filtered_val.csv")
