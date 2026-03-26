GestureFlow: Real-Time Spatiotemporal Hand Gesture Recognition
Overview
GestureFlow is a deep learning framework developed to enable contactless User Interface (UI) control. By interpreting dynamic temporal hand gestures from raw RGB video streams, the system is designed to facilitate sterile interactions in medical environments and immersive control in AR/VR applications.

This project evolved from a baseline 3D Convolutional Neural Network (3D CNN) to a Long-term Recurrent Convolutional Network (LRCN) architecture to address the fundamental challenge of directional ambiguity in time-series visual data.

Technical Stack
Languages & Frameworks: Python, PyTorch, Torchvision

Data Processing & Computer Vision: OpenCV (cv2), Pandas, NumPy

Evaluation & Visualization: Scikit-learn, Matplotlib, Seaborn

Techniques Applied: Transfer Learning, Spatiotemporal Sequence Modeling, L2 Regularization, Dynamic Learning Rate Scheduling.

Data Pipeline & Engineering
Handling high-dimensional video data requires efficient processing pipelines. This project involved building robust data ingestion modules:

Jester Dataset Processing: Filtered and processed the massive 20BN-Jester dataset down to 7 core UI-relevant gesture classes (Swipe Up/Down/Left/Right, Thumb Up/Down, Stop Sign).

Custom Data Ingestion Script: Engineered an automated data bridge (prep_custom_data.py) to parse unstructured .mp4 and .mov video files, extract sequential frames at consistent sampling rates via OpenCV, and automatically generate structured CSV labels for DataLoader consumption.

Model Architecture
The final model utilizes a hybrid CNN-LSTM (LRCN) approach:

Spatial Feature Extraction: A ResNet-18 backbone (excluding the final fully connected layer) processes each frame independently, extracting high-level spatial feature vectors.

Temporal Sequence Modeling: A Long Short-Term Memory (LSTM) network processes the sequence of spatial vectors across 8 frames to capture the temporal gradient and motion direction.

Classification Head: A fully connected layer with aggressive Dropout (p=0.5) outputs the final classification probabilities.

Repository Structure
src/data_prep/: Scripts for processing the Jester dataset and extracting frames from custom raw videos.

src/Final_models/: PyTorch definitions and training loops for the LRCN architecture.

src/evaluation/: Evaluation scripts including automated directory structuring and confusion matrix generation.

saved_models/: Location for saved .pth model weights (excluded via .gitignore).

Custom_Dataset/: Raw .mp4 validation files recorded in unconstrained environments.

How to Run
Clone the repository and navigate to the project root.

Install dependencies (PyTorch, OpenCV, Pandas, Scikit-learn, Seaborn).

Process custom video data:

Bash
python -m src.data_prep.prep_custom_data
Train the LRCN model:

Bash
python -m src.Final_models.train_final
Evaluate on unseen custom data and generate the confusion matrix:

Bash
python -m src.evaluation.eval_custom_data