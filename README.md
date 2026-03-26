# GestureFlow: Real-Time Spatiotemporal Hand Gesture Recognition

> **Project Focus:** Contactless UI Control, Deep Learning, Spatiotemporal Analysis, Domain Shift.

---

## 1. Overview
**GestureFlow** is a deep learning framework developed to enable *contactless* User Interface (UI) control. By interpreting dynamic temporal hand gestures from raw RGB video streams, the system is designed to facilitate **sterile interactions** in medical environments and **immersive control** in AR/VR applications. 

This project evolved from a baseline **3D Convolutional Neural Network (3D CNN)** to a **Long-term Recurrent Convolutional Network (LRCN)** architecture to address the fundamental challenge of directional ambiguity in time-series visual data.

## 2. Technical Stack
* **Languages & Frameworks:** Python, PyTorch, Torchvision
* **Data Processing & CV:** OpenCV (`cv2`), Pandas, NumPy
* **Evaluation & Visualization:** Scikit-learn, Matplotlib, Seaborn
* **Techniques Applied:** *Transfer Learning, Spatiotemporal Sequence Modeling, L2 Regularization, Dynamic Learning Rate Scheduling.*

---

## 3. Data Pipeline & Engineering
Handling high-dimensional video data requires efficient processing pipelines. This project involved building robust data ingestion modules:

* **Jester Dataset Processing:** Filtered and processed the massive 20BN-Jester dataset down to `7` core UI-relevant gesture classes.
* **Custom Data Ingestion Script:** Engineered an automated data bridge (`prep_custom_data.py`) to parse unstructured `.mp4` and `.mov` video files, extract sequential frames at consistent sampling rates via OpenCV, and automatically generate structured `CSV` labels for DataLoader consumption.

## 4. Model Architecture (LRCN)
The final model utilizes a hybrid **CNN-LSTM** approach:
1. **Spatial Feature Extraction:** A pre-trained `ResNet-18` backbone extracts high-level spatial feature vectors from independent frames.
2. **Temporal Sequence Modeling:** An `LSTM` network (256 hidden units) processes the sequence of spatial vectors across 8 frames to capture the temporal gradient.
3. **Classification Head:** A fully connected layer with aggressive Dropout (`p=0.5`) outputs the final classification probabilities.

---
